"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.


Copy force trainer from https://github.com/Open-Catalyst-Project/ocp/tree/6cd108e95f006a268f19459ca1b5ec011749da37

"""


import logging
import os
import pathlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch_geometric
from tqdm import tqdm

# Visualization Imports
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE # Import TSNE
try:
    from mpl_toolkits.mplot3d import Axes3D
    MPL_3D_AVAILABLE = True
except ImportError:
    Axes3D = None # Define Axes3D as None if import fails
    MPL_3D_AVAILABLE = False
    logging.warning("mpl_toolkits.mplot3d not found. 3D PCA/t-SNE plot will be disabled.")

from ocpmodels.common import distutils
from ocpmodels.common.registry import registry
from ocpmodels.common.relaxation.ml_relaxation import ml_relax
from ocpmodels.common.utils import check_traj_files
from ocpmodels.modules.evaluator import Evaluator
from ocpmodels.modules.normalizer import Normalizer
#from ocpmodels.trainers.base_trainer import BaseTrainer
from .base_trainer_v2 import BaseTrainerV2
from .engine import AverageMeter


@registry.register_trainer("forces_v2")
class ForcesTrainerV2(BaseTrainerV2):
    """
    Trainer class for the Structure to Energy & Force (S2EF) and Initial State to
    Relaxed State (IS2RS) tasks.

    .. note::

        Examples of configurations for task, model, dataset and optimizer
        can be found in `configs/ocp_s2ef <https://github.com/Open-Catalyst-Project/baselines/tree/master/configs/ocp_is2re/>`_
        and `configs/ocp_is2rs <https://github.com/Open-Catalyst-Project/baselines/tree/master/configs/ocp_is2rs/>`_.

    Args:
        task (dict): Task configuration.
        model (dict): Model configuration.
        dataset (dict): Dataset configuration. The dataset needs to be a SinglePointLMDB dataset.
        optimizer (dict): Optimizer configuration.
        identifier (str): Experiment identifier that is appended to log directory.
        run_dir (str, optional): Path to the run directory where logs are to be saved.
            (default: :obj:`None`)
        is_debug (bool, optional): Run in debug mode.
            (default: :obj:`False`)
        is_hpo (bool, optional): Run hyperparameter optimization with Ray Tune.
            (default: :obj:`False`)
        print_every (int, optional): Frequency of printing logs.
            (default: :obj:`100`)
        seed (int, optional): Random number seed.
            (default: :obj:`None`)
        logger (str, optional): Type of logger to be used.
            (default: :obj:`tensorboard`)
        local_rank (int, optional): Local rank of the process, only applicable for distributed training.
            (default: :obj:`0`)
        amp (bool, optional): Run using automatic mixed precision.
            (default: :obj:`False`)
        slurm (dict): Slurm configuration. Currently just for keeping track.
            (default: :obj:`{}`)
    """

    def __init__(
        self,
        task,
        model,
        dataset,
        optimizer,
        identifier,
        normalizer=None,
        timestamp_id=None,
        run_dir=None,
        is_debug=False,
        is_hpo=False,
        print_every=100,
        seed=None,
        logger="tensorboard",
        local_rank=0,
        amp=False,
        cpu=False,
        slurm={},
        noddp=False,
    ):
        super().__init__(
            task=task,
            model=model,
            dataset=dataset,
            optimizer=optimizer,
            identifier=identifier,
            normalizer=normalizer,
            timestamp_id=timestamp_id,
            run_dir=run_dir,
            is_debug=is_debug,
            is_hpo=is_hpo,
            print_every=print_every,
            seed=seed,
            logger=logger,
            local_rank=local_rank,
            amp=amp,
            cpu=cpu,
            name="s2ef",
            slurm=slurm,
            noddp=noddp,
        )

        # --- MPFlow Visualization Configuration ---
        self.visualize_mpflow = self.config.get("visualize_mpflow", True) # Enable/disable visualization
        self.viz_output_dir = self.config.get("visualization_output_dir", os.path.join(self.run_dir, "mpflow_visualizations")) # Default to run_dir subdir
        self.viz_num_samples = self.config.get("visualization_num_samples", 100) # Max samples to plot
        self.viz_plots = self.config.get("visualization_plots", ["2d", "3d"]) # Plots to generate ('2d', '3d')
        self.viz_pca_dpi = self.config.get("visualization_pca_dpi", 300) # DPI for saved plots

        if self.visualize_mpflow and distutils.is_master():
            os.makedirs(self.viz_output_dir, exist_ok=True)
            if hasattr(self, 'file_logger') and self.file_logger:
                self.file_logger.info(f"MPFlow visualizations will be saved to: {self.viz_output_dir}")
            else:
                # Fallback to print if file_logger is not available, though it should be from BaseTrainerV2
                print(f"MPFlow visualizations will be saved to: {self.viz_output_dir}")

    def load_task(self):
        self.file_logger.info(f"Loading dataset: {self.config['task']['dataset']}")

        if "relax_dataset" in self.config["task"]:

            self.relax_dataset = registry.get_dataset_class(self.config["task"]["dataset"])(
                self.config["task"]["relax_dataset"]
            )
            self.relax_sampler = self.get_sampler(
                self.relax_dataset,
                self.config["optim"].get(
                    "eval_batch_size", self.config["optim"]["batch_size"]
                ),
                shuffle=False,
            )
            self.relax_loader = self.get_dataloader(
                self.relax_dataset,
                self.relax_sampler,
            )

        self.num_targets = 1

        # If we're computing gradients wrt input, set mean of normalizer to 0 --
        # since it is lost when compute dy / dx -- and std to forward target std
        if (self.config["model_attributes"].get("regress_forces", True)
            or self.config['model_attributes'].get('use_auxiliary_task', False)):
            if self.normalizer.get("normalize_labels", False):
                if "grad_target_mean" in self.normalizer:
                    self.normalizers["grad_target"] = Normalizer(
                        mean=self.normalizer["grad_target_mean"],
                        std=self.normalizer["grad_target_std"],
                        device=self.device,
                    )
                else:
                    self.normalizers["grad_target"] = Normalizer(
                        tensor=self.train_loader.dataset.data.y[
                            self.train_loader.dataset.__indices__
                        ],
                        device=self.device,
                    )
                    self.normalizers["grad_target"].mean.fill_(0)

    # Takes in a new data source and generates predictions on it.
    @torch.no_grad()
    def predict(
        self,
        data_loader,
        per_image=True,
        results_file=None,
        disable_tqdm=False,
    ):
        if per_image:
            self.file_logger.info("Predicting on test.")
        assert isinstance(
            data_loader,
            (
                torch.utils.data.dataloader.DataLoader,
                torch_geometric.data.Batch,
            ),
        )
        rank = distutils.get_rank()

        if isinstance(data_loader, torch_geometric.data.Batch):
            data_loader = [[data_loader]]

        self.model.eval()
        if self.ema:
            self.ema.store()
            self.ema.copy_to()

        if self.normalizers is not None and "target" in self.normalizers:
            self.normalizers["target"].to(self.device)
            self.normalizers["grad_target"].to(self.device)

        predictions = {"id": [], "energy": [], "forces": [], "chunk_idx": []}

        for i, batch_list in tqdm(
            enumerate(data_loader),
            total=len(data_loader),
            position=rank,
            desc="device {}".format(rank),
            disable=disable_tqdm,
        ):
            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                out = self._forward(batch_list)

            if self.normalizers is not None and "target" in self.normalizers:
                out["energy"] = self.normalizers["target"].denorm(
                    out["energy"]
                )
                out["forces"] = self.normalizers["grad_target"].denorm(
                    out["forces"]
                )
            if per_image:
                systemids = [
                    str(i) + "_" + str(j)
                    for i, j in zip(
                        batch_list[0].sid.tolist(), batch_list[0].fid.tolist()
                    )
                ]
                predictions["id"].extend(systemids)
                predictions["energy"].extend(
                    out["energy"].to(torch.float16).tolist()
                )
                batch_natoms = torch.cat(
                    [batch.natoms for batch in batch_list]
                )
                batch_fixed = torch.cat([batch.fixed for batch in batch_list])
                forces = out["forces"].cpu().detach().to(torch.float16)
                per_image_forces = torch.split(forces, batch_natoms.tolist())
                per_image_forces = [
                    force.numpy() for force in per_image_forces
                ]
                # evalAI only requires forces on free atoms
                if results_file is not None:
                    _per_image_fixed = torch.split(
                        batch_fixed, batch_natoms.tolist()
                    )
                    _per_image_free_forces = [
                        force[(fixed == 0).tolist()]
                        for force, fixed in zip(
                            per_image_forces, _per_image_fixed
                        )
                    ]
                    _chunk_idx = np.array(
                        [
                            free_force.shape[0]
                            for free_force in _per_image_free_forces
                        ]
                    )
                    per_image_forces = _per_image_free_forces
                    predictions["chunk_idx"].extend(_chunk_idx)
                predictions["forces"].extend(per_image_forces)
            else:
                predictions["energy"] = out["energy"].detach()
                predictions["forces"] = out["forces"].detach()
                if self.ema:
                    self.ema.restore()
                return predictions

        predictions["forces"] = np.array(predictions["forces"])
        predictions["chunk_idx"] = np.array(predictions["chunk_idx"])
        predictions["energy"] = np.array(predictions["energy"])
        predictions["id"] = np.array(predictions["id"])
        self.save_results(
            predictions, results_file, keys=["energy", "forces", "chunk_idx"]
        )

        if self.ema:
            self.ema.restore()

        return predictions

    def update_best(
        self,
        primary_metric,
        val_metrics,
        disable_eval_tqdm=True,
    ):
        if (
            "mae" in primary_metric
            and val_metrics[primary_metric]["metric"] < self.best_val_metric
        ) or (
            "mae" not in primary_metric
            and val_metrics[primary_metric]["metric"] > self.best_val_metric
        ):
            self.best_val_metric = val_metrics[primary_metric]["metric"]
            self.save(
                metrics=val_metrics,
                checkpoint_file="best_checkpoint.pt",
                training_state=False,
            )
            if self.test_loader is not None:
                self.predict(
                    self.test_loader,
                    results_file="predictions",
                    disable_tqdm=disable_eval_tqdm,
                )

    def train(self, disable_eval_tqdm=False):
        eval_every = self.config["optim"].get(
            "eval_every", len(self.train_loader)
        )
        checkpoint_every = self.config["optim"].get(
            "checkpoint_every", eval_every
        )
        primary_metric = self.config["task"].get(
            "primary_metric", self.evaluator.task_primary_metric[self.name]
        )
        if (
            not hasattr(self, "primary_metric")
            or self.primary_metric != primary_metric
        ):
            self.best_val_metric = 1e9 if "mae" in primary_metric else -1.0
        else:
            primary_metric = self.primary_metric
        self.metrics = {}

        # Calculate start_epoch from step instead of loading the epoch number
        # to prevent inconsistencies due to different batch size in checkpoint.
        start_epoch = self.step // len(self.train_loader)

        for epoch_int in range(
            start_epoch, self.config["optim"]["max_epochs"]
        ):
            self.train_sampler.set_epoch(epoch_int)
            skip_steps = self.step % len(self.train_loader)
            train_loader_iter = iter(self.train_loader)

            self.metrics = {}

            for i in range(skip_steps, len(self.train_loader)):
                self.epoch = epoch_int + (i + 1) / len(self.train_loader)
                self.step = epoch_int * len(self.train_loader) + i + 1
                self.model.train()

                # Get a batch.
                batch = next(train_loader_iter)

                # Forward, loss, backward.
                with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                    out = self._forward(batch)
                    loss = self._compute_loss(out, batch)
                loss = self.scaler.scale(loss) if self.scaler else loss
                if self.grad_accumulation_steps != 1:
                    loss = loss / self.grad_accumulation_steps
                self._backward(loss)
                scale = self.scaler.get_scale() if self.scaler else 1.0

                # Compute metrics.
                self.metrics = self._compute_metrics(
                    out,
                    batch,
                    self.evaluator,
                    self.metrics,
                )
                self.metrics = self.evaluator.update(
                    "loss", loss.item() / scale * self.grad_accumulation_steps, self.metrics
                )

                # Log metrics.
                log_dict = {k: self.metrics[k]["metric"] for k in self.metrics}
                log_dict.update(
                    {
                        "lr_energy": self.scheduler.get_lr('energy'),
                        "lr_force": self.scheduler.get_lr('force'),
                        "lr_mpflow": self.scheduler.get_lr('mpflow'),
                        "epoch": self.epoch,
                        "step": self.step,
                    }
                )
                if (
                    (self.step % self.config["cmd"]["print_every"] == 0
                        or i == 0
                        or i == (len(self.train_loader) - 1))
                    and distutils.is_master()
                    and not self.is_hpo
                ):
                    log_str = [
                        "{}: {:.2e}".format(k, v) for k, v in log_dict.items()
                    ]
                    self.file_logger.info(", ".join(log_str))
                    # self.metrics = {}

                if self.logger is not None:
                    self.logger.log(
                        log_dict,
                        step=self.step,
                        split="train",
                    )

                if (
                    (checkpoint_every != -1
                    and self.step % checkpoint_every == 0)
                    or i == (len(self.train_loader) - 1)
                ):
                    self.save(
                        checkpoint_file="checkpoint.pt", training_state=True
                    )

                # Evaluate on val set every `eval_every` iterations.
                if (self.step % eval_every == 0
                    or i == (len(self.train_loader) - 1)):
                    if self.val_loader is not None:
                        if self.ema:
                            val_metrics = self.validate(split="val",
                                disable_tqdm=disable_eval_tqdm, use_ema=True)
                            self.update_best(primary_metric,
                                val_metrics, disable_eval_tqdm=disable_eval_tqdm)
                        else:
                            val_metrics = self.validate(split="val",
                                disable_tqdm=disable_eval_tqdm, use_ema=False)
                            self.update_best(primary_metric,
                                val_metrics, disable_eval_tqdm=disable_eval_tqdm)

                        if self.is_hpo:
                            self.hpo_update(
                                self.epoch,
                                self.step,
                                self.metrics,
                                val_metrics,
                            )

                    if self.config["task"].get("eval_relaxations", False):
                        if "relax_dataset" not in self.config["task"]:
                            logging.warning(
                                "Cannot evaluate relaxations, relax_dataset not specified"
                            )
                        else:
                            self.run_relaxations()

                if self.scheduler.scheduler_type == "ReduceLROnPlateau":
                    if self.step % eval_every == 0:
                        self.scheduler.step(
                            metrics=val_metrics[primary_metric]["metric"],
                        )
                else:
                    if self.grad_accumulation_steps != 1:
                        if self.step % self.grad_accumulation_steps == 0:
                            self.scheduler.step()
                    else:
                        self.scheduler.step()

            torch.cuda.empty_cache()

            if checkpoint_every == -1:
                self.save(checkpoint_file="checkpoint.pt", training_state=True)

        self.train_dataset.close_db()
        if self.config.get("val_dataset", False):
            self.val_dataset.close_db()
        if self.config.get("test_dataset", False):
            self.test_dataset.close_db()

    def _forward(self, batch_list):
        # forward pass.
        if (self.config["model_attributes"].get("regress_forces", True)
            or self.config['model_attributes'].get('use_auxiliary_task', False)):
            out_energy, out_forces, out_grad_forces, ut, predicted_ut, x0_embedding, x1_embedding, predicted_x1_embedding, node_energy, node_energy2 = self.model(batch_list)
        else:
            out_energy, ut, predicted_ut, x0_embedding, x1_embedding, predicted_x1_embedding, node_energy, node_energy2 = self.model(batch_list)

        if out_energy.shape[-1] == 1:
            out_energy = out_energy.view(-1)

        out = {
            "energy": out_energy,
            "ut": ut,
            "predicted_ut": predicted_ut,
            "x0": x0_embedding,
            "x1": x1_embedding,
            "predicted_x1": predicted_x1_embedding,
            "node_energy": node_energy,
            "node_energy2": node_energy2,
        }

        if (self.config["model_attributes"].get("regress_forces", True)
           or self.config['model_attributes'].get('use_auxiliary_task', False)):
            out["forces"] = out_forces
            out["grad_forces"] = out_grad_forces
            
        return out

    def _compute_loss(self, out, batch_list):
        loss = []
        
        # # MPFlow loss.
        # mpflow_mult = self.config["optim"].get("mpflow_coefficient", 100)
        # mpflow_loss = self.loss_fn["mpflow"](out["predicted_ut"], out["ut"])
        # loss.append(
        #     mpflow_mult * mpflow_loss
        # )
        
        # n2n loss.
        n2n_mult = self.config["optim"].get("n2n_coefficient", 100)
        n2n_loss = self.loss_fn["mpflow"](out["predicted_x1"], out["x1"])
        loss.append(
            n2n_mult * n2n_loss
        )
        
        # Energy loss.
        energy_target = torch.cat(
            [batch.y.to(self.device) for batch in batch_list], dim=0
        )
        if self.normalizer.get("normalize_labels", False):
            energy_target = self.normalizers["target"].norm (energy_target)
        energy_mult = self.config["optim"].get("energy_coefficient", 4)
        loss.append(
            energy_mult * self.loss_fn["energy"](out["energy"], energy_target)
        )
        
        # Per-atom energy loss.
        if out["node_energy"] is not None:
            energy_mult2 = self.config["optim"].get("energy_coefficient", 4)
            loss.append(
                energy_mult2 * self.loss_fn["mpflow"](out["node_energy"], out["node_energy2"])
            )

        # Force loss.
        if (self.config["model_attributes"].get("regress_forces", True)
        or self.config['model_attributes'].get('use_auxiliary_task', False)):
            force_target = torch.cat(
                [batch.force.to(self.device) for batch in batch_list], dim=0
            )
            if self.normalizer.get("normalize_labels", False):
                force_target = self.normalizers["grad_target"].norm(
                    force_target
                )

            tag_specific_weights = self.config["task"].get(
                "tag_specific_weights", []
            )
            if tag_specific_weights != []:
                # handle tag specific weights as introduced in forcenet
                assert len(tag_specific_weights) == 3

                batch_tags = torch.cat(
                    [
                        batch.tags.float().to(self.device)
                        for batch in batch_list
                    ],
                    dim=0,
                )
                weight = torch.zeros_like(batch_tags)
                weight[batch_tags == 0] = tag_specific_weights[0]
                weight[batch_tags == 1] = tag_specific_weights[1]
                weight[batch_tags == 2] = tag_specific_weights[2]

                loss_force_list = torch.abs(out["forces"] - force_target)
                train_loss_force_unnormalized = torch.sum(
                    loss_force_list * weight.view(-1, 1)
                )
                train_loss_force_normalizer = 3.0 * weight.sum()

                # add up normalizer to obtain global normalizer
                distutils.all_reduce(train_loss_force_normalizer)

                # perform loss normalization before backprop
                train_loss_force_normalized = train_loss_force_unnormalized * (
                    distutils.get_world_size() / train_loss_force_normalizer
                )
                loss.append(train_loss_force_normalized)

            else:
                # Force coefficient = 30 has been working well for us.
                force_mult = self.config["optim"].get("force_coefficient", 100)
                if self.config["task"].get("train_on_free_atoms", False):
                    fixed = torch.cat(
                        [batch.fixed.to(self.device) for batch in batch_list]
                    )
                    mask = fixed == 0
                    if self.config["optim"]["loss_force"].startswith(
                        "atomwise"
                    ):
                        force_mult = self.config["optim"].get(
                            "force_coefficient", 100
                        )
                        natoms = torch.cat(
                            [
                                batch.natoms.to(self.device)
                                for batch in batch_list
                            ]
                        )
                        natoms = torch.repeat_interleave(natoms, natoms)
                        force_loss = force_mult * self.loss_fn["force"](
                            out["forces"][mask],
                            force_target[mask],
                            natoms=natoms[mask],
                            batch_size=batch_list[0].natoms.shape[0],
                        )
                        loss.append(force_loss)
                    else:
                        loss.append(
                            force_mult
                            * self.loss_fn["force"](
                                out["forces"][mask], force_target[mask]
                            )
                        )
                        
                        # # Gradient loss.
                        # if out["grad_forces"] is not None:
                        #     loss.append(
                        #         energy_mult
                        #         * self.loss_fn["force"](out["grad_forces"][mask], force_target[mask])
                        #     )
                else:
                    loss.append(
                        force_mult
                        * self.loss_fn["force"](out["forces"], force_target)
                    )

        # Sanity check to make sure the compute graph is correct.
        for lc in loss:
            assert hasattr(lc, "grad_fn")

        loss = sum(loss)
        return loss

    def _compute_metrics(self, out, batch_list, evaluator, metrics={}):
        natoms = torch.cat(
            [batch.natoms.to(self.device) for batch in batch_list], dim=0
        )

        target = {
            "energy": torch.cat(
                [batch.y.to(self.device) for batch in batch_list], dim=0
            ),
            "forces": torch.cat(
                [batch.force.to(self.device) for batch in batch_list], dim=0
            ),
            "natoms": natoms,
            "ut": out["ut"],
            "predicted_ut": out["predicted_ut"],
            "x1": out["x1"],
            "predicted_x1": out["predicted_x1"],
        }

        out["natoms"] = natoms

        if self.config["task"].get("eval_on_free_atoms", True):
            fixed = torch.cat(
                [batch.fixed.to(self.device) for batch in batch_list]
            )
            mask = fixed == 0
            out["forces"] = out["forces"][mask]
            target["forces"] = target["forces"][mask]

            s_idx = 0
            natoms_free = []
            for natoms in target["natoms"]:
                natoms_free.append(
                    torch.sum(mask[s_idx : s_idx + natoms]).item()
                )
                s_idx += natoms
            target["natoms"] = torch.LongTensor(natoms_free).to(self.device)
            out["natoms"] = torch.LongTensor(natoms_free).to(self.device)

        if self.normalizer.get("normalize_labels", False):
            out["energy"] = self.normalizers["target"].denorm(out["energy"])
            out["forces"] = self.normalizers["grad_target"].denorm(
                out["forces"]
            )

        metrics = evaluator.eval(out, target, prev_metrics=metrics)
        return metrics

    def run_relaxations(self, split="val"):
        self.file_logger.info("Running ML-relaxations")
        self.model.eval()
        if self.ema:
            self.ema.store()
            self.ema.copy_to()

        evaluator_is2rs, metrics_is2rs = Evaluator(task="is2rs"), {}
        evaluator_is2re, metrics_is2re = Evaluator(task="is2re"), {}

        if hasattr(self.relax_dataset[0], "pos_relaxed") and hasattr(
            self.relax_dataset[0], "y_relaxed"
        ):
            split = "val"
        else:
            split = "test"

        # load IS2RS pos predictions
        pred_pos_dict = None
        if self.config["task"]["relax_opt"].get("pred_pos_path", None):
            pred_pos_dict = torch.load(self.config["task"]["relax_opt"]["pred_pos_path"])

        ids = []
        relaxed_positions = []
        chunk_idx = []
        for i, batch in tqdm(
            enumerate(self.relax_loader), total=len(self.relax_loader)
        ):
            if i >= self.config["task"].get("num_relaxation_batches", 1e9):
                break

            # If all traj files already exist, then skip this batch
            if check_traj_files(
                batch, self.config["task"]["relax_opt"].get("traj_dir", None)
            ):
                self.file_logger.info(f"Skipping batch: {batch[0].sid.tolist()}")
                continue

            # Initailize pos with IS2RS direct prediction
            if pred_pos_dict is not None:
                sid_list = batch[0].sid.tolist()
                pred_pos_list = []
                for sid in sid_list:
                    pred_pos_list.append(pred_pos_dict[str(sid)])
                pred_pos = torch.cat(pred_pos_list, dim=0)
                batch[0].pos = pred_pos

            relaxed_batch = ml_relax(
                batch=batch,
                model=self,
                steps=self.config["task"].get("relaxation_steps", 200),
                fmax=self.config["task"].get("relaxation_fmax", 0.0),
                relax_opt=self.config["task"]["relax_opt"],
                save_full_traj=True,
                device=self.device,
                transform=None,
            )

            if self.config["task"].get("write_pos", False):
                systemids = [str(i) for i in relaxed_batch.sid.tolist()]
                natoms = relaxed_batch.natoms.tolist()
                positions = torch.split(relaxed_batch.pos, natoms)
                batch_relaxed_positions = [pos.tolist() for pos in positions]

                relaxed_positions += batch_relaxed_positions
                chunk_idx += natoms
                ids += systemids

            if split == "val":
                mask = relaxed_batch.fixed == 0
                s_idx = 0
                natoms_free = []
                for natoms in relaxed_batch.natoms:
                    natoms_free.append(
                        torch.sum(mask[s_idx : s_idx + natoms]).item()
                    )
                    s_idx += natoms

                target = {
                    "energy": relaxed_batch.y_relaxed,
                    "positions": relaxed_batch.pos_relaxed[mask],
                    "cell": relaxed_batch.cell,
                    "pbc": torch.tensor([True, True, True]),
                    "natoms": torch.LongTensor(natoms_free),
                }

                prediction = {
                    "energy": relaxed_batch.y,
                    "positions": relaxed_batch.pos[mask],
                    "cell": relaxed_batch.cell,
                    "pbc": torch.tensor([True, True, True]),
                    "natoms": torch.LongTensor(natoms_free),
                }

                metrics_is2rs = evaluator_is2rs.eval(
                    prediction,
                    target,
                    metrics_is2rs,
                )
                metrics_is2re = evaluator_is2re.eval(
                    {"energy": prediction["energy"]},
                    {"energy": target["energy"]},
                    metrics_is2re,
                )

                # Log metrics.
                #log_dict = {k: metrics_is2re[k]["metric"] for k in metrics_is2re}
                #log_dict.update({k: metrics_is2rs[k]["metric"] for k in metrics_is2rs})
                if (
                    (   (i + 1) % self.config["cmd"]["print_every"] == 0
                        or i == 0
                        or i == (len(self.relax_loader) - 1)
                    )
                ):
                    distutils.synchronize()
                    log_dict = {}
                    for task in ["is2rs", "is2re"]:
                        metrics = eval(f"metrics_{task}")
                        aggregated_metrics = {}
                        for k in metrics:
                            aggregated_metrics[k] = {
                                "total": distutils.all_reduce(
                                    metrics[k]["total"],
                                    average=False,
                                    device=self.device,
                                    ),
                                "numel": distutils.all_reduce(
                                    metrics[k]["numel"],
                                    average=False,
                                    device=self.device,
                                    ),
                                }
                            aggregated_metrics[k]["metric"] = aggregated_metrics[k]["total"] / aggregated_metrics[k]["numel"]
                            log_dict[k] = aggregated_metrics[k]["metric"]
                    log_str = ["{}: {:.2e}".format(k, v) for k, v in log_dict.items()]
                    log_str = ", ".join(log_str)
                    self.file_logger.info('[{}/{}] {}'.format(i, len(self.relax_loader), log_str))

        if self.config["task"].get("write_pos", False):
            rank = distutils.get_rank()
            pos_filename = os.path.join(
                self.config["cmd"]["results_dir"], f"relaxed_pos_{rank}.npz"
            )
            np.savez_compressed(
                pos_filename,
                ids=ids,
                pos=np.array(relaxed_positions, dtype=object),
                chunk_idx=chunk_idx,
            )

            distutils.synchronize()
            if distutils.is_master():
                gather_results = defaultdict(list)
                full_path = os.path.join(
                    self.config["cmd"]["results_dir"],
                    "relaxed_positions.npz",
                )

                for i in range(distutils.get_world_size()):
                    rank_path = os.path.join(
                        self.config["cmd"]["results_dir"],
                        f"relaxed_pos_{i}.npz",
                    )
                    rank_results = np.load(rank_path, allow_pickle=True)
                    gather_results["ids"].extend(rank_results["ids"])
                    gather_results["pos"].extend(rank_results["pos"])
                    gather_results["chunk_idx"].extend(
                        rank_results["chunk_idx"]
                    )
                    os.remove(rank_path)

                # Because of how distributed sampler works, some system ids
                # might be repeated to make no. of samples even across GPUs.
                _, idx = np.unique(gather_results["ids"], return_index=True)
                gather_results["ids"] = np.array(gather_results["ids"])[idx]
                gather_results["pos"] = np.concatenate(
                    np.array(gather_results["pos"])[idx]
                )
                gather_results["chunk_idx"] = np.cumsum(
                    np.array(gather_results["chunk_idx"])[idx]
                )[
                    :-1
                ]  # np.split does not need last idx, assumes n-1:end

                self.file_logger.info(f"Writing results to {full_path}")
                np.savez_compressed(full_path, **gather_results)

        if split == "val":
            for task in ["is2rs", "is2re"]:
                metrics = eval(f"metrics_{task}")
                aggregated_metrics = {}
                for k in metrics:
                    aggregated_metrics[k] = {
                        "total": distutils.all_reduce(
                            metrics[k]["total"],
                            average=False,
                            device=self.device,
                        ),
                        "numel": distutils.all_reduce(
                            metrics[k]["numel"],
                            average=False,
                            device=self.device,
                        ),
                    }
                    aggregated_metrics[k]["metric"] = (
                        aggregated_metrics[k]["total"]
                        / aggregated_metrics[k]["numel"]
                    )
                metrics = aggregated_metrics

                # Make plots.
                log_dict = {
                    f"{task}_{k}": metrics[k]["metric"] for k in metrics
                }
                if self.logger is not None:
                    self.logger.log(
                        log_dict,
                        step=self.step,
                        split=split,
                    )

                if distutils.is_master():
                    self.file_logger.info(metrics)

        if self.ema:
            self.ema.restore()


    @torch.no_grad()
    def validate(self, split="val", disable_tqdm=False, use_ema=False):
        self.file_logger.info(f"Evaluating on {split}.")

        if self.is_hpo:
            disable_tqdm = True

        self.model.eval()
        if self.ema and use_ema:
            self.ema.store()
            self.ema.copy_to()

        evaluator, metrics = Evaluator(task=self.name), {}
        rank = distutils.get_rank()

        loader = self.val_loader if split == "val" else self.test_loader
        
        # Data collection for visualization (only on master)
        viz_data = {"x0": [], "x1": [], "predicted_x1": []}
        samples_collected = 0
        
        for i, batch in tqdm(
            enumerate(loader),
            total=len(loader),
            position=rank,
            desc="device {}".format(rank),
            disable=disable_tqdm,
        ):
            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                out = self._forward(batch)
            loss = self._compute_loss(out, batch)
            metrics = self._compute_metrics(out, batch, evaluator, metrics)
            metrics = evaluator.update("loss", loss.item(), metrics)
            
            # Collect visualization data on master process
            if distutils.is_master() and self.visualize_mpflow and samples_collected < self.viz_num_samples:
                needed = self.viz_num_samples - samples_collected
                
                # Ensure tensors are on CPU for storage
                x0_batch = out["x0"].detach().cpu()[:needed]
                x1_batch = out["x1"].detach().cpu()[:needed]
                predicted_x1_batch = out["predicted_x1"].detach().cpu()[:needed]
                
                viz_data["x0"].append(x0_batch)
                viz_data["x1"].append(x1_batch)
                viz_data["predicted_x1"].append(predicted_x1_batch)
                samples_collected += len(x0_batch)

        aggregated_metrics = {}
        for k in metrics:
            aggregated_metrics[k] = {
                "total": distutils.all_reduce(
                    metrics[k]["total"], average=False, device=self.device
                ),
                "numel": distutils.all_reduce(
                    metrics[k]["numel"], average=False, device=self.device
                ),
            }
            aggregated_metrics[k]["metric"] = (
                aggregated_metrics[k]["total"] / aggregated_metrics[k]["numel"]
            )
        metrics = aggregated_metrics

        log_dict = {k: metrics[k]["metric"] for k in metrics}
        log_dict.update({"epoch": self.epoch})
        log_str = ["{}: {:.4f}".format(k, v) for k, v in log_dict.items()]
        log_str = ", ".join(log_str)
        log_str = "[{}] ".format(split) + log_str
        self.file_logger.info(log_str)

        if self.logger is not None:
            self.logger.log(
                log_dict,
                step=self.step,
                split=split,
            )

        # Call visualization function on master process if enabled and data collected
        if distutils.is_master() and self.visualize_mpflow and samples_collected > 0:
            # Concatenate collected batches
            viz_x0 = torch.cat(viz_data["x0"], dim=0)
            viz_x1 = torch.cat(viz_data["x1"], dim=0)
            viz_predicted_x1 = torch.cat(viz_data["predicted_x1"], dim=0)
            
            self.visualize_predictions(viz_x0, viz_x1, viz_predicted_x1)
            
        if self.ema and use_ema:
            self.ema.restore()

        return metrics

    def visualize_predictions(self, x0: torch.Tensor, x1_true: torch.Tensor, x1_pred: torch.Tensor):
        """
        Visualizes the predicted MPFlow states (x1_pred) compared to the ground
        truth (x1_true), starting from x0, using t-SNE.

        Args:
            x0 (torch.Tensor): Start points tensor [batch, N, D]. Assumed on CPU.
            x1_true (torch.Tensor): Ground truth end points tensor [batch, N, D]. Assumed on CPU.
            x1_pred (torch.Tensor): Predicted end points tensor [batch, N, D]. Assumed on CPU.
        """
        n_samples = x0.shape[0]
        if n_samples == 0:
            self.file_logger.warning("No samples provided for MPFlow visualization.")
            return

        self.file_logger.info(f"Generating MPFlow prediction visualizations for {n_samples} samples using t-SNE...")

        # --- Define t-SNE plotting helper nested within visualize_predictions ---
        # This keeps the t-SNE logic contained and avoids polluting the class namespace
        def _create_tsne_plot(
            n_components: int,
            x0_tsne_in: np.ndarray,
            x1_true_tsne_in: np.ndarray,
            x1_pred_tsne_in: np.ndarray,
            output_dir: str,
            plot_suffix: str = ""
        ):
            # Create plot
            fig = plt.figure(figsize=(14, 12) if n_components == 3 else (12, 10))
            ax = fig.add_subplot(111, projection='3d' if n_components == 3 else None)
            # Use plasma colormap and ensure n_samples is at least 1 for linspace
            colors = plt.cm.plasma(np.linspace(0, 1, max(1, n_samples)))

            for i in range(n_samples):
                # Get t-SNE coordinates for sample i
                start_coords = x0_tsne_in[i]
                true_end_coords = x1_true_tsne_in[i]
                pred_end_coords = x1_pred_tsne_in[i]
                color = colors[i] # Get color for this sample

                if n_components == 2:
                    # Plot lines: start -> true_end (dashed), start -> pred_end (solid)
                    ax.plot([start_coords[0], true_end_coords[0]], [start_coords[1], true_end_coords[1]],
                            '--', color=color, alpha=0.5, linewidth=1.2, label='True Flow' if i==0 else "") # Slightly thinner dashed line
                    ax.plot([start_coords[0], pred_end_coords[0]], [start_coords[1], pred_end_coords[1]],
                            '-', color=color, alpha=0.7, linewidth=1.5, label='Predicted Flow' if i==0 else "") # Solid line
                    # Mark points with edges and adjusted sizes/alpha
                    ax.scatter(start_coords[0], start_coords[1], color='blue', s=45, marker='o', alpha=0.8, edgecolors='k', linewidths=0.5, label='Start (x0)' if i==0 else "")
                    ax.scatter(true_end_coords[0], true_end_coords[1], color='green', s=60, marker='x', alpha=0.9, linewidths=0.5, label='True End (x1)' if i==0 else "")
                    ax.scatter(pred_end_coords[0], pred_end_coords[1], color='red', s=60, marker='+', alpha=0.9, linewidths=0.5, label='Predicted End' if i==0 else "")
                elif MPL_3D_AVAILABLE and n_components == 3: # Only plot 3D if available and requested
                     # Plot lines
                    ax.plot([start_coords[0], true_end_coords[0]], [start_coords[1], true_end_coords[1]], [start_coords[2], true_end_coords[2]],
                            '--', color=color, alpha=0.5, linewidth=1.2, label='True Flow' if i==0 else "")
                    ax.plot([start_coords[0], pred_end_coords[0]], [start_coords[1], pred_end_coords[1]], [start_coords[2], pred_end_coords[2]],
                            '-', color=color, alpha=0.7, linewidth=1.5, label='Predicted Flow' if i==0 else "")
                     # Mark points with edges and adjusted sizes/alpha
                    ax.scatter(start_coords[0], start_coords[1], start_coords[2], color='blue', s=45, marker='o', alpha=0.8, edgecolors='k', linewidths=0.5, label='Start (x0)' if i==0 else "")
                    ax.scatter(true_end_coords[0], true_end_coords[1], true_end_coords[2], color='green', s=60, marker='x', alpha=0.9, linewidths=0.5, label='True End (x1)' if i==0 else "")
                    ax.scatter(pred_end_coords[0], pred_end_coords[1], pred_end_coords[2], color='red', s=60, marker='+', alpha=0.9, linewidths=0.5, label='Predicted End' if i==0 else "")

            # Titles and labels
            epoch_step_info = f"Epoch_{self.epoch:.2f}_Step_{self.step}" if hasattr(self, 'epoch') and hasattr(self, 'step') else "Unknown_Epoch_Step"
            # Ensure perplexity_value is defined before being used in the title
            # It will be defined in the outer scope before this helper is called.
            title = f'MPFlow Prediction vs Truth - {n_components}D t-SNE (Perplexity={_perplexity_value_for_plot}, {epoch_step_info})'


            ax.set_title(title, fontsize=14)
            ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
            ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
            if n_components == 3 and MPL_3D_AVAILABLE:
                ax.set_zlabel('t-SNE Dimension 3', fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.2) # Refined grid
            ax.legend(loc='best', fontsize=10) # Adjust legend font size

            # Save plot
            plot_filename = f"mpflow_tsne_{n_components}d_{epoch_step_info}{plot_suffix}.png"
            plot_path = os.path.join(self.viz_output_dir, plot_filename)
            try:
                plt.savefig(plot_path, dpi=self.viz_pca_dpi, bbox_inches='tight')
                self.file_logger.info(f"Saved {n_components}D t-SNE plot to {plot_path}")
            except IOError as e:
                self.file_logger.error(f"Failed to save {n_components}D t-SNE plot to {plot_path}: {e}")
            finally:
                plt.close(fig) # Ensure figure is closed
        # --- End of nested helper function ---

        # --- Main visualization logic ---
        embedding_dim = x0.shape[1] * x0.shape[2] # N * D

        # Flatten embedding dimensions: [batch, N*D]
        # Tensors should already be on CPU if passed correctly
        try:
            x0_flat = x0.numpy().reshape(n_samples, embedding_dim)
            x1_true_flat = x1_true.numpy().reshape(n_samples, embedding_dim)
            x1_pred_flat = x1_pred.numpy().reshape(n_samples, embedding_dim)
        except Exception as e:
            self.file_logger.error(f"Error converting tensors to NumPy for t-SNE: {e}")
            return

        # Combine all points for t-SNE fitting: [3 * batch, N*D]
        all_points_flat = np.vstack((x0_flat, x1_true_flat, x1_pred_flat))

        # --- Perform t-SNE and Plotting ---
        perplexity_value = 50 # Default perplexity
        # Adjust perplexity if it's too large for the number of samples
        # n_samples for t-SNE is the number of rows in all_points_flat
        if perplexity_value >= all_points_flat.shape[0]:
            perplexity_value = max(5.0, float(all_points_flat.shape[0] - 1)) # t-SNE perplexity must be float and < n_samples
            self.file_logger.warning(f"Perplexity adjusted to {perplexity_value} due to low sample count ({all_points_flat.shape[0]} total points for t-SNE).")

        _perplexity_value_for_plot = perplexity_value # To pass to the plotting helper

        seed = self.config.get("cmd", {}).get("seed", None) # Safely get seed

        if "2d" in self.viz_plots:
            try:
                tsne_2d = TSNE(n_components=2, perplexity=perplexity_value, random_state=seed, n_iter=300, init='pca', learning_rate='auto')
                all_points_transformed_2d = tsne_2d.fit_transform(all_points_flat)
                # Separate the transformed points
                x0_tsne_2d = all_points_transformed_2d[0*n_samples : 1*n_samples]
                x1_true_tsne_2d = all_points_transformed_2d[1*n_samples : 2*n_samples]
                x1_pred_tsne_2d = all_points_transformed_2d[2*n_samples : 3*n_samples]
                # Create plot using the helper
                _create_tsne_plot(2, x0_tsne_2d, x1_true_tsne_2d, x1_pred_tsne_2d, self.viz_output_dir)
            except Exception as e:
                self.file_logger.error(f"t-SNE or 2D plotting failed: {e}", exc_info=True) # Log traceback

        if "3d" in self.viz_plots:
            if MPL_3D_AVAILABLE:
                try:
                    tsne_3d = TSNE(n_components=3, perplexity=perplexity_value, random_state=seed, n_iter=300, init='pca', learning_rate='auto')
                    all_points_transformed_3d = tsne_3d.fit_transform(all_points_flat)
                    # Separate the transformed points
                    x0_tsne_3d = all_points_transformed_3d[0*n_samples : 1*n_samples]
                    x1_true_tsne_3d = all_points_transformed_3d[1*n_samples : 2*n_samples]
                    x1_pred_tsne_3d = all_points_transformed_3d[2*n_samples : 3*n_samples]
                    # Create plot using the helper
                    _create_tsne_plot(3, x0_tsne_3d, x1_true_tsne_3d, x1_pred_tsne_3d, self.viz_output_dir)
                except Exception as e:
                    self.file_logger.error(f"t-SNE or 3D plotting failed: {e}", exc_info=True) # Log traceback
            else:
                self.file_logger.warning("Skipping 3D t-SNE plot because mpl_toolkits.mplot3d is not available.")
