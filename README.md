# EquiformerV2: Improved Equivariant Transformer for Scaling to Higher-Degree Representations

This repository is the extension of the [EquiformerV2](https://github.com/atomicarchitects/equiformer) repository.

## Environment Setup ##


### Environment 

We use conda to install required packages:
```
    conda env create -f env/env.yml
```

We activate the environment:
```
    conda activate equiformer_v2
```

Finally, we install `fairchem` by running:
```
    cd fairchem
    pip install -e .
```


### OC20

The OC20 S2EF dataset can be downloaded by following instructions in their [GitHub repository](https://github.com/Open-Catalyst-Project/ocp/blob/5a7738f9aa80b1a9a7e0ca15e33938b4d2557edd/DATASET.md#download-and-preprocess-the-dataset).

For example, we can download the OC20 S2EF-2M dataset of O absorbates by running:
```
    cd fairchem
    python scripts/download_data_Oabs.py --task is2re --split "Oabs" --num-workers 8 --ref-energy
```

After downloading, place the datasets under `datasets/oc20/` by using `ln -s`:
```
    cd datasets
    mkdir oc20
    cd oc20
    ln -s ../../fairchem/data/Oabs Oabs
```


## Inference ##


### OC20

1. We predict the energy and embedding of EquiformerV2 on the OC20 **is2re** dataset of O absorbates by running:
    
    ```bash
        sh scripts/train/oc20/s2ef/equiformer_v2/31M_exp.sh
        sh scripts/train/oc20/s2ef/equiformer_v2/153M_exp.sh
    ```

## Checkpoints ##

We provide the checkpoints of EquiformerV2 trained on S2EF-2M dataset for 30 epochs, EquiformerV2 (31M) trained on S2EF-All+MD, and EquiformerV2 (153M) trained on S2EF-All+MD.
|Model	|Split	|Download	|val force MAE (meV / Å) |val energy MAE (meV) |
|---	|---	|---	|---	|---	| 
|EquiformerV2	|2M	|[checkpoint](https://dl.fbaipublicfiles.com/opencatalystproject/models/2023_06/oc20/s2ef/eq2_83M_2M.pt) \| [config](oc20/configs/s2ef/2M/equiformer_v2/equiformer_v2_N@12_L@6_M@2_epochs@30.yml)	|19.4 | 278 |
|EquiformerV2 (31M)|All+MD |[checkpoint](https://dl.fbaipublicfiles.com/opencatalystproject/models/2023_06/oc20/s2ef/eq2_31M_ec4_allmd.pt) \| [config](oc20/configs/s2ef/all_md/equiformer_v2/equiformer_v2_N@8_L@4_M@2_31M.yml) |16.3 | 232 |
|EquiformerV2 (153M) |All+MD | [checkpoint](https://dl.fbaipublicfiles.com/opencatalystproject/models/2023_06/oc20/s2ef/eq2_153M_ec4_allmd.pt) \| [config](oc20/configs/s2ef/all_md/equiformer_v2/equiformer_v2_N@20_L@6_M@3_153M.yml) |15.0 | 227 |

