import logging
import time
import math
import numpy as np
import torch
import torch.nn as nn
from pyexpat.model import XML_CQUANT_OPT

from ocpmodels.common.registry import registry
from ocpmodels.common.utils import conditional_grad
from ocpmodels.models.base import BaseModel
from ocpmodels.models.scn.sampling import CalcSpherePoints
from ocpmodels.models.scn.smearing import (
    GaussianSmearing,
    LinearSigmoidSmearing,
    SigmoidSmearing,
    SiLUSmearing,
)

try:
    from e3nn import o3
    from torchdiffeq import odeint
except ImportError:
    print("torchdiffeq not found. Please install it: pip install torchdiffeq")
    odeint = None

from .gaussian_rbf import GaussianRadialBasisLayer
from torch.nn import Linear
from .edge_rot_mat import init_edge_rot_mat
from .so3 import (
    CoefficientMappingModule,
    SO3_Embedding,
    SO3_Grid,
    SO3_Rotation,
    SO3_LinearV2
)
from .module_list import ModuleListInfo
from .so2_ops import SO2_Convolution
from .radial_function import RadialFunction
from .layer_norm import (
    EquivariantLayerNormArray, 
    EquivariantLayerNormArraySphericalHarmonics, 
    EquivariantRMSNormArraySphericalHarmonics,
    EquivariantRMSNormArraySphericalHarmonicsV2,
    get_normalization_layer
)
from .transformer_block import (
    SO2EquivariantGraphAttention,
    FeedForwardNetwork,
    TransBlockV2, 
)
from .input_block import EdgeDegreeEmbedding
from .MPFlow import EquivariantMPFlow

# Statistics of IS2RE 100K 
_AVG_NUM_NODES  = 77.81317
_AVG_DEGREE     = 23.395238876342773    # IS2RE: 100k, max_radius = 5, max_neighbors = 100


@registry.register_model("equiformer_v2")
class EquiformerV2_OC20(BaseModel):
    """
    Equiformer with graph attention built upon SO(2) convolution and feedforward network built upon S2 activation

    Args:
        use_pbc (bool):         Use periodic boundary conditions
        regress_forces (bool):  Compute forces
        otf_graph (bool):       Compute graph On The Fly (OTF)
        max_neighbors (int):    Maximum number of neighbors per atom
        max_radius (float):     Maximum distance between nieghboring atoms in Angstroms
        max_num_elements (int): Maximum atomic number

        num_layers (int):             Number of layers in the GNN
        sphere_channels (int):        Number of spherical channels (one set per resolution)
        attn_hidden_channels (int): Number of hidden channels used during SO(2) graph attention
        num_heads (int):            Number of attention heads
        attn_alpha_head (int):      Number of channels for alpha vector in each attention head
        attn_value_head (int):      Number of channels for value vector in each attention head
        ffn_hidden_channels (int):  Number of hidden channels used during feedforward network
        norm_type (str):            Type of normalization layer (['layer_norm', 'layer_norm_sh', 'rms_norm_sh'])

        lmax_list (int):              List of maximum degree of the spherical harmonics (1 to 10)
        mmax_list (int):              List of maximum order of the spherical harmonics (0 to lmax)
        grid_resolution (int):        Resolution of SO3_Grid
        
        num_sphere_samples (int):     Number of samples used to approximate the integration of the sphere in the output blocks
        
        edge_channels (int):                Number of channels for the edge invariant features
        use_atom_edge_embedding (bool):     Whether to use atomic embedding along with relative distance for edge scalar features
        share_atom_edge_embedding (bool):   Whether to share `atom_edge_embedding` across all blocks
        use_m_share_rad (bool):             Whether all m components within a type-L vector of one channel share radial function weights
        distance_function ("gaussian", "sigmoid", "linearsigmoid", "silu"):  Basis function used for distances
        
        attn_activation (str):      Type of activation function for SO(2) graph attention
        use_s2_act_attn (bool):     Whether to use attention after S2 activation. Otherwise, use the same attention as Equiformer
        use_attn_renorm (bool):     Whether to re-normalize attention weights
        ffn_activation (str):       Type of activation function for feedforward network
        use_gate_act (bool):        If `True`, use gate activation. Otherwise, use S2 activation
        use_grid_mlp (bool):        If `True`, use projecting to grids and performing MLPs for FFNs. 
        use_sep_s2_act (bool):      If `True`, use separable S2 activation when `use_gate_act` is False.

        alpha_drop (float):         Dropout rate for attention weights
        drop_path_rate (float):     Drop path rate
        proj_drop (float):          Dropout rate for outputs of attention and FFN in Transformer blocks

        weight_init (str):          ['normal', 'uniform'] initialization of weights of linear layers except those in radial functions
    """
    def __init__(
        self,
        num_atoms,      # not used
        bond_feat_dim,  # not used
        num_targets,    # not used
        use_pbc=True,
        regress_forces=True,
        otf_graph=True,
        max_neighbors=500,
        max_radius=5.0,
        max_num_elements=90,

        num_layers=12,
        sphere_channels=128,
        attn_hidden_channels=128,
        num_heads=8,
        attn_alpha_channels=32,
        attn_value_channels=16,
        ffn_hidden_channels=512,
        
        norm_type='rms_norm_sh',
        
        lmax_list=[6],
        mmax_list=[2],
        grid_resolution=None, 

        num_sphere_samples=128,

        edge_channels=128,
        use_atom_edge_embedding=True, 
        share_atom_edge_embedding=False,
        use_m_share_rad=False,
        distance_function="gaussian",
        num_distance_basis=512, 

        attn_activation='scaled_silu',
        use_s2_act_attn=False, 
        use_attn_renorm=True,
        ffn_activation='scaled_silu',
        use_gate_act=False,
        use_grid_mlp=False, 
        use_sep_s2_act=True,

        alpha_drop=0.1,
        drop_path_rate=0.05, 
        proj_drop=0.0, 

        weight_init='normal'
    ):
        super().__init__()

        self.use_pbc = use_pbc
        self.regress_forces = regress_forces
        self.otf_graph = otf_graph
        self.max_neighbors = max_neighbors
        self.max_radius = max_radius
        self.cutoff = max_radius
        self.max_num_elements = max_num_elements

        self.num_layers = num_layers
        self.sphere_channels = sphere_channels
        self.attn_hidden_channels = attn_hidden_channels
        self.num_heads = num_heads
        self.attn_alpha_channels = attn_alpha_channels
        self.attn_value_channels = attn_value_channels
        self.ffn_hidden_channels = ffn_hidden_channels
        self.norm_type = norm_type
        
        self.lmax_list = lmax_list
        self.mmax_list = mmax_list
        self.grid_resolution = grid_resolution

        self.num_sphere_samples = num_sphere_samples

        self.edge_channels = edge_channels
        self.use_atom_edge_embedding = use_atom_edge_embedding 
        self.share_atom_edge_embedding = share_atom_edge_embedding
        if self.share_atom_edge_embedding:
            assert self.use_atom_edge_embedding
            self.block_use_atom_edge_embedding = False
        else:
            self.block_use_atom_edge_embedding = self.use_atom_edge_embedding
        self.use_m_share_rad = use_m_share_rad
        self.distance_function = distance_function
        self.num_distance_basis = num_distance_basis

        self.attn_activation = attn_activation
        self.use_s2_act_attn = use_s2_act_attn
        self.use_attn_renorm = use_attn_renorm
        self.ffn_activation = ffn_activation
        self.use_gate_act = use_gate_act
        self.use_grid_mlp = use_grid_mlp
        self.use_sep_s2_act = use_sep_s2_act
        
        self.alpha_drop = alpha_drop
        self.drop_path_rate = drop_path_rate
        self.proj_drop = proj_drop

        self.weight_init = weight_init
        assert self.weight_init in ['normal', 'uniform']

        self.device = 'cpu' #torch.cuda.current_device()

        self.grad_forces = False
        self.num_resolutions = len(self.lmax_list)
        self.sphere_channels_all = self.num_resolutions * self.sphere_channels
        
        # Weights for message initialization
        self.sphere_embedding = nn.Embedding(self.max_num_elements, self.sphere_channels_all)
        
        # Initialize the function used to measure the distances between atoms
        assert self.distance_function in [
            'gaussian',
        ]
        if self.distance_function == 'gaussian':
            self.distance_expansion = GaussianSmearing(
                0.0,
                self.cutoff,
                600,
                2.0,
            )
            #self.distance_expansion = GaussianRadialBasisLayer(num_basis=self.num_distance_basis, cutoff=self.max_radius)
        else:
            raise ValueError
        
        # Initialize the sizes of radial functions (input channels and 2 hidden channels)
        self.edge_channels_list = [int(self.distance_expansion.num_output)] + [self.edge_channels] * 2

        # Initialize atom edge embedding
        if self.share_atom_edge_embedding and self.use_atom_edge_embedding:
            self.source_embedding = nn.Embedding(self.max_num_elements, self.edge_channels_list[-1])
            self.target_embedding = nn.Embedding(self.max_num_elements, self.edge_channels_list[-1])
            self.edge_channels_list[0] = self.edge_channels_list[0] + 2 * self.edge_channels_list[-1]
        else:
            self.source_embedding, self.target_embedding = None, None
        
        # Initialize the module that compute WignerD matrices and other values for spherical harmonic calculations
        self.SO3_rotation = nn.ModuleList()
        for i in range(self.num_resolutions):
            self.SO3_rotation.append(SO3_Rotation(self.lmax_list[i]))

        # Initialize conversion between degree l and order m layouts
        self.mappingReduced = CoefficientMappingModule(self.lmax_list, self.mmax_list)

        # Initialize the transformations between spherical and grid representations
        self.SO3_grid = ModuleListInfo('({}, {})'.format(max(self.lmax_list), max(self.lmax_list)))
        for l in range(max(self.lmax_list) + 1):
            SO3_m_grid = nn.ModuleList()
            for m in range(max(self.lmax_list) + 1):
                SO3_m_grid.append(
                    SO3_Grid(
                        l, 
                        m, 
                        resolution=self.grid_resolution, 
                        normalization='component'
                    )
                )
            self.SO3_grid.append(SO3_m_grid)

        # Edge-degree embedding
        self.edge_degree_embedding = EdgeDegreeEmbedding(
            self.sphere_channels,
            self.lmax_list,
            self.mmax_list,
            self.SO3_rotation,
            self.mappingReduced,
            self.max_num_elements,
            self.edge_channels_list,
            self.block_use_atom_edge_embedding,
            rescale_factor=_AVG_DEGREE
        )

        # Initialize the blocks for each layer of EquiformerV2
        self.blocks = nn.ModuleList()
        for i in range(self.num_layers):
            block = TransBlockV2(
                self.sphere_channels,
                self.attn_hidden_channels,
                self.num_heads,
                self.attn_alpha_channels,
                self.attn_value_channels,
                self.ffn_hidden_channels,
                self.sphere_channels, 
                self.lmax_list,
                self.mmax_list,
                self.SO3_rotation,
                self.mappingReduced,
                self.SO3_grid,
                self.max_num_elements,
                self.edge_channels_list,
                self.block_use_atom_edge_embedding,
                self.use_m_share_rad,
                self.attn_activation,
                self.use_s2_act_attn,
                self.use_attn_renorm,
                self.ffn_activation,
                self.use_gate_act,
                self.use_grid_mlp,
                self.use_sep_s2_act,
                self.norm_type,
                self.alpha_drop, 
                self.drop_path_rate,
                self.proj_drop
            )
            self.blocks.append(block)
        
        self.norm = get_normalization_layer(self.norm_type, lmax=max(self.lmax_list), num_channels=self.sphere_channels)
        
        self.energy_block = FeedForwardNetwork(
            self.sphere_channels,
            self.ffn_hidden_channels, 
            1,
            self.lmax_list,
            self.mmax_list,
            self.SO3_grid,  
            self.ffn_activation,
            self.use_gate_act,
            self.use_grid_mlp,
            self.use_sep_s2_act
        )
        if self.regress_forces:
            self.force_block = SO2EquivariantGraphAttention(
                self.sphere_channels,
                self.attn_hidden_channels,
                self.num_heads, 
                self.attn_alpha_channels,
                self.attn_value_channels, 
                1,
                self.lmax_list,
                self.mmax_list,
                self.SO3_rotation, 
                self.mappingReduced, 
                self.SO3_grid, 
                self.max_num_elements,
                self.edge_channels_list,
                self.block_use_atom_edge_embedding, 
                self.use_m_share_rad,
                self.attn_activation, 
                self.use_s2_act_attn, 
                self.use_attn_renorm,
                self.use_gate_act,
                self.use_sep_s2_act,
                alpha_drop=0.0
            )
        
        # Equivariant MPFlow
        self.mpflow = EquivariantMPFlow(
            self.sphere_channels,
            self.attn_hidden_channels,
            self.num_heads,
            self.attn_alpha_channels,
            self.attn_value_channels,
            self.ffn_hidden_channels,
            self.sphere_channels, 
            self.lmax_list,
            self.mmax_list,
            self.SO3_rotation,
            self.mappingReduced,
            self.SO3_grid,
            self.max_num_elements,
            self.edge_channels_list,
            self.block_use_atom_edge_embedding,
            self.use_m_share_rad,
            self.attn_activation,
            self.use_s2_act_attn,
            self.use_attn_renorm,
            self.ffn_activation,
            self.use_gate_act,
            self.use_grid_mlp,
            self.use_sep_s2_act,
            self.norm_type,
            self.alpha_drop,
            self.drop_path_rate,
            self.proj_drop,
            num_layers=2
        )
        
        self.mpflow_delta = FeedForwardNetwork(
            self.sphere_channels,
            self.ffn_hidden_channels, 
            self.sphere_channels,
            self.lmax_list,
            self.mmax_list,
            self.SO3_grid,  
            self.ffn_activation,
            self.use_gate_act,
            self.use_grid_mlp,
            self.use_sep_s2_act
        )
        
        self.norm2 = get_normalization_layer(norm_type, lmax=max(self.lmax_list), num_channels=self.sphere_channels)
        
        self.energy_block2 = FeedForwardNetwork(
            self.sphere_channels,
            self.ffn_hidden_channels, 
            1,
            self.lmax_list,
            self.mmax_list,
            self.SO3_grid,  
            self.ffn_activation,
            self.use_gate_act,
            self.use_grid_mlp,
            self.use_sep_s2_act
        )
        
        self.apply(self._init_weights)
        self.apply(self._uniform_init_rad_func_linear_weights)
        
        # Add this at the end of __init__ after all model components are initialized
        self.freeze_backbone()

    def freeze_backbone(self):
        for param in self.parameters():
            param.requires_grad = False
        
        if self.regress_forces:
            for param in self.force_block.parameters():
                param.requires_grad = True
        
        for param in self.mpflow.parameters():
            param.requires_grad = True
            
        for param in self.mpflow_delta.parameters():
            param.requires_grad = True
        
        for param in self.norm2.parameters():
            param.requires_grad = True
            
        for param in self.energy_block2.parameters():
            param.requires_grad = True
            
    @conditional_grad(torch.enable_grad())
    def forward(self, data):
        self.batch_size = len(data.natoms)
        self.dtype = data.pos.dtype
        self.device = data.pos.device

        atomic_numbers = data.atomic_numbers.long()
        num_atoms = len(atomic_numbers)
        pos = data.pos
        # if self.regress_forces and self.training:
        #     pos.requires_grad_(True)

        (
            edge_index,
            edge_distance,
            edge_distance_vec,
            cell_offsets,
            _,  # cell offset distances
            neighbors,
        ) = self.generate_graph(data)

        ###############################################################
        # Initialize data structures
        ###############################################################

        # Compute 3x3 rotation matrix per edge
        edge_rot_mat = self._init_edge_rot_mat(
            data, edge_index, edge_distance_vec
        )

        # Initialize the WignerD matrices and other values for spherical harmonic calculations
        for i in range(self.num_resolutions):
            self.SO3_rotation[i].set_wigner(edge_rot_mat)

        ###############################################################
        # Initialize node embeddings
        ###############################################################

        # Init per node representations using an atomic number based embedding
        offset = 0
        x = SO3_Embedding(
            num_atoms,
            self.lmax_list,
            self.sphere_channels,
            self.device,
            self.dtype,
        )

        offset_res = 0
        offset = 0
        # Initialize the l = 0, m = 0 coefficients for each resolution
        for i in range(self.num_resolutions):
            if self.num_resolutions == 1:
                x.embedding[:, offset_res, :] = self.sphere_embedding(atomic_numbers)
            else:
                x.embedding[:, offset_res, :] = self.sphere_embedding(
                    atomic_numbers
                    )[:, offset : offset + self.sphere_channels]
            offset = offset + self.sphere_channels
            offset_res = offset_res + int((self.lmax_list[i] + 1) ** 2)

        # Edge encoding (distance and atom edge)
        edge_distance = self.distance_expansion(edge_distance)
        if self.share_atom_edge_embedding and self.use_atom_edge_embedding:
            source_element = atomic_numbers[edge_index[0]]  # Source atom atomic number
            target_element = atomic_numbers[edge_index[1]]  # Target atom atomic number
            source_embedding = self.source_embedding(source_element)
            target_embedding = self.target_embedding(target_element)
            edge_distance = torch.cat((edge_distance, source_embedding, target_embedding), dim=1)

        # Edge-degree embedding
        edge_degree = self.edge_degree_embedding(
            atomic_numbers,
            edge_distance,
            edge_index)
        x.embedding = x.embedding + edge_degree.embedding

        ###############################################################
        # Update spherical node embeddings
        ###############################################################

        x0 = x.clone()
        
        pure_eqv2 = False
        speed_compare = False
        
        if speed_compare:
            start_time1 = time.time()
        
        for i in range(self.num_layers):
            x = self.blocks[i](
                x,                  # SO3_Embedding
                atomic_numbers,
                edge_distance,
                edge_index,
                batch=data.batch    # for GraphDropPath
            )

        # Final layer norm
        x.embedding = self.norm(x.embedding)
        
        if speed_compare:
            end_time1 = time.time()
            print(f"Time taken for {self.num_layers} layers: {end_time1 - start_time1} seconds", flush=True)
            
        x1 = x.clone()
        
        ###############################################################
        # MPFlow
        ###############################################################
        if pure_eqv2:
            ut = x1.embedding - x0.embedding
            predicted_ut = ut
            predicted_x1 = x1.clone()
        else:
            ut, predicted_ut = self.calculate_predicted_ut(x0, x1, atomic_numbers, edge_distance, edge_index, data.batch, self.device)
            
            if speed_compare:
                start_time2 = time.time()
                
            predicted_x1 = self.sample_trajectory(x0, atomic_numbers, edge_distance, edge_index, data.batch, self.device)
            
            res_x = predicted_x1.embedding
            predicted_x1 = self.mpflow_delta(predicted_x1)
            predicted_x1.embedding = res_x + predicted_x1.embedding
            
            if speed_compare:
                end_time2 = time.time()
                print(f"Time taken for mpflow: {end_time2 - start_time2} seconds", flush=True)
                print(f"Time ratio: {((end_time1 - start_time1) / (end_time2 - start_time2))}", flush=True)
                
        ###############################################################
        # Energy estimation
        ###############################################################
        node_energy = None
        if self.training:
            node_energy = self.energy_block(x) 
            node_energy = node_energy.embedding.narrow(1, 0, 1)
        
        node_energy2 = self.energy_block2(predicted_x1)
        node_energy2 = node_energy2.embedding.narrow(1, 0, 1)
        
        _energy = torch.zeros(len(data.natoms), device=node_energy2.device, dtype=node_energy2.dtype)
        _energy.index_add_(0, data.batch, node_energy2.view(-1))
        energy = _energy / _AVG_NUM_NODES

        ###############################################################
        # Force estimation
        ###############################################################
        forces = None # Initialize forces to None
        if self.regress_forces:
            forces = self.force_block(predicted_x1,
                atomic_numbers,
                edge_distance,
                edge_index)
            forces = forces.embedding.narrow(1, 1, 3)
            forces = forces.view(-1, 3)
            grad_forces = None
            # if self.training:
            #     # Calculate forces as the negative gradient of total energy w.r.t. positions
            #     total_energy = energy.sum() # Sum energy over the batch to get a scalar
            #     grad_forces = -torch.autograd.grad(
            #         outputs=total_energy, 
            #         inputs=pos, 
            #         create_graph=True,      # Allow for higher-order derivatives
            #         retain_graph=True       # Keep graph for potential further use (e.g. loss.backward())
            #     )[0]
            #     # forces will have shape [num_atoms, 3] matching pos
            #     if grad_forces is None: # Safeguard check
            #         raise RuntimeError("Forces were not computed despite self.regress_forces being True.")
            
        if not self.regress_forces:
            return energy, ut, predicted_ut, x0.embedding, x1.embedding, predicted_x1.embedding, node_energy, node_energy2
        else:
            return energy, forces, grad_forces, ut, predicted_ut, x0.embedding, x1.embedding, predicted_x1.embedding, node_energy, node_energy2

    def sample_trajectory(self, x0, atomic_numbers, edge_distance, edge_index, batch, device):
        """
        Samples a trajectory using Euler's method from t=0 to t=1.
        
        Args:
            x0: Starting point (SO3_Embedding object)
            atomic_numbers: Atomic numbers of the atoms
            edge_distance: Edge distance of the atoms
            edge_index: Edge index of the atoms
            batch: Batch tensor of the atoms
            device: The device tensors are on.
        """
        x = x0.clone()
        dt = 1.0 # Integrate from t=0 to t=1

        # Initial state and time
        y0 = x.embedding
        t0 = torch.zeros((y0.shape[0], 1), device=device, dtype=y0.dtype) # Time t=0

        # Calculate velocity at t=0
        v0 = self.mpflow(x, t0, atomic_numbers, edge_distance, edge_index, batch) # v(t0, y0)
        v0.embedding = self.norm2(v0.embedding)

        # Calculate final state using Euler's method
        y1 = y0 + v0.embedding * dt
        
        # Update the SO3_Embedding object with the final state
        x.embedding = y1
        
        return x
    
    def calculate_predicted_ut(self, x0, x1, atomic_numbers, edge_distance, edge_index, batch, device):
        num_nodes = x0.embedding.shape[0]
        num_graphs = batch.max().item() + 1  # Get the number of graphs in the batch
        
        # Detach embeddings from x0 and x1 to treat them as fixed targets/inputs for MPFlow
        detached_x0_embedding = x0.embedding.detach()
        detached_x1_embedding = x1.embedding.detach()

        xt = x0.clone() # Create a new SO3_Embedding for xt
        
        # Generate a random t for each graph in the batch
        t_batch = torch.rand(num_graphs, device=device, dtype=detached_x0_embedding.dtype)  # [num_graphs]
        
        # Map the batch-specific t to each node
        t = t_batch[batch]  # [num_nodes]
        t = t.unsqueeze(-1) # [num_nodes, 1]
        t_expanded = t.unsqueeze(-1) # [num_nodes, 1, 1]
        
        # ut_target is based on detached versions of x0 and x1 embeddings
        ut_target = detached_x1_embedding - detached_x0_embedding
        
        # xt.embedding is the interpolated state for MPFlow input, also based on detached x0 and ut_target
        xt.embedding = detached_x0_embedding + t_expanded * ut_target
        
        # MPFlow predicts ut based on xt (which is built from detached components) and t
        predicted_ut_embedding = self.mpflow(xt, t, atomic_numbers, edge_distance, edge_index, batch).embedding
        predicted_ut_embedding = self.norm2(predicted_ut_embedding)
        
        return ut_target, predicted_ut_embedding
    
    
    # Initialize the edge rotation matrics
    def _init_edge_rot_mat(self, data, edge_index, edge_distance_vec):
        return init_edge_rot_mat(edge_distance_vec)
        

    @property
    def num_params(self):
        # divide pretrained equiforemrv2 and mpflow
        all_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        mpflow_params = 0
        for name, param in self.named_parameters():
            if name.startswith('mpflow'):
                mpflow_params += param.numel()

        # Calculate backbone params (total - mpflow)
        backbone_params = all_params - mpflow_params

        # Calculate trainable backbone params
        trainable_backbone_params = 0
        trainable_mpflow_params = 0
        for name, param in self.named_parameters():
            if param.requires_grad:
                if name.startswith('mpflow'):
                    trainable_mpflow_params += param.numel()
                else:
                    trainable_backbone_params += param.numel()

        # Verify calculation (optional)
        # assert trainable_params == trainable_backbone_params + trainable_mpflow_params, "Trainable parameter mismatch"

        return f"Backbone: {backbone_params} (Trainable: {trainable_backbone_params}), MPFlow: {mpflow_params} (Trainable: {trainable_mpflow_params})"
    

    def _init_weights(self, m):
        if (isinstance(m, torch.nn.Linear)
            or isinstance(m, SO3_LinearV2)
        ):
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
            if self.weight_init == 'normal':
                std = 1 / math.sqrt(m.in_features)
                torch.nn.init.normal_(m.weight, 0, std)

        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)

    
    def _uniform_init_rad_func_linear_weights(self, m):
        if (isinstance(m, RadialFunction)):
            m.apply(self._uniform_init_linear_weights)


    def _uniform_init_linear_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
            std = 1 / math.sqrt(m.in_features)
            torch.nn.init.uniform_(m.weight, -std, std)

    
    @torch.jit.ignore
    def no_weight_decay(self):
        no_wd_list = []
        named_parameters_list = [name for name, _ in self.named_parameters()]
        for module_name, module in self.named_modules():
            if (isinstance(module, torch.nn.Linear) 
                or isinstance(module, SO3_LinearV2)
                or isinstance(module, torch.nn.LayerNorm)
                or isinstance(module, EquivariantLayerNormArray)
                or isinstance(module, EquivariantLayerNormArraySphericalHarmonics)
                or isinstance(module, EquivariantRMSNormArraySphericalHarmonics)
                or isinstance(module, EquivariantRMSNormArraySphericalHarmonicsV2)
                or isinstance(module, GaussianRadialBasisLayer)):
                for parameter_name, _ in module.named_parameters():
                    if (isinstance(module, torch.nn.Linear)
                        or isinstance(module, SO3_LinearV2)
                    ):
                        if 'weight' in parameter_name:
                            continue
                    global_parameter_name = module_name + '.' + parameter_name
                    assert global_parameter_name in named_parameters_list
                    no_wd_list.append(global_parameter_name)
        return set(no_wd_list)