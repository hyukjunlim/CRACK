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
except ImportError:
    pass

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

        
        # Output blocks for energy and forces
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
        # if self.regress_forces:
            # self.force_block = SO2EquivariantGraphAttention(
            #     self.sphere_channels,
            #     self.attn_hidden_channels,
            #     self.num_heads, 
            #     self.attn_alpha_channels,
            #     self.attn_value_channels, 
            #     1,
            #     self.lmax_list,
            #     self.mmax_list,
            #     self.SO3_rotation, 
            #     self.mappingReduced, 
            #     self.SO3_grid, 
            #     self.max_num_elements,
            #     self.edge_channels_list,
            #     self.block_use_atom_edge_embedding, 
            #     self.use_m_share_rad,
            #     self.attn_activation, 
            #     self.use_s2_act_attn, 
            #     self.use_attn_renorm,
            #     self.use_gate_act,
            #     self.use_sep_s2_act,
            #     alpha_drop=0.0
            # )
            
        # Equivariant MPFlow
        # Define MPFlow-specific parameters here, e.g.:
        mpflow_time_embed_dim = 128
        mpflow_ffn_hidden_channels = 256
        mpflow_num_layers = 6

        self.mpflow = EquivariantMPFlow(
            sphere_channels=self.sphere_channels,
            ffn_hidden_channels=mpflow_ffn_hidden_channels,
            time_embed_dim=mpflow_time_embed_dim,
            lmax_list=self.lmax_list,
            mmax_list=self.mmax_list, # Pass mmax_list
            SO3_grid=self.SO3_grid,     # Pass SO3_grid
            activation=self.ffn_activation, # Reuse ffn activation
            norm_type=self.norm_type,      # Reuse norm type
            use_gate_act=self.use_gate_act,
            use_grid_mlp=self.use_grid_mlp,
            use_sep_s2_act=self.use_sep_s2_act,
            num_layers=mpflow_num_layers,
            proj_drop=self.proj_drop, # Pass if needed and supported by EquivariantMPFlow's blocks
        ).to(self.device)
        
        self.apply(self._init_weights)
        self.apply(self._uniform_init_rad_func_linear_weights)

        # Add this at the end of __init__ after all model components are initialized
        self.freeze_backbone()
        
        # Create a normalization layer specifically for MPFlow input/output
        self.mpflow_norm = get_normalization_layer(self.norm_type, lmax=max(self.lmax_list), num_channels=self.sphere_channels)

    def freeze_backbone(self):
        """Freeze all parameters except energy_block, and mpflow"""
        # """Freeze all parameters except energy_block, force_block, and mpflow"""
        # First freeze everything
        for param in self.parameters():
            param.requires_grad = False
        
        # Then unfreeze
        for param in self.norm.parameters():
            param.requires_grad = True
        
        for param in self.energy_block.parameters():
            param.requires_grad = True
        
        # if self.regress_forces:
        #     for param in self.force_block.parameters():
        #         param.requires_grad = True
        
        for param in self.mpflow.parameters():
            param.requires_grad = True


    @conditional_grad(torch.enable_grad())
    def forward(self, data, predict_with_mpflow=False):
        # Enable gradient tracking at the beginning of forward
        data.pos.requires_grad_(True)
        
        self.batch_size = len(data.natoms)
        self.dtype = data.pos.dtype
        self.device = data.pos.device

        atomic_numbers = data.atomic_numbers.long()
        num_atoms = len(atomic_numbers)

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
        start_time = time.time()

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
        end_time_1 = time.time()
        time_first = torch.full((data.batch.max() + 1,), end_time_1 - start_time, device=x.embedding.device, dtype=x.embedding.dtype)

        x0 = x.embedding
        if not predict_with_mpflow:
            for i in range(self.num_layers):
                x = self.blocks[i](
                    x,                  # SO3_Embedding
                    atomic_numbers,
                    edge_distance,
                    edge_index,
                    batch=data.batch    # for GraphDropPath
                )
            x1 = x.embedding
        
        end_time_2 = time.time()
        time_last = torch.full((data.batch.max() + 1,), end_time_2 - end_time_1, device=x.embedding.device, dtype=x.embedding.dtype)
        
        ###############################################################
        # MPFlow
        ###############################################################
        if not predict_with_mpflow:
            # Option 3: Per-degree normalization respecting equivariant structure
            # Create copy of embeddings for normalization to avoid modifying originals
            x0_copy = x0.clone()
            x1_copy = x1.clone()
            
            # Apply normalization that respects spherical harmonic structure
            x0_norm = self.mpflow_norm(x0_copy)
            x1_norm = self.mpflow_norm(x1_copy)
            
            # Calculate residuals for later denormalization
            x0_residual = x0 - x0_norm
            x1_residual = x1 - x1_norm
            
            ut, predicted_ut = self.calculate_predicted_ut(x0_norm, x1_norm, self.device)
            predicted_x1_norm = self.sample_trajectory(x0_norm, method="heun", enable_grad=True)
            
            # Apply the same residual to restore the distribution
            # This preserves learned correlations between channels while keeping equivariance
            predicted_x1 = predicted_x1_norm + x0_residual
            x.embedding = predicted_x1
        else:
            # Apply normalization for inference
            x0_copy = x0.clone()
            x0_norm = self.mpflow_norm(x0_copy)
            x0_residual = x0 - x0_norm
            
            # Run MPFlow on normalized data
            x1_norm = self.sample_trajectory(x0_norm, method="heun", enable_grad=False)
            
            # Restore original scale and distribution
            x1 = x1_norm + x0_residual
            x.embedding = x1
        
        end_time_3 = time.time()
        time_mpflow = torch.full((data.batch.max() + 1,), end_time_3 - end_time_2, device=x.embedding.device, dtype=x.embedding.dtype)
        
        # Final layer norm
        x.embedding = self.norm(x.embedding)
        
        ###############################################################
        # Energy estimation
        ###############################################################
        node_energy = self.energy_block(x) 
        node_energy = node_energy.embedding.narrow(1, 0, 1)
        _energy = torch.scatter_add(
            torch.zeros(data.batch.max() + 1, device=node_energy.device, dtype=node_energy.dtype),
            0,
            data.batch,
            node_energy.view(-1)
        )
        energy = _energy / _AVG_NUM_NODES

        ###############################################################
        # Force estimation ( + gradient aided)
        ###############################################################
        if self.regress_forces:
            if not predict_with_mpflow:
                dy = torch.autograd.grad(
                    energy,  # [n_graphs,]
                    data.pos,  # [n_nodes, 3]
                    grad_outputs=torch.ones_like(energy),
                    create_graph=True,
                    retain_graph=True,
                    allow_unused=True
                )[0]
                assert dy is not None
                forces = -1 * dy  # [n_nodes, 3]
            # forces = self.force_block(x,
            #     atomic_numbers,
            #     edge_distance,
            #     edge_index)
            # forces = forces.embedding.narrow(1, 1, 3)
            # forces = forces.view(-1, 3)    
        
        if not predict_with_mpflow:
            if not self.regress_forces:
                return energy, ut, predicted_ut, x0, x1, predicted_x1, time_first, time_last, time_mpflow
            else:
                return energy, forces, ut, predicted_ut, x0, x1, predicted_x1, time_first, time_last, time_mpflow
        else:
            if not self.regress_forces:
                return energy, x0, x1, time_first, time_last, time_mpflow
            else:
                return energy, forces, x0, x1, time_first, time_last, time_mpflow


    def sample_trajectory(self, x0, method="heun", enable_grad=False):
        """
        Samples a trajectory using ultra-fast, high-accuracy ODE solver.
        
        Args:
            x0: Starting point
            method: Sampling method ("heun", "rk4", "dopri5")
            enable_grad: Whether to enable gradient computation
        """
        self.mpflow.eval()
        batch_size = x0.shape[0]
        num_nodes = x0.shape[0] # x0 shape: [num_nodes, num_coefficients, channels]
        device = x0.device
        x = x0.clone()
        
        if not enable_grad:
            with torch.no_grad():
                return self._compute_trajectory(x, method, batch_size, device)
        else:
            return self._compute_trajectory(x, method, batch_size, device)
        
        
    def _compute_trajectory(self, x, method, batch_size, device):
        num_nodes = x.shape[0]
        if method == "euler":
            # Simple Euler method
            # Time shape should be [num_nodes, 1]
            t = torch.zeros((num_nodes, 1), device=device)
            velocity = self.mpflow(x, t)
            x = x + velocity
            
        elif method == "heun":
            # Previous Heun implementation
            t = torch.zeros((num_nodes, 1), device=device)
            k1 = self.mpflow(x, t)
            x_pred = x + k1
            t_next = torch.ones((num_nodes, 1), device=device) # time t+1
            k2 = self.mpflow(x_pred, t_next)
            x = x + 0.5 * (k1 + k2)
        
        elif method == "rk4":
            # Original RK4 implementation
            t = torch.zeros((num_nodes, 1), device=device)
            t_half = torch.full((num_nodes, 1), 0.5, device=device)
            t_one = torch.ones((num_nodes, 1), device=device)
            k1 = self.mpflow(x, t)
            k2 = self.mpflow(x + 0.5 * k1, t_half)
            k3 = self.mpflow(x + 0.5 * k2, t_half)
            k4 = self.mpflow(x + k3, t_one)
            x = x + (1.0 / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            
        return x


    def calculate_predicted_ut(self, x0, x1, device):
        num_nodes = x0.shape[0]
        # Time shape should be [num_nodes, 1]
        t = torch.rand(num_nodes, 1, device=device)
        ut = x1 - x0
        # Explicitly broadcast t before multiplying with ut
        t_expanded = t.unsqueeze(-1) # Shape [N, 1, 1]
        xt = x0 + t_expanded * ut # Shape [N, 49, C] + [N, 49, C]
        predicted_ut = self.mpflow(xt, t)
        
        return ut, predicted_ut


    # Initialize the edge rotation matrics
    def _init_edge_rot_mat(self, data, edge_index, edge_distance_vec):
        return init_edge_rot_mat(edge_distance_vec)
        

    @property
    def num_params(self):
        # divide pretrained equiforemrv2 and mpflow
        all_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        # Check if mpflow is initialized before accessing parameters
        if hasattr(self, 'mpflow') and self.mpflow is not None:
            mpflow_params = sum(p.numel() for p in self.mpflow.parameters())
        else:
            mpflow_params = 0 # Or handle as an error if mpflow should always exist

        # Calculate backbone params (total - mpflow)
        backbone_params = all_params - mpflow_params

        # Calculate trainable backbone params
        trainable_backbone_params = 0
        trainable_mpflow_params = 0
        for name, param in self.named_parameters():
            if param.requires_grad:
                if name.startswith('mpflow.'):
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