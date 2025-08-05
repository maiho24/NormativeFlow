# src/models/coupling_layer.py
import torch
import torch.nn as nn
import numpy as np


class ConditionalCouplingLayer(nn.Module):
    def __init__(self, input_dim, hidden_dims, covariate_net_dims, c_dim, mask_type='alternating'):
        super().__init__()
        self.input_dim = input_dim
        self.c_dim = c_dim
        self.mask_type = mask_type
        
        # Generate mask
        if mask_type == 'random':
            mask_tensor = torch.from_numpy(np.random.randint(0, 2, size=input_dim)).float()
        elif mask_type == 'alternating':
            mask_tensor = torch.zeros(input_dim)
            mask_tensor[::2] = 1
        elif mask_type == 'reverse_alternating':
            mask_tensor = torch.ones(input_dim)
            mask_tensor[::2] = 0
        else:
            raise ValueError(f"Unknown mask type: {mask_type}")
        self.register_buffer('mask', mask_tensor)
        
        # Covariate processing network
        self.condition_net = self._build_network(
            input_dim=c_dim,
            hidden_dims=covariate_net_dims[:-1],
            output_dim=covariate_net_dims[-1]
        )
        
        # Scale and translation networks
        processed_dim = input_dim + covariate_net_dims[-1]
        self.scale_net = self._build_network(processed_dim, hidden_dims, input_dim)
        self.translation_net = self._build_network(processed_dim, hidden_dims, input_dim)
        self.scale_factor = 1.0
        
    def _build_network(self, input_dim, hidden_dims, output_dim, use_batch_norm=False):
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, output_dim))
        return nn.Sequential(*layers)
        
    def forward(self, y, c, reverse=False):
        """
        y: input data tensor [batch_size, input_dim]
        c: covariates tensor [batch_size, c_dim]
        reverse: if True, applies the inverse transformation
        """
        assert y.shape[1] == self.input_dim, f"y must have shape [batch_size, {self.input_dim}]"
        assert c.shape[0] == y.shape[0], "Batch size of c and y must match"
        
        c_processed = self.condition_net(c) 
        masked_y = y * self.mask
        y_c = torch.cat([masked_y, c_processed], dim=1)
        
        # Compute scale and translation
        s = self.scale_net(y_c) * (1 - self.mask)
        s = torch.tanh(s) * self.scale_factor
        t = self.translation_net(y_c) * (1 - self.mask)
        
        # Numerical stability checks
        if torch.isnan(s).any() or torch.isinf(s).any():
            raise ValueError("NaN or Inf detected in scale parameters")
        if torch.isnan(t).any() or torch.isinf(t).any():
            raise ValueError("NaN or Inf detected in translation parameters")
            
        if not reverse:
            # Forward transformation
            output = masked_y + (1 - self.mask) * (y * torch.exp(s) + t)
            log_det = torch.sum(s * (1 - self.mask), dim=1)
        else:
            # Inverse transformation
            output = masked_y + (1 - self.mask) * ((y - t) * torch.exp(-s))
            log_det = -torch.sum(s * (1 - self.mask), dim=1)
            
        return output, log_det
        
        
class AdaptiveConditionalCouplingLayer(nn.Module):
    def __init__(self, input_dim, hidden_dims, covariate_net_dims, c_dim, 
                 mask_type='alternating', layer_idx=0, total_layers=6):
        super().__init__()
        self.input_dim = input_dim
        self.c_dim = c_dim
        self.mask_type = mask_type
        self.layer_idx = layer_idx
        
        # Generate mask
        if mask_type == 'random':
            mask_tensor = torch.from_numpy(np.random.randint(0, 2, size=input_dim)).float()
        elif mask_type == 'alternating':
            mask_tensor = torch.zeros(input_dim)
            mask_tensor[::2] = 1
        elif mask_type == 'reverse_alternating':
            mask_tensor = torch.ones(input_dim)
            mask_tensor[::2] = 0
        else:
            raise ValueError(f"Unknown mask type: {mask_type}")
        self.register_buffer('mask', mask_tensor)
        
        # Covariate processing network
        self.condition_net = self._build_network(
            input_dim=c_dim,
            hidden_dims=covariate_net_dims[:-1],
            output_dim=covariate_net_dims[-1]
        )
        
        # Scale and translation networks
        processed_dim = input_dim + covariate_net_dims[-1]
        self.scale_net = self._build_network(processed_dim, hidden_dims, input_dim)
        self.translation_net = self._build_network(processed_dim, hidden_dims, input_dim)
        
        # Progressive scaling
        base_scale = 0.1
        progression_factor = 1.5
        self.scale_factor = base_scale * (progression_factor ** layer_idx)
        self.scale_factor = min(self.scale_factor, 1.0)
        
        print(f"Layer {layer_idx}: scale_factor = {self.scale_factor:.3f}")
        
    def _build_network(self, input_dim, hidden_dims, output_dim, use_batch_norm=False):
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, output_dim))
        return nn.Sequential(*layers)
        
    def forward(self, y, c, reverse=False):
        c_processed = self.condition_net(c) 
        masked_y = y * self.mask
        y_c = torch.cat([masked_y, c_processed], dim=1)
        
        s_raw = self.scale_net(y_c) * (1 - self.mask)
        t = self.translation_net(y_c) * (1 - self.mask)
        
        s = torch.tanh(s_raw) * self.scale_factor
        
        if not reverse:
            output = masked_y + (1 - self.mask) * (y * torch.exp(s) + t)
            log_det = torch.sum(s * (1 - self.mask), dim=1)
        else:
            output = masked_y + (1 - self.mask) * ((y - t) * torch.exp(-s))
            log_det = -torch.sum(s * (1 - self.mask), dim=1)
            
        return output, log_det