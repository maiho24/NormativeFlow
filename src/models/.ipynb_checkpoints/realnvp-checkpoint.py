# src/models/realnvp.py
import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal

from .coupling_layer import ConditionalCouplingLayer
from .transformation_layer import AdaptiveTransformationLayer, ConditionalAdaptiveTransformationLayer


class ConditionalRealNVP(nn.Module):
    def __init__(self, 
                 input_dim, 
                 hidden_dims,
                 covariate_net_dims, 
                 c_dim,
                 n_layers=4,
                 learning_rate=0.001,
                 weight_decay=1e-5,
                 use_transformation_layer=True,
                 conditional_transform=True,
                 transform_hidden_dims=[64, 32]):
        super().__init__()
        self.input_dim = input_dim
        self.c_dim = c_dim
        self.use_transformation_layer = use_transformation_layer
        self.conditional_transform = conditional_transform
        
        # Initialize transformation layer if enabled
        if use_transformation_layer:
            if conditional_transform:
                self.transform_layer = ConditionalAdaptiveTransformationLayer(
                    input_dim=input_dim,
                    c_dim=c_dim,
                    hidden_dims=transform_hidden_dims
                )
            else:
                self.transform_layer = AdaptiveTransformationLayer(input_dim=input_dim)
        
        # Create coupling layers
        self.layers = nn.ModuleList([
            ConditionalCouplingLayer(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                covariate_net_dims=covariate_net_dims,
                c_dim=c_dim,
                mask_type='alternating' if i % 2 == 0 else 'reverse_alternating'
            ) for i in range(n_layers)
        ])
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
    def forward(self, y, c, reverse=False):
        """
        Forward transformation (y -> z) and inverse transformation (z -> y).
        """
        log_det_total = torch.zeros(y.shape[0]).to(y.device)
        
        if not reverse:
            # Forward: data -> latent
            # Apply transformation layer first if enabled
            if self.use_transformation_layer:
                if self.conditional_transform:
                    y, ldj = self.transform_layer(y, c)
                else:
                    y, ldj = self.transform_layer(y)
                log_det_total += ldj
            
            # Then apply RealNVP coupling layers
            for layer in self.layers:
                y, log_det = layer(y, c)
                log_det_total += log_det
        else:
            # Inverse: latent -> data
            # Apply RealNVP coupling layers in reverse
            for layer in reversed(self.layers):
                y, log_det = layer(y, c, reverse=True)
                log_det_total += log_det
                
            # Apply inverse transformation at the end if enabled
            if self.use_transformation_layer:
                if self.conditional_transform:
                    y, ldj = self.transform_layer(y, c, reverse=True)
                else:
                    y, ldj = self.transform_layer(y, reverse=True)
                log_det_total += ldj
                
        return y, log_det_total
    
    def log_prob(self, y, c):
        """
        Compute log probability of y given c under the normalizing flow model.
        """
        z, log_det = self.forward(y, c)
        
        # Log probability under base standard normal distribution
        base_dist = Normal(0, 1)
        log_prob_base = torch.sum(base_dist.log_prob(z), dim=1)
        
        # Add log determinant of transformation
        log_prob = log_prob_base + log_det
        
        return log_prob, log_det, log_prob_base
    
    def sample(self, n_samples, c, device='cpu'):
        """
        Generate samples from the model given c.
        """
        if c.shape[0] != n_samples:
            raise ValueError(f"Expected c to have batch size {n_samples}, but got {c.shape[0]}")

        z = torch.randn(n_samples, self.input_dim).to(device)
        c = c.to(device)

        y, _ = self.forward(z, c, reverse=True)
        
        return y
        
    def loss_function(self, y, c):
        """
        Computes the negative log-likelihood loss.
        """
        log_prob, log_det, log_prob_base = self.log_prob(y, c)
        total_loss = -torch.mean(log_prob)
        
        return {
            'Total Loss': total_loss,
            'Log Determinant': torch.mean(log_det),
            'Negative Log Likelihood': -torch.mean(log_prob_base)
        }
        
    def per_feature_log_prob(self, y, c):
        """
        Compute the per-feature log probability of y given c under the normalizing flow model.
        
        Args:
            y: input data tensor [batch_size, input_dim]
            c: covariates tensor [batch_size, c_dim]
        """
        transform_contribution = torch.zeros_like(y)
        
        # Apply transformation layer if enabled
        if self.use_transformation_layer:
            if self.conditional_transform:
                signs = torch.sign(y)
                beta = torch.clamp(self.transform_layer.get_params(c)[1], min=self.transform_layer.beta_min)
                x_abs = torch.abs(y) + beta
                alpha = self.transform_layer.get_params(c)[0]
                transform_contribution = torch.log(alpha) + (alpha - 1) * torch.log(x_abs)
                y_transformed = signs * torch.pow(x_abs, alpha)
            else:
                signs = torch.sign(y)
                beta = torch.clamp(self.transform_layer.beta, min=self.transform_layer.beta_min)
                x_abs = torch.abs(y) + beta
                transform_contribution = torch.log(self.transform_layer.alpha) + (self.transform_layer.alpha - 1) * torch.log(x_abs)
                y_transformed = signs * torch.pow(x_abs, self.transform_layer.alpha)
        else:
            y_transformed = y
            
        scale_list = []
        
        # Apply coupling layers
        current_y = y_transformed
        for layer in self.layers:
            mask = layer.mask
            
            y_masked = current_y * mask
            c_processed = layer.condition_net(c)
            y_c = torch.cat([y_masked, c_processed], dim=1)
            
            s = layer.scale_net(y_c) * (1 - mask)
            s = torch.tanh(s) * layer.scale_factor
            t = layer.translation_net(y_c) * (1 - mask)
            
            current_y = y_masked + (1 - mask) * (current_y * torch.exp(s) + t)
            
            scale_list.append(s)
        
        # current_y is now the final z in latent space
        z = current_y
        
        # Calculate base log probability for each dimension (standard normal)
        log_p_base = -0.5 * (z**2) - 0.5 * torch.log(torch.tensor(2 * torch.pi))
        
        # Sum up scale factors from all layers for each dimension
        total_s = torch.zeros_like(z)
        for s in scale_list:
            total_s += s
        
        # Feature-wise log probability: log_p_base + total jacobian contribution (transform + coupling)
        per_feature_log_prob = log_p_base + total_s + transform_contribution
        
        return per_feature_log_prob, z
    
    def get_transformation_parameters(self, c=None):
        """
        Return the learned transformation parameters.
        For the conditional transform, provide covariates c to get specific parameters.
        """
        if not self.use_transformation_layer:
            return None
            
        if self.conditional_transform and c is not None:
            return self.transform_layer.get_transform_parameters(c)
        elif not self.conditional_transform:
            return self.transform_layer.get_transform_parameters()
        else:
            return None