# src/models/transformation_layer.py
import torch
import torch.nn as nn


class AdaptiveTransformationLayer(nn.Module):
    """
    Learnable transformation layer to handle skewed and heavy-tailed distributions.
    This layer applies a feature-specific power transformation to help normalize
    the data distribution before passing it through the normalizing flow.
    """
    def __init__(self, input_dim, beta_min=1e-3):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(input_dim) * 0.5)
        self.beta = nn.Parameter(torch.ones(input_dim) * beta_min)
        self.beta_min = beta_min
        
    def forward(self, x, reverse=False):
        """
        Apply forward transformation (from data to latent) or 
        reverse transformation (from latent to data)
        
        Args:
            x: Input tensor [batch_size, input_dim]
            reverse: Boolean indicating forward or inverse transformation
            
        Returns:
            Transformed tensor and log-determinant of Jacobian
        """
        beta = torch.clamp(self.beta, min=self.beta_min)
        
        if not reverse:
            # Forward transformation (data -> latent)
            signs = torch.sign(x)
            x_abs = torch.abs(x) + beta
            y = signs * torch.pow(x_abs, self.alpha)
            
            # Compute log-determinant of Jacobian
            # d(y)/d(x) = alpha * |x|^(alpha-1) for each feature
            ldj = torch.sum(torch.log(self.alpha) + (self.alpha - 1) * torch.log(x_abs), dim=1)
            
        else:
            # Inverse transformation (latent -> data)
            signs = torch.sign(x)
            x_abs = torch.abs(x)
            y = signs * torch.pow(x_abs, 1/self.alpha) - beta * signs
            y_abs = torch.abs(y) + beta
            ldj = -torch.sum(torch.log(self.alpha) + (self.alpha - 1) * torch.log(y_abs), dim=1)
            
        return y, ldj
    
    def get_transform_parameters(self):
        """Return the learned transformation parameters for inspection"""
        return {
            'alpha': self.alpha.detach().cpu(),
            'beta': self.beta.detach().cpu()
        }


class ConditionalAdaptiveTransformationLayer(nn.Module):
    """
    Conditional version of the adaptive transformation layer where
    transformation parameters can depend on covariates.
    """
    def __init__(self, input_dim, c_dim, hidden_dims=[64, 32], beta_min=1e-3):
        super().__init__()
        self.input_dim = input_dim
        self.c_dim = c_dim
        self.beta_min = beta_min
        
        # Base values when no conditioning
        self.alpha_base = nn.Parameter(torch.ones(input_dim) * 0.5)
        self.beta_base = nn.Parameter(torch.ones(input_dim) * beta_min)
        
        # Network to predict alpha and beta from covariates
        self.param_net = self._build_network(
            input_dim=c_dim,
            hidden_dims=hidden_dims,
            output_dim=input_dim * 2
        )
        
    def _build_network(self, input_dim, hidden_dims, output_dim):
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, output_dim))
        return nn.Sequential(*layers)
    
    def get_params(self, c):
        """Get transformation parameters based on covariates"""
        batch_size = c.shape[0]
        params = self.param_net(c)
        
        # Split into alpha and beta modifiers
        alpha_mod = torch.sigmoid(params[:, :self.input_dim]) * 0.9 + 0.1  # Range: (0.1, 1.0)
        beta_mod = torch.exp(params[:, self.input_dim:]) * self.beta_min  # Range: (beta_min, inf)
        
        # Combine with base parameters
        alpha = self.alpha_base.expand(batch_size, -1) * alpha_mod
        beta = self.beta_base.expand(batch_size, -1) * beta_mod
        
        return alpha, beta
        
    def forward(self, x, c, reverse=False):
        """
        Apply conditional transformation
        
        Args:
            x: Input tensor [batch_size, input_dim]
            c: Covariates tensor [batch_size, c_dim]
            reverse: Boolean indicating forward or inverse transformation
            
        Returns:
            Transformed tensor and log-determinant of Jacobian
        """
        batch_size = x.shape[0]
        
        # Get transformation parameters based on condition
        alpha, beta = self.get_params(c)
        
        beta = torch.clamp(beta, min=self.beta_min)
        
        if not reverse:
            # Forward transformation (data -> latent)
            signs = torch.sign(x)
            x_abs = torch.abs(x) + beta
            y = signs * torch.pow(x_abs, alpha)
            ldj = torch.sum(torch.log(alpha) + (alpha - 1) * torch.log(x_abs), dim=1)
            
        else:
            # Inverse transformation (latent -> data)
            signs = torch.sign(x)
            x_abs = torch.abs(x)
            y = signs * torch.pow(x_abs, 1/alpha) - beta * signs
            y_abs = torch.abs(y) + beta
            ldj = -torch.sum(torch.log(alpha) + (alpha - 1) * torch.log(y_abs), dim=1)
            
        return y, ldj
    
    def get_transform_parameters(self, c):
        """Return the learned transformation parameters for specific covariates"""
        alpha, beta = self.get_params(c)
        return {
            'alpha': alpha.detach().cpu(),
            'beta': beta.detach().cpu()
        }