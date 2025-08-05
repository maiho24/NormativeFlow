# src/training/train.py
import torch
import torch.utils.data as data
from torch.optim.lr_scheduler import ReduceLROnPlateau
import logging
from pathlib import Path
import pickle
import numpy as np
import pandas as pd

from utils import MyDataset, Logger, plot_losses
from models import ConditionalRealNVP

logger = logging.getLogger(__name__)

def _parse_hidden_dim(hidden_dim):
    """
    Parse hidden dimensions from string format.
    
    Args:
        hidden_dim: String in format "dim1_dim2" or "dim1"
        
    Returns:
        List of integers representing hidden dimensions
    """
    if not isinstance(hidden_dim, str):
        raise ValueError(f"Hidden dimensions must be a string (e.g., '64_32'), got {type(hidden_dim)}")
    
    try:
        return [int(x) for x in hidden_dim.split('_')]
    except (ValueError, AttributeError) as e:
        logger.error(f"Error parsing hidden dimensions {hidden_dim}: {str(e)}")
        raise ValueError(f"Invalid hidden dimension format: {hidden_dim}. Expected format: 'dim1_dim2' or 'dim1'")

def extract_transformation_parameters(model, val_X, config, device):
    """
    Extract and save transformation parameters from the trained model.
    
    Args:
        model: Trained ConditionalRealNVP model
        val_X: Validation covariates for generating sample-specific parameters
        config: Configuration dictionary with paths
        device: Computing device
    """
    if not model.use_transformation_layer:
        logger.info("No transformation layer used in the model.")
        return
    
    logger.info("Extracting transformation layer parameters...")
    
    model_dir = Path(config['paths']['model_dir'])
    transform_params_dir = model_dir / 'transform_params'
    transform_params_dir.mkdir(exist_ok=True, parents=True)
    
    if model.conditional_transform:
        # Log the base parameters
        logger.info("Base transformation parameters:")
        alpha_base = model.transform_layer.alpha_base.detach().cpu().numpy()
        beta_base = model.transform_layer.beta_base.detach().cpu().numpy()
        logger.info(f"Alpha base shape: {alpha_base.shape}")
        logger.info(f"Alpha base min: {alpha_base.min()}, max: {alpha_base.max()}, mean: {alpha_base.mean()}")
        logger.info(f"Beta base min: {beta_base.min()}, max: {beta_base.max()}, mean: {beta_base.mean()}")
        
        # Save base parameters
        np.save(transform_params_dir / 'alpha_base.npy', alpha_base)
        np.save(transform_params_dir / 'beta_base.npy', beta_base)
        
        alpha_df = pd.DataFrame([alpha_base], columns=[f'feature_{i}' for i in range(len(alpha_base))])
        beta_df = pd.DataFrame([beta_base], columns=[f'feature_{i}' for i in range(len(beta_base))])
        alpha_df.to_csv(transform_params_dir / 'alpha_base.csv', index=False)
        beta_df.to_csv(transform_params_dir / 'beta_base.csv', index=False)
        
        # Log a few sample-specific parameters
        sample_indices = np.random.choice(len(val_X), min(5, len(val_X)), replace=False)
        sample_covs = torch.tensor(val_X[sample_indices], dtype=torch.float32).to(device)
        
        logger.info("Sample-specific transformation parameters (5 examples):")
        for i, cov in enumerate(sample_covs):
            params = model.get_transformation_parameters(cov.unsqueeze(0))
            logger.info(f"Sample {i+1}:")
            logger.info(f"Alpha min: {params['alpha'].min().item()}, max: {params['alpha'].max().item()}, mean: {params['alpha'].mean().item()}")
            logger.info(f"Beta min: {params['beta'].min().item()}, max: {params['beta'].max().item()}, mean: {params['beta'].mean().item()}")
        
        # Generate and save parameters for a larger sample set
        all_samples = min(50, len(val_X))
        sample_indices = np.random.choice(len(val_X), all_samples, replace=False)
        sample_covs = torch.tensor(val_X[sample_indices], dtype=torch.float32).to(device)
        
        alphas = []
        betas = []
        for cov in sample_covs:
            params = model.get_transformation_parameters(cov.unsqueeze(0))
            alphas.append(params['alpha'].numpy()[0])
            betas.append(params['beta'].numpy()[0])
            
        alpha_df = pd.DataFrame(alphas, columns=[f'feature_{i}' for i in range(len(alphas[0]))])
        beta_df = pd.DataFrame(betas, columns=[f'feature_{i}' for i in range(len(betas[0]))])
        
        alpha_df.to_csv(transform_params_dir / 'sample_alphas.csv', index=False)
        beta_df.to_csv(transform_params_dir / 'sample_betas.csv', index=False)
        
    else:
        # For non-conditional transform, get the parameters directly
        params = model.get_transformation_parameters()
        alpha = params['alpha'].numpy()
        beta = params['beta'].numpy()
        
        logger.info("Transformation parameters:")
        logger.info(f"Alpha shape: {alpha.shape}")
        logger.info(f"Alpha min: {alpha.min()}, max: {alpha.max()}, mean: {alpha.mean()}")
        logger.info(f"Beta min: {beta.min()}, max: {beta.max()}, mean: {beta.mean()}")
        
        # Save parameters
        np.save(transform_params_dir / 'alpha.npy', alpha)
        np.save(transform_params_dir / 'beta.npy', beta)
        
        alpha_df = pd.DataFrame([alpha], columns=[f'feature_{i}' for i in range(len(alpha))])
        beta_df = pd.DataFrame([beta], columns=[f'feature_{i}' for i in range(len(beta))])
        
        alpha_df.to_csv(transform_params_dir / 'alpha.csv', index=False)
        beta_df.to_csv(transform_params_dir / 'beta.csv', index=False)
    
    logger.info(f"Transformation parameters saved to {transform_params_dir}")
    
def validate_model(model, generator_val, device):
    """Run validation step."""
    total_val_loss = 0
    total_logdet = 0
    total_negloglik = 0
    num_batches = 0

    with torch.no_grad():
        for Y_val, X_val in generator_val:
            Y_val, X_val = Y_val.to(device), X_val.to(device)
            val_loss = model.loss_function(Y_val, X_val)
            total_val_loss += val_loss['Total Loss'].item()
            total_logdet += val_loss['Log Determinant'].item()
            total_negloglik += val_loss['Negative Log Likelihood'].item()
            num_batches += 1

    return {
        'total_loss': total_val_loss / num_batches,
        'logdet': total_logdet / num_batches,
        'negloglik': total_negloglik / num_batches
    }

def train_model(train_Y, train_X, val_Y, val_X, config):
    """Train the RealNVP model with detailed loss logging."""
    device = torch.device("cuda" if config['device']['gpu'] and torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    train_dataset = MyDataset(train_Y, train_X)
    val_dataset = MyDataset(val_Y, val_X)
    
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True
    )
    val_loader = data.DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False
    )

    hidden_dims = _parse_hidden_dim(config['model']['hidden_dim'])
    covariate_net_dims = _parse_hidden_dim(config['model']['covariate_net_dim'])
    use_transformation_layer = config['model'].get('use_transformation_layer', False)
    conditional_transform = config['model'].get('conditional_transform', False)
    transform_hidden_dims = _parse_hidden_dim(config['model'].get('transform_hidden_dims', '32_64'))
    weight_decay = config['model'].get('weight_decay', 1e-5)
    
    model = ConditionalRealNVP(
        input_dim=train_Y.shape[1],
        hidden_dims=hidden_dims,
        covariate_net_dims=covariate_net_dims,
        c_dim=train_X.shape[1],
        n_layers=config['model']['n_layers'],
        learning_rate=config['model']['learning_rate'],
        weight_decay=weight_decay,
        use_transformation_layer=use_transformation_layer,
        conditional_transform=conditional_transform,
        transform_hidden_dims=transform_hidden_dims
    )
    model.to(device)
    
    scheduler = ReduceLROnPlateau(
        model.optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True,
        min_lr=1e-6
    )

    train_logger = Logger()
    train_logger.on_train_init(['total_loss', 'logdet', 'negloglik'])
    train_logger.on_val_init(['total_loss', 'logdet', 'negloglik'])
    
    best_loss = float('inf')
    patience_counter = 0
    best_epoch = None
    
    models_dir = Path(config['paths']['model_dir'])
    models_dir.mkdir(exist_ok=True, parents=True)
    
    #------ TRAINING & VALIDATION ------#
    for epoch in range(config['training']['epochs']):
        # Training phase
        model.train()
        train_losses = {'total_loss': 0, 'logdet': 0, 'negloglik': 0}
        num_batches = 0
        
        for batch_Y, batch_X in train_loader:
            batch_Y = batch_Y.to(device)
            batch_X = batch_X.to(device)
            
            loss_dict = model.loss_function(batch_Y, batch_X)
            
            model.optimizer.zero_grad()
            loss_dict['Total Loss'].backward()
            model.optimizer.step()
            
            train_losses['total_loss'] += loss_dict['Total Loss'].item()
            train_losses['logdet'] += loss_dict['Log Determinant'].item()
            train_losses['negloglik'] += loss_dict['Negative Log Likelihood'].item()
            num_batches += 1

        avg_train_losses = {k: v/num_batches for k, v in train_losses.items()}
        current_lr = model.optimizer.param_groups[0]['lr']

        # Validation phase
        model.eval()
        avg_val_losses = validate_model(model, val_loader, device)
        
        scheduler.step(avg_val_losses['total_loss'])

        # Early stopping check
        if avg_val_losses['total_loss'] < best_loss:
            best_loss = avg_val_losses['total_loss']
            patience_counter = 0
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': model.optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'loss': best_loss
            }
            best_checkpoint_path = models_dir / "best_model_checkpoint.pt"
            torch.save(checkpoint, best_checkpoint_path)
            logger.info(f'Epoch {epoch}: New best model saved')
            
            logger_path = models_dir / "training_logger.pkl"
            with open(logger_path, 'wb') as f:
                pickle.dump(train_logger, f)
        else:
            patience_counter += 1
            if patience_counter >= config['training']['early_stopping_patience']:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break

        train_logger.on_train_step(avg_train_losses)
        train_logger.on_val_step(avg_val_losses)

        logger.info(
            f"Epoch {epoch}: "
            f"Train Loss = {avg_train_losses['total_loss']:.4f} "
            f"(LogDet: {avg_train_losses['logdet']:.4f}, "
            f"NegLogLik: {avg_train_losses['negloglik']:.4f}), "
            f"Val Loss = {avg_val_losses['total_loss']:.4f} "
            f"(LogDet: {avg_val_losses['logdet']:.4f}, "
            f"NegLogLik: {avg_val_losses['negloglik']:.4f}), "
            f"Learning Rate: {current_lr:.2e}"
        )
    
    plot_losses(train_logger, models_dir, '_training_validation')
    
    best_checkpoint_path = models_dir / f"best_model_checkpoint.pt"
    if best_checkpoint_path.exists():
        checkpoint = torch.load(best_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded best model with validation loss: {checkpoint['loss']}")
    else:
        logger.warning("No best model checkpoint found. Returning the final model state.")

    if config['training']['extract_transform_params']:
        extract_transformation_parameters(model, val_X, config, device)
        
    return model