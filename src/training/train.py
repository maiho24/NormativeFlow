# src/training/train.py
import torch
import torch.utils.data as data
from torch.optim.lr_scheduler import ReduceLROnPlateau
import logging
from pathlib import Path
import pickle

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
    
    checkpoints_dir = models_dir / 'checkpoints'
    checkpoints_dir.mkdir(exist_ok=True, parents=True)
    
    top_n = config.get('training', {}).get('save_top_n_models', 3)
    best_models = []
    
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
            best_epoch = epoch
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
            logger.info(f'Epoch {epoch}: New best model saved (val_loss: {best_loss:.6f})')
            
            checkpoint_filename = f"model_epoch_{epoch:03d}_loss_{best_loss:.3f}.pt"
            checkpoint_path = checkpoints_dir / checkpoint_filename
            torch.save(checkpoint, checkpoint_path)
            
            best_models.append((best_loss, checkpoint_path, epoch))
            best_models.sort(key=lambda x: x[0])
            
            # Keep only the top N models
            if len(best_models) > top_n:
                _, old_checkpoint_path, _ = best_models.pop()
                if old_checkpoint_path.exists():
                    old_checkpoint_path.unlink()
            
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
    
    if best_models:
        logger.info(f"Top {len(best_models)} models saved:")
        for i, (loss, path, epoch) in enumerate(best_models):
            logger.info(f"  #{i+1}: Epoch {epoch}, Loss: {loss:.6f}, Path: {path.name}")
    
    best_checkpoint_path = models_dir / "best_model_checkpoint.pt"
    if best_checkpoint_path.exists():
        checkpoint = torch.load(best_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded best model from epoch {checkpoint['epoch']} with validation loss: {checkpoint['loss']:.3f}")
    else:
        logger.warning("No best model checkpoint found. Returning the final model state.")
        
    return model