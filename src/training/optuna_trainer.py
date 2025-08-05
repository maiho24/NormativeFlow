# src/training/optuna_trainer.py
import optuna
import multiprocessing
import torch
import torch.utils.data as data
from torch.optim.lr_scheduler import ReduceLROnPlateau
import logging
import yaml
from pathlib import Path
import pickle
import os
from datetime import datetime

from utils import MyDataset, Logger, plot_losses
from models import ConditionalRealNVP
from . import validate_model

logger = logging.getLogger(__name__)


class OptunaTrainer:
    def __init__(self, train_Y, train_X, val_Y, val_X, config):
        """
        Initialize the Optuna trainer.
        
        Args:
            train_data: Training data numpy array
            train_covariates: Training covariates numpy array
            val_data: Validation data numpy array
            val_covariates: Validation covariates numpy array
            config: Configuration dictionary
        """
        self.train_Y = train_Y
        self.train_X = train_X
        self.val_Y = val_Y
        self.val_X = val_X
        self.config = config
        self.best_trial_logger = None
        self.top_models = []  # Store top 5 models info
        
        storage = self.config['optuna'].get('storage', None)
        cores_per_trial = self.config.get('optuna', {}).get('cores_per_trial', 1)
        logger.info(f"Using ~{cores_per_trial} cores per trial.")
        self.n_jobs = self._determine_n_jobs(cores_per_trial)
            
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=5, 
            n_warmup_steps=20,
            interval_steps=1
        )
        
        sampler = optuna.samplers.TPESampler(seed=config.get('seed', 42))
        
        self.study = optuna.create_study(
            study_name=self.config['optuna']['study_name'],
            direction="minimize",
            storage=storage,
            load_if_exists=True,
            pruner=pruner,
            sampler=sampler
        )
    
    def _determine_n_jobs(self, cores_per_trial=1):
        """Determine optimal number of parallel jobs based on hardware resources."""
        user_n_jobs = self.config.get('optuna', {}).get('n_jobs', -1)
        cpu_count = multiprocessing.cpu_count()
        gpu_count = torch.cuda.device_count() if self.config.get('device', {}).get('gpu', True) else 0
        
        cores_reserved = self.config.get('optuna', {}).get('cores_reserved', 2)
        max_safe_parallel = max(1, (cpu_count - cores_reserved) // cores_per_trial)
        
        logger.info(f"Detected {cpu_count} CPUs and {gpu_count} GPUs.")
        
        if gpu_count > 0:
            if user_n_jobs <= 0:
                optimal_n_jobs = gpu_count
                logger.info(f"Using {optimal_n_jobs} GPUs.")
            else:
                optimal_n_jobs = user_n_jobs
                if user_n_jobs > gpu_count + max_safe_parallel:
                    logger.warning(
                        f"User-specified n_jobs={user_n_jobs} may oversubscribe resources. "
                        f"Consider reducing to {gpu_count + max_safe_parallel} for better performance."
                    )
        else:
            if self.config.get('device', {}).get('gpu', True):
                logger.info(f"Detected 0 GPUs -> Using {cpu_count} CPUs instead.")
            if user_n_jobs <= 0:
                optimal_n_jobs = max_safe_parallel
                logger.info(f"Using {optimal_n_jobs} CPUs.")
            else:
                optimal_n_jobs = min(user_n_jobs, max_safe_parallel)
                if optimal_n_jobs < user_n_jobs:
                    logger.info(f"Limiting user-specified n_jobs={user_n_jobs} to {optimal_n_jobs} to prevent CPU oversubscription")
        
        return max(1, optimal_n_jobs)
    
    def _get_device_for_trial(self):
        """Select the most appropriate device for a trial with better memory management."""
        try:
            if torch.cuda.is_available() and self.config.get('device', {}).get('gpu', True):
                if not hasattr(self, '_gpu_assignment_count'):
                    self._gpu_assignment_count = {i: 0 for i in range(torch.cuda.device_count())}
                    
                free_mem = []
                for i in range(torch.cuda.device_count()):
                    try:
                        total_mem = torch.cuda.get_device_properties(i).total_memory
                        allocated_mem = torch.cuda.memory_allocated(i)
                        free_mem_bytes = total_mem - allocated_mem
                        free_mem_gb = free_mem_bytes / (1024**3)
                        free_mem.append((free_mem_gb, self._gpu_assignment_count[i], i))
                    except Exception as e:
                        logger.warning(f"Error querying GPU {i}: {e}")
                        continue
                
                if free_mem:
                    # Sort by: 1) most free memory 2) least assigned trials
                    free_mem.sort(key=lambda x: (-x[0], x[1]))
                    best_free_mem, _, best_gpu = free_mem[0]
                    
                    # Check against minimum memory threshold
                    min_gpu_mem_gb = self.config.get('device', {}).get('min_gpu_memory_gb', 1.0)
                    
                    if best_free_mem >= min_gpu_mem_gb:
                        self._gpu_assignment_count[best_gpu] += 1
                        device = torch.device(f"cuda:{best_gpu}")
                        return device
                    else:
                        logger.info(f"Best GPU has only {best_free_mem:.2f}GB free memory (below {min_gpu_mem_gb}GB threshold)")
        except Exception as e:
            logger.warning(f"Error during device selection: {e}")
        
        return torch.device("cpu")
        
    def _parse_hidden_dim(self, hidden_dim_str):
        """
        Parse string representation of hidden dimensions into list of integers.
        
        Args:
            hidden_dim_str: String representation of hidden dimensions (e.g., "64_32")
            
        Returns:
            List of integers representing hidden dimensions
        """
        try:
            return [int(x) for x in hidden_dim_str.split('_')]
        except (ValueError, AttributeError) as e:
            logger.error(f"Error parsing hidden dimensions {hidden_dim_str}: {str(e)}")
            raise ValueError(f"Invalid hidden dimension format: {hidden_dim_str}")
    
    def _update_top_models(self, trial_number, validation_loss, trial_params):
        """Update the list of top 5 models based on validation loss."""
        model_info = {
            'trial_number': trial_number,
            'validation_loss': validation_loss,
            'params': trial_params.copy()
        }
        
        # Add to top models list
        self.top_models.append(model_info)
        
        # Sort by validation loss (ascending) and keep only top 5
        self.top_models.sort(key=lambda x: x['validation_loss'])
        if len(self.top_models) > 5:
            self.top_models = self.top_models[:5]
        
        logger.info(f"Updated top models list. Current top 5 validation losses: "
                   f"{[round(m['validation_loss'], 6) for m in self.top_models]}")
    
    def _extract_model_config(self, model, trial_params):
        """Extract configuration needed to recreate the model."""
        hidden_dims = self._parse_hidden_dim(trial_params['hidden_dim'])
        covariate_net_dims = self._parse_hidden_dim(trial_params['covariate_net_dim'])
        
        # Handle optional transformation layer parameters
        use_transformation_layer = self.config['optuna'].get('use_transformation_layer', False)
        conditional_transform = trial_params.get('conditional_transform', 
                                                self.config.get('model', {}).get('conditional_transform', False))
        
        transform_hidden_dims = None
        if use_transformation_layer and conditional_transform and 'transform_hidden_dims' in trial_params:
            transform_hidden_dims = self._parse_hidden_dim(trial_params['transform_hidden_dims'])
        elif use_transformation_layer and conditional_transform:
            transform_hidden_dims = self._parse_hidden_dim(
                self.config.get('model', {}).get('transform_hidden_dims', '32_64')
            )
        
        return {
            'input_dim': self.train_Y.shape[1],
            'hidden_dims': hidden_dims,
            'covariate_net_dims': covariate_net_dims,
            'c_dim': self.train_X.shape[1],
            'n_layers': trial_params['n_layers'],
            'learning_rate': trial_params['learning_rate'],
            'weight_decay': trial_params['weight_decay'],
            'use_transformation_layer': use_transformation_layer,
            'conditional_transform': conditional_transform,
            'transform_hidden_dims': transform_hidden_dims
        }
        
    def create_model(self, trial):
        """Create model with parameters suggested by Optuna."""
        use_transformation_layer = self.config['optuna'].get('use_transformation_layer', True)
        
        # Suggest hyperparameters
        hidden_dim_str = trial.suggest_categorical(
            'hidden_dim', 
            self.config['optuna']['search_space']['hidden_dim']['choices']
        )
        hidden_dims = self._parse_hidden_dim(hidden_dim_str)
        
        covariate_net_str = trial.suggest_categorical(
            'covariate_net_dim',
            self.config['optuna']['search_space']['covariate_net_dim']['choices']
        )
        covariate_net_dims = self._parse_hidden_dim(covariate_net_str)
        
        n_layers = trial.suggest_int(
            'n_layers',
            self.config['optuna']['search_space']['n_layers']['min'],
            self.config['optuna']['search_space']['n_layers']['max']
        )
        
        learning_rate = trial.suggest_float(
            'learning_rate',
            float(self.config['optuna']['search_space']['learning_rate']['min']),
            float(self.config['optuna']['search_space']['learning_rate']['max']),
            log=True
        )
        
        weight_decay = trial.suggest_float(
            'weight_decay',
            float(self.config['optuna']['search_space'].get('weight_decay', {}).get('min', 1e-6)),
            float(self.config['optuna']['search_space'].get('weight_decay', {}).get('max', 1e-3)),
            log=True
        )
        
        batch_size = trial.suggest_categorical(
            'batch_size',
            self.config['optuna']['search_space']['batch_size']['choices']
        )
        
        conditional_transform = False
        transform_hidden_dims = None
        
        if use_transformation_layer:
            # Optionally make the transformation conditional or not
            if 'conditional_transform' in self.config['optuna']['search_space']:
                conditional_transform = trial.suggest_categorical(
                    'conditional_transform',
                    self.config['optuna']['search_space']['conditional_transform']['choices']
                )
            else:
                conditional_transform = self.config['model'].get('conditional_transform', False)
            
            # If conditional, suggest hidden dimensions for the transformation network
            if conditional_transform and 'transform_hidden_dims' in self.config['optuna']['search_space']:
                transform_hidden_dim_str = trial.suggest_categorical(
                    'transform_hidden_dims',
                    self.config['optuna']['search_space']['transform_hidden_dims']['choices']
                )
                transform_hidden_dims = self._parse_hidden_dim(transform_hidden_dim_str)
            elif conditional_transform:
                transform_hidden_dims = self._parse_hidden_dim(
                    self.config['model'].get('transform_hidden_dims', '32_64')
                )
        
        # Create model with suggested parameters
        model = ConditionalRealNVP(
            input_dim=self.train_Y.shape[1],
            hidden_dims=hidden_dims,
            covariate_net_dims=covariate_net_dims,
            c_dim=self.train_X.shape[1],
            n_layers=n_layers,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            use_transformation_layer=use_transformation_layer,
            conditional_transform=conditional_transform,
            transform_hidden_dims=transform_hidden_dims
        )
        
        return model, batch_size
        
    def objective(self, trial):
        """Optuna objective function."""
        try:
            device = torch.device(self._get_device_for_trial()) 
            logger.info(f"Trial {trial.number} running on {device}")
        
            model, batch_size = self.create_model(trial)
            model = model.to(device)
            
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                model.optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                verbose=True,
                min_lr=1e-6
            )
            
            train_dataset = MyDataset(self.train_Y, self.train_X)
            val_dataset = MyDataset(self.val_Y, self.val_X)
            
            train_loader = data.DataLoader(
                train_dataset, 
                batch_size=batch_size, 
                shuffle=True,
                pin_memory=True if device.type == 'cuda' else False
            )
            
            val_loader = data.DataLoader(
                val_dataset, 
                batch_size=batch_size, 
                shuffle=False,
                pin_memory=True if device.type == 'cuda' else False
            )
        
            best_val_loss = float('inf')
            patience_counter = 0
            logger_trial = Logger()
            logger_trial.on_train_init(['total_loss', 'logdet', 'negloglik'])
            logger_trial.on_val_init(['total_loss', 'logdet', 'negloglik'])
        
            checkpoint_dir = Path(self.config['paths']['model_dir']) / 'checkpoints'
            checkpoint_dir.mkdir(exist_ok=True, parents=True)
            
            #------ TRAINING & VALIDATION ------#        
            for epoch in range(self.config['training']['epochs']):
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
                current_val_loss = avg_val_losses['total_loss']
                scheduler.step(current_val_loss)
        
                # Early stopping check
                if current_val_loss < best_val_loss:
                    best_val_loss = current_val_loss
                    patience_counter = 0
                    
                    model_config = self._extract_model_config(model, trial.params)
                    
                    checkpoint = {
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': model.optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'model_config': model_config,
                        'trial_number': trial.number,
                        'epoch': epoch,
                        'validation_loss': best_val_loss,
                        'params': trial.params.copy()
                    }
                    
                    checkpoint_path = checkpoint_dir / f"trial_{trial.number}_checkpoint.pt"
                    torch.save(checkpoint, checkpoint_path)
                    
                    # Save logger separately
                    logger_path = checkpoint_dir / f"trial_{trial.number}_logger.pkl"
                    with open(logger_path, 'wb') as f:
                        pickle.dump(logger_trial, f)
                else:
                    patience_counter += 1
                    if patience_counter >= self.config['training']['early_stopping_patience']:
                        break
                
                logger_trial.on_train_step(avg_train_losses)
                logger_trial.on_val_step(avg_val_losses)
                
                trial.report(current_val_loss, epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
            
            # Update top models list after trial completion
            self._update_top_models(trial.number, best_val_loss, trial.params)
            
            return best_val_loss
            
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def _save_top_models(self, model_dir):
        """Save the top 5 models in the checkpoints folder with ranking."""
        checkpoint_dir = model_dir / 'checkpoints'
        
        logger.info("Organizing top 5 models in checkpoints folder...")
        
        top_models_summary = []
        successfully_saved = 0
        
        for i, model_info in enumerate(self.top_models, 1):
            trial_number = model_info['trial_number']
            validation_loss = model_info['validation_loss']
            params = model_info['params']
            
            source_path = checkpoint_dir / f"trial_{trial_number}_checkpoint.pt"
            
            if source_path.exists():
                try:
                    ranked_path = checkpoint_dir / f"rank_{i}_trial_{trial_number}_loss_{validation_loss:.6f}.pt"
                    
                    checkpoint = torch.load(source_path, map_location='cpu')
                    checkpoint['rank'] = i
                    checkpoint['total_trials'] = len(self.study.trials)
                    
                    torch.save(checkpoint, ranked_path)
                    
                    source_path.unlink()
                    
                    successfully_saved += 1
                    logger.info(f"Rank #{i}: Trial {trial_number}, Loss: {validation_loss:.6f} -> {ranked_path.name}")
                    
                    summary_info = {
                        'rank': i,
                        'trial_number': trial_number,
                        'validation_loss': validation_loss,
                        'file_name': ranked_path.name,
                        'parameters': params
                    }
                    top_models_summary.append(summary_info)
                    
                except Exception as e:
                    logger.error(f"Failed to process model for trial {trial_number}: {str(e)}")
                    # Keep the original file if processing failed
                    if ranked_path.exists():
                        ranked_path.unlink()
            else:
                logger.warning(f"Checkpoint file not found for trial {trial_number}: {source_path}")
        
        logger.info(f"Successfully saved {successfully_saved}/{len(self.top_models)} top models")
        
        # Save summary only if we saved at least one model
        if top_models_summary:
            summary_path = checkpoint_dir / 'top_5_models_summary.yaml'
            with open(summary_path, 'w') as f:
                yaml.dump(top_models_summary, f, default_flow_style=False)
            logger.info(f"Saved top models summary to {summary_path}")
        
        return top_models_summary
    
    def run_optimisation(self):
        """Run the full Optuna optimisation process."""
        logger.info(f"Starting Optuna optimisation with {self.config['optuna']['n_trials']} trials")
    
        try:
            logger.info(f"Running optimisation with {self.n_jobs} parallel workers")
    
            self.study.optimize(
                self.objective, 
                n_trials=self.config['optuna']['n_trials'],
                n_jobs=self.n_jobs,
                show_progress_bar=True,
                timeout=self.config.get('optuna', {}).get('timeout', None)
            )
    
            best_params = self.study.best_params
            best_value = self.study.best_value
            best_trial = self.study.best_trial
    
            logger.info(f"Best trial (#{best_trial.number}) achieved validation loss: {best_value}")
            logger.info("Best hyperparameters:")
            for key, value in best_params.items():
                logger.info(f"\t{key}: {value}")
    
            model_dir = Path(self.config['paths']['model_dir'])
            
            # Save best parameters
            with open(model_dir / 'best_params.yaml', 'w') as f:
                yaml.dump(best_params, f)
            
            # Save study results
            df_study = self.study.trials_dataframe()
            df_study.to_csv(model_dir / 'optuna_study_results.csv', index=False)
            
            # Plot and save training curves for best trial
            checkpoint_dir = model_dir / 'checkpoints'
            best_logger_path = checkpoint_dir / f"trial_{best_trial.number}_logger.pkl"
            if os.path.exists(best_logger_path):
                with open(best_logger_path, 'rb') as f:
                    self.best_trial_logger = pickle.load(f)
                plot_losses(self.best_trial_logger, model_dir, '_best_trial')
                logger.info(f"Saved loss plots for best trial to {model_dir}")

            # Save top 5 models in checkpoints folder
            top_models_summary = self._save_top_models(model_dir)
            
            # Clean up non-top-5 trial files
            top_trial_numbers = {model['trial_number'] for model in self.top_models}
            
            for file in checkpoint_dir.glob("trial_*_checkpoint.pt"):
                trial_num = int(file.stem.split('_')[1])
                if trial_num not in top_trial_numbers:
                    file.unlink()
                    
            for file in checkpoint_dir.glob("trial_*_logger.pkl"):
                trial_num = int(file.stem.split('_')[1])
                if trial_num not in top_trial_numbers:
                    file.unlink()
            
            logger.info(f"Cleaned up intermediate checkpoints. Kept {len(top_trial_numbers)} trial files.")
            
            # Load the best model checkpoint and recreate the model
            best_model_path = checkpoint_dir / f"rank_1_trial_{best_trial.number}_loss_{best_value:.6f}.pt"
            if best_model_path.exists():
                checkpoint = torch.load(best_model_path, map_location='cpu')
                
                model_config = checkpoint['model_config']
                final_model = ConditionalRealNVP(
                    input_dim=model_config['input_dim'],
                    hidden_dims=model_config['hidden_dims'],
                    covariate_net_dims=model_config['covariate_net_dims'],
                    c_dim=model_config['c_dim'],
                    n_layers=model_config['n_layers'],
                    learning_rate=model_config['learning_rate'],
                    weight_decay=model_config['weight_decay'],
                    use_transformation_layer=model_config.get('use_transformation_layer', False),
                    conditional_transform=model_config.get('conditional_transform', False),
                    transform_hidden_dims=model_config.get('transform_hidden_dims')
                )
                final_model.load_state_dict(checkpoint['model_state_dict'])
                
                device = torch.device("cuda" if self.config['device']['gpu'] and torch.cuda.is_available() else "cpu")
                final_model = final_model.to(device)
                
                logger.info(f"Loaded best model from trial {best_trial.number} and moved to {device}")
            else:
                logger.error("Best model checkpoint not found after reorganisation")
                raise FileNotFoundError(f"Could not find best model at {best_model_path}")
            
            return final_model, best_params
    
        except Exception as e:
            logger.error(f"Optimisation failed: {str(e)}")
            raise