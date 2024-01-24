"""
General helper functions for setting up distillation experiments
"""
import os
import random
import numpy as np
import torch

from logging_utils import _format_arg


def init_wandb(args):
    if args.no_wandb:
        wandb = None
    else:
        import wandb
        print("entity", args.wandb_entity)
        print("run name", args.run_name)
        print("project name", args.project_name)
        wandb.init(config={},
                   entity=args.wandb_entity,
                   name=args.run_name,
                   project=args.project_name)
    return wandb


def seed_everything(seed):
    """
    Seed everything
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def flatten_config(config: dict, flattened: dict, key: str):
    """
    Recursive way to flatten config args for saving to WandB
    """
    for k, v in config.items():
        if type(v) is dict:
            flatten_config(v, flattened, f'{key}{k}_')
        elif type(v) is list:
            for ix, _config in enumerate(v):
                if type(_config) is dict:
                    flatten_config(_config, flattened, f'{key}{k}_{ix}_')
        else:
            flattened[f'{key}{k}'] = v
    return flattened

# Update configs
def update_config_from_args(config, args):
    """
    Quick hacks to override default configs
    """
    # Optimizer
    for arg in ['lr', 'weight_decay']:
        argval = getattr(args, arg)
        if argval is not None:
            setattr(config.optimizer, arg, argval)
            args.run_name += f'-{_format_arg(arg)}={argval}'
    if args.optim is not None:
        config.optimizer.method = args.optim
        args.run_name += f'-o={args.optim}'
    
    # Scheduler
    if args.scheduler is not None:
        config.lr_scheduler.lr_scheduler_type = args.scheduler
        args.run_name += f'-sc={args.scheduler}'

    # Control number of training samples
    if args.num_train_samples is not None:
        config.dataset.dataset_config.num_train_samples = args.num_train_samples
        args.run_name += f'-nts={args.num_train_samples}'
        config.lr_scheduler.num_training_steps = args.num_train_samples // config.trainer.gradient_accumulation_steps * 5
        config.lr_scheduler.num_warmup_steps = config.lr_scheduler.num_training_steps // 10
    
    if args.num_val_samples is not None:
        config.dataset.dataset_config.num_val_samples = args.num_val_samples
        args.run_name += f'-nvs={args.num_val_samples}'

    # Dataset
    for arg in [a for a in dir(args) if 'dataset_' in a]:
        argval = getattr(args, arg)
        if argval is not None:
            setattr(config.dataset.dataset_config, arg[len('dataset_'):], argval)
            args.run_name += f'-{_format_arg(arg)}={argval}'

    # Dataloader
    for arg in ['batch_size', 'num_workers']:
        argval = getattr(args, arg)
        if argval is not None:
            setattr(config.dataloader, arg, argval)
            args.run_name += f'-{_format_arg(arg)}={argval}'

    # Trainer
    for arg in ['gradient_accumulation_steps', 'num_train_epochs', 
                'max_steps', 'eval_steps']:
        argval = getattr(args, arg)
        if argval is not None:
            setattr(config.trainer, arg, argval)
            args.run_name += f'-{_format_arg(arg)}={argval}'
        
    return config
