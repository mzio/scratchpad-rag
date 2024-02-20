from collections import OrderedDict
from os.path import join
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def get_optimizer(optim: str, model: nn.Module, **kwargs: any):
    """
    Return PyTorch optimizer
    """
    if optim == 'sgd':
        return torch.optim.SGD(model.parameters(), **kwargs)
    elif optim == 'adam':
        return torch.optim.Adam(model.parameters(), **kwargs)
    elif optim in ['adamw', 'adamw_torch']:
        return torch.optim.AdamW(model.parameters(), **kwargs)
    elif optim == 'adamw_torch_fused':
        return torch.optim.AdamW(model.parameters(), **kwargs, fused=True) 
    else:
        raise NotImplementedError(f"{optim} optimizer not implemented sorry.")


def get_scheduler(lr_scheduler_type, optimizer, **kwargs: any):
    if lr_scheduler_type in ['plateau', 'reduce_lr_on_plateau']:
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        return ReduceLROnPlateau(optimizer=optimizer, **kwargs)
    
    elif lr_scheduler_type == 'cosine_warmup':
        from transformers import get_cosine_schedule_with_warmup
        return get_cosine_schedule_with_warmup(optimizer=optimizer, **kwargs)
    
    elif lr_scheduler_type in ['linear_warmup', 'linear']:
        from transformers import get_linear_schedule_with_warmup
        return get_linear_schedule_with_warmup(optimizer=optimizer, **kwargs)
    
    elif lr_scheduler_type == 'timm_cosine':
        from timm.scheduler.cosine_lr import CosineLRScheduler
        return CosineLRScheduler(optimizer=optimizer, **kwargs)
    else:
        return None


class OurTrainer():
    """
    Basic parent trainer class. Defaults to language modeling. 
    -> Replacement for HuggingFace Trainer
    """
    def __init__(self, model, args, 
                 train_loader: DataLoader, 
                 eval_loader: DataLoader, 
                 optimizer_and_scheduler: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler], 
                 device: torch.device, 
                 wandb,  # WandB object
                 max_eval_batches: int = -1,
                 checkpoint_suffix: str = None,
                 **kwargs: any):
        super().__init__()
        self.model = model
        self.args = args

        if optimizer_and_scheduler is None:
            self.optimizer = get_optimizer(model=self.model, **self.args.optimizer)
            self.scheduler = get_scheduler(optimizer=self.optimizer, **self.args.lr_scheduler)
        else:
            self.optimizer, self.scheduler = optimizer_and_scheduler
        self.scheduler_step_after_epoch = 'plateau' in args.lr_scheduler['lr_scheduler_type']

        # Dataloaders
        self.train_loader = train_loader
        self.eval_loader = eval_loader

        self.device = device
        self.wandb = wandb

        self.init_checkpointing(args, checkpoint_suffix=checkpoint_suffix)
        self.is_better = (lambda x, y: x > y if self.args.greater_is_better 
                          else x < y)
        # Custom arguments
        self.max_eval_batches = self.args.max_eval_batches

        self.train_metrics = {'train/loss': None, 
                              'train/epoch': None, 
                              'train/step': None}
        self.eval_metrics = {'eval/loss': None}
        self.criterion = nn.CrossEntropyLoss(reduction='mean')

    def train(self):
        """
        Entire training run
        """
        model = self.model
        pbar = tqdm(range(self.args.num_train_epochs), leave=False, colour='white',
                    desc=f'Training')

        self.step = 0  # Total steps taken
        self.grad_step = 0  # Total gradient updates
        for epoch in pbar:
            model = self.train_step(model, epoch)
            if self.args.evaluation_strategy == 'epoch':
                model = self.eval_step(model, step=self.grad_step)
                   
        if self.args.load_best_model_at_end:  # Return best checkpoint
            try:
                state_dict = torch.load(self.best_val_checkpoint_path)['model_state_dict']
                model.load_state_dict(state_dict, strict=False)
            except Exception as e:
                print(e)
                print('-> Returning most recent model instead')
        return model            

    def train_step(self, model: nn.Module, epoch: int):
        """
        Training loop for one data epoch
        """
        # Gradient accumulation
        if self.args.gradient_accumulation_steps is None:
            accum_iter = 1
        else:
            accum_iter = self.args.gradient_accumulation_steps

        model.train()
        model.zero_grad()
        # model.to(self.device)
        
        pbar = tqdm(self.train_loader, leave=False, colour='blue', 
                    desc=f'-> Training (epoch {epoch} / {self.args.num_train_epochs})')
        total_loss = 0

        eval_for_step = False
        model.to(self.device)
        for ix, data in enumerate(pbar):
            loss, train_metrics = self.compute_loss(model, data, return_outputs=True)
            loss /= accum_iter
            loss.backward()

            if (self.step + 1) % accum_iter == 0:  # and self.step != 0:
                self.optimizer.step()
                if not self.scheduler_step_after_epoch and self.scheduler is not None:
                    self.scheduler.step()
                self.optimizer.zero_grad()
                self.grad_step += 1

            self.step += 1
            loss = loss.cpu()
            total_loss += loss.item()
            desc = f"Training epoch {epoch} | loss: {total_loss / (ix + 1):.3f} | lr: {self.optimizer.param_groups[0]['lr']:.5f}"
            desc += f' | gradient step: {self.grad_step}'
            for k, v in train_metrics.items():
                try:
                    desc += f' | {k}: {v:.3f}'
                except:
                    desc += f' | {k}: {v}'
                    
            pbar.set_description(desc)

            # Logging
            if (self.grad_step) % (self.args.logging_steps):
                self.train_metrics['train/loss'] = loss.item()
                self.train_metrics['train/epoch'] = epoch
                self.train_metrics['train/step'] = self.grad_step
                self.train_metrics['train/lr'] = self.optimizer.param_groups[0]['lr']
                for k, v in train_metrics.items():
                    self.train_metrics[f'train/{k}'] = v
                
                if self.wandb is not None:
                    self.wandb.log(self.train_metrics, step=self.grad_step)

            if self.args.evaluation_strategy == 'steps':
                if self.grad_step % self.args.eval_steps == 0 and self.grad_step > 0 and not eval_for_step:
                    model = self.eval_step(model, step=self.grad_step)
                    eval_for_step = True
                elif self.grad_step % self.args.eval_steps == 0 and self.grad_step > 0 and eval_for_step:
                    pass
                else:
                    eval_for_step = False
                # model.cpu()

            if self.step == self.args.max_steps:
                break
        return model

    def eval_step(self, model: nn.Module, **kwargs: any):
        """
        Evaluation step where we also save metrics
        """
        self.eval_metrics = self.evaluate(model, **kwargs)
        val_metric = self.eval_metrics[self.metric_for_best_model]
        # Save best metric and checkpoint
        if self.is_better(val_metric, self.best_val_metric):
            self.best_val_metric = val_metric
            self.best_val_metric_step = self.grad_step
            torch.save({
                'model_state_dict': self.save_trainable_weights(model),
                'step': self.grad_step, 
                self.metric_for_best_model: val_metric
            }, self.best_val_checkpoint_path)
            print(f'\n-> Saved best model checkpoint to: {self.best_val_checkpoint_path}!')
        
        if self.scheduler_step_after_epoch and self.scheduler is not None:
            self.scheduler.step(val_metric)
        print(self.eval_metrics)

        # Logging
        if self.wandb is not None:
            self.wandb.log(self.eval_metrics, step=self.grad_step)
        return model

    def evaluate(self, model: nn.Module, step: int, 
                 max_batches: int = None, 
                 dataloader: DataLoader = None):
        """
        One evaluation loop over a validation dataset
        """
        max_batches = (self.max_eval_batches if max_batches is None 
                       else max_batches)
        dataloader = self.eval_loader if dataloader is None else dataloader
        pbar = tqdm(dataloader, leave=False, colour='green',
                    desc=f'Evaluating at step {step}')
        
        model.eval()
        model.to(self.device)
        total_loss = 0
        with torch.no_grad():
            for ix, data in enumerate(pbar):
                loss, eval_metrics = self.compute_loss(model, data, return_outputs=True)
                loss = loss.cpu()
                self.eval_metrics['eval/loss'] = loss.item()
                for k, v in eval_metrics.items():
                    self.eval_metrics[f'eval/{k}'] = v
                total_loss += loss.item()
                desc = f"Evaluating at step {step} | loss: {total_loss / (ix + 1):.3f} | lr: {self.optimizer.param_groups[0]['lr']:.5f}"
                pbar.set_description(desc)
                if ix == max_batches:
                    break
        return self.eval_metrics

    def compute_loss(self, model: nn.Module, data: torch.Tensor, 
                     return_outputs: bool = False, **kwargs: any):
        """
        Main method to determine how models are trained. 
        -> Defaults to next-token prediction / classification, 
           but override in child classes

        return_outputs is True will depend on logic below
        """
        input_keys = {'input_ids', 'attention_mask'}
        inputs = {k: v.to(model.device) for k, v in data.items() 
                  if k in input_keys}  
        outputs = model(**inputs, output_attentions=False)
        outputs = outputs.get('logits')[..., :-1, :].contiguous()
        targets = data.get('labels')[..., 1:].contiguous()

        # Flatten and compute cross-entropy loss
        outputs = outputs.view(-1, outputs.shape[-1])
        targets = targets.view(-1).to(outputs.device)
        loss = self.criterion(outputs, targets)

        targets = targets.cpu()
        outputs = outputs.cpu()

        if return_outputs:
            return loss, {}
        else:
            return loss

    def save_trainable_weights(self, model: nn.Module):
        """
        Save checkpoint with only weights actively being trained (e.g., for adapters). 
        Make sure to later load with model.load_state_dict(state_dict, strict=False)
        """
        state_dict = OrderedDict()
        for n, p in model.named_parameters():
            if p.requires_grad:
                state_dict[n] = p
        return state_dict
        
    def init_checkpointing(self, args, checkpoint_suffix):
        """
        Initialize checkpoint attributes

        Inputs:
        - args: Argparse or HuggingFace TrainingArguments object
        """
        self.best_val_checkpoint_path = f'{join(args.output_dir, args.run_name)}.pt'
        if checkpoint_suffix is not None:
            self.best_val_checkpoint_path = self.best_val_checkpoint_path.replace(
                '.pt', f'{checkpoint_suffix}.pt')
        print(f'-> Saving best model checkpoint to {self.best_val_checkpoint_path}')

        # Best metric setup
        self.best_val_metric = 0 if args.greater_is_better else 1e10
        self.best_val_metric_epoch = 0
        self.best_val_metric_step = 0
        self.best_train_metric = 0 if args.greater_is_better else 1e10
        self.best_train_metric_epoch = 0
        self.best_train_metric_step = 0
        self.metric_for_best_model = args.metric_for_best_model
        if 'eval' not in self.metric_for_best_model:
            self.metric_for_best_model = f'eval/{self.metric_for_best_model}'
        