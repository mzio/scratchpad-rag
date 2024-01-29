"""
Simple training script
"""
import os
from os.path import join
#test

import argparse
from omegaconf import OmegaConf

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from setup import init_wandb, seed_everything, update_config_from_args, flatten_config
from logging_utils import print_config, print_header, _format_arg

from model.pretrained import get_pretrained_loader

from dataloaders import load_data

from trainer import get_trainer, get_optimizer, get_scheduler
from evaluate.subspan_em import evaluate_mqa, plot_histogram_em, plot_lineplot_em


def get_args():
    parser = argparse.ArgumentParser()
    # Specify configs
    parser.add_argument("--project_name", type=str, default='scratchpad_rag')
    parser.add_argument("--experiment_config", type=str, default=None)
    parser.add_argument("--model_config", type=str, default=None)
    parser.add_argument("--peft_config", type=str, default=None)
    parser.add_argument("--train_method", type=str, default=None)

    # Override default configs
    ## Training
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--optim", type=str, default=None)
    parser.add_argument("--scheduler", type=str, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--num_train_epochs", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--num_train_steps_pos", type=int, default=None)

    ## Data
    parser.add_argument("--num_train_samples", type=int, default=None)
    parser.add_argument("--num_val_samples", type=int, default=None)
    
    ## Dataloading
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)

    ## Evaluation
    parser.add_argument("--retrieve_topk", type=int, default=None)
    parser.add_argument("--load_checkpoint", default=False, action='store_true')
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--eval_steps", type=int, default=None)
    parser.add_argument("--eval_only", default=False, action='store_true')
    parser.add_argument("--eval_split", type=str, default='val_anc')
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--eval_start", type=int, default=None)
    parser.add_argument("--eval_end", type=int, default=None)
    parser.add_argument("--last_answer_only", default=False, action='store_true')
    parser.add_argument("--print_outputs", default=False, action='store_true')

    # Misc.
    parser.add_argument("--output_dir", type=str, default='./checkpoints')
    parser.add_argument("--replicate", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_wandb", action='store_true', default=None)
    parser.add_argument("--wandb_entity", type=str, default='aunell')

    args = parser.parse_args()
    args.run_name = f'd={args.experiment_config}-m={args.model_config}-p={args.peft_config}-s={args.seed}'
    args.run_name = args.run_name.replace('True', '1').replace('False', '0')  # concise hacks
    return args


# Copied from https://github.com/facebookresearch/llama-recipes/blob/main/examples/quickstart.ipynb
def create_peft_config(model, peft_config: dict):
    if peft_config['method'] == 'lora':
        from peft import (
            get_peft_model,
            LoraConfig,
            TaskType,
            prepare_model_for_int8_training,
        )
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            **peft_config['kwargs'],
        )
        # prepare int-8 model for training
        model = prepare_model_for_int8_training(model)
        
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

        _dtype = model.base_model.model.model.layers[0].self_attn.k_proj.weight.dtype
        if _dtype is not torch.int8:
            model.to(dtype=torch.bfloat32)  # hardcode hack
        return model, peft_config
    else:
        raise NotImplementedError(f"Sorry PEFT method {peft_config['method']} not implemented yet.")


def set_eval_config_and_args(config, args):
    """Patchy updates for final evaluation"""
    if args.retrieve_topk is not None:
        args.run_name += f'-topk={args.retrieve_topk}'
        config.dataset.dataset_config['retrieve_topk'] = args.retrieve_topk
    
    if args.print_outputs is not None:
        config.evaluate.print_outputs = args.print_outputs

    if args.eval_split is None:
        args.eval_split = 'val_anc'
    args.run_name += f'-val={args.eval_split}'
    
    for arg in ['eval_start', 'eval_end']:
        argv = getattr(args, arg)
        setattr(config.evaluate, arg, argv)
        if argv is not None:
            args.run_name += f'-{_format_arg(arg)}={argv}'

    if args.last_answer_only:  # Should be true for scratchpad
        args.run_name += '-lao=1'
        config.evaluate.last_answer_only = args.last_answer_only

    if args.max_new_tokens is not None:  # Number of tokens to generate during eval
        args.run_name += f'-mnt={args.max_new_tokens}'
        config.evaluate.max_new_tokens = args.max_new_tokens
    
    return config, args


def main():
    args = get_args()
    args.output_dir = join(args.output_dir, args.model_config)
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    seed_everything(args.seed)
    device = torch.device('cuda:0')  # Assume one GPU

    # Load experiment configs
    experiment_config_path = join('./configs/experiment', f'{args.experiment_config}.yaml')
    experiment_config = OmegaConf.load(experiment_config_path)
    experiment_config = update_config_from_args(experiment_config, args)
    experiment_config, args = set_eval_config_and_args(experiment_config, args)

    # Load model configs
    model_config_path = join('./configs/model', f'{args.model_config}.yaml')
    model_config = OmegaConf.load(model_config_path)
    
    # Update tokenizer to match model
    for k in ['pretrained_model_name_or_path', 'cache_dir']:
        experiment_config.dataset.pretrained_model_config[k] = model_config.pretrained_config[k]

    # Specify scratchpad training or not -> updates the training dataset
    if args.train_method == 'scratchpad':
        experiment_config.dataset.dataset_config.include_support = True
        experiment_config.trainer.train_method = args.train_method
    args.run_name += f'-tm={args.train_method}'

    print_header('Experiment Config')
    print_config(experiment_config)
    print_header('Model Config')
    print_config(OmegaConf.create(model_config))

    # Override default run name with checkpoint
    if args.checkpoint_path is not None:
        args.run_name = args.checkpoint_path.split('/')[-1].split('.pt')[0]
    _, args = set_eval_config_and_args(experiment_config, args)  # Add back eval configs

    # Get data
    dataloaders  = load_data(experiment_config.dataset, experiment_config.dataloader)
    train_loader = dataloaders[experiment_config.trainer.train_split]
    eval_loader  = dataloaders[experiment_config.trainer.val_split]

    # Get pretrained model
    model_loader = get_pretrained_loader(**model_config['pretrained_config'])
    tokenizer = model_loader.load_tokenizer()
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'right'
    model = model_loader.load()
    

    if not args.eval_only or args.load_checkpoint:
        model.to(device)
        # Initialize PEFT configs
        model.train()
        peft_config = OmegaConf.load(join('./configs/peft', f'{args.peft_config}.yaml'))
        # model, lora_config = create_peft_config(model, peft_config)
   
        # Initialize optimizer and scheduler
        optimizer = get_optimizer(model=model, **experiment_config.optimizer)
        scheduler = get_scheduler(optimizer=optimizer, **experiment_config.lr_scheduler)
    else:
        lora_config = None
        optimizer, scheduler = None, None
        

    # WandB
    wandb = init_wandb(args)
    if wandb is not None:
        experiment_config['model'] = model_config  # Combine for logging
        _flattened = {'model': model_config,
                      'model_config': args.model_config,  # config file names
                      'experiment_config': args.experiment_config,
                      'peft_config': args.peft_config,
                    #   'lora': lora_config,
                      'replicate': args.replicate,
                      'eval_split': args.eval_split,}
        flatten_config(OmegaConf.to_container(experiment_config), _flattened, '')
        wandb.config.update(_flattened)

    # Load trainer 
    for arg, argv in experiment_config.trainer.items():
        if arg != 'name':
            setattr(args, arg, argv)
    for _config in ['dataloader', 'optimizer', 'lr_scheduler']:
        setattr(args, _config, OmegaConf.to_container(getattr(experiment_config, _config)))

    OurTrainer = get_trainer(experiment_config.trainer.name)
    trainer = OurTrainer(model=model, 
                         args=args,
                         train_loader=train_loader,
                         eval_loader=eval_loader,
                         optimizer_and_scheduler=(optimizer, scheduler),
                         device=device,
                         wandb=wandb,
                         max_eval_batches=experiment_config.trainer.max_eval_batches,
                         checkpoint_suffix='_lm',)
    if not args.eval_only:
        print_header(f'*** Training ***')
        print(f'├── Experiment name: {args.run_name}')
        print(f'├── Device: {device}')
        print(f'├── Seed: {args.seed}')
        model = trainer.train()

    if args.load_checkpoint:
        if args.checkpoint_path is not None:
            trainer.best_val_checkpoint_path = args.checkpoint_path
        state_dict = torch.load(trainer.best_val_checkpoint_path)['model_state_dict']
        model.load_state_dict(state_dict, strict=False)
        
    # Final evaluation
    # -> Adjust dataset if needed
    if experiment_config.dataset.dataset_config.include_support:
        experiment_config.dataset.dataset_config.include_support = False
        dataloaders = load_data(experiment_config.dataset, experiment_config.dataloader)
    
    eval_loader = dataloaders[args.eval_split]
    experiment_config.evaluate.max_samples = len(eval_loader)

    print_header(f'*** Evaluating ***')
    # todo: refactor to abstract this into trainer class, make more general
    # -> right now this is specific to exact match
    eval_metrics = evaluate_mqa(model, eval_loader, tokenizer, **experiment_config.evaluate)
    print_header('Final metrics')
    mean_em = sum(eval_metrics['subspan_em']) / len(eval_metrics['subspan_em'])
    print(f'├── Overall Subspan Exact Match: {mean_em:.4f}')

    logging_metrics = {'Overall EM': mean_em}
    # Slice by supporting document index
    em_by_doc_idx = plot_lineplot_em(eval_metrics, show_plot=False)
    for idx, doc_pos in enumerate(em_by_doc_idx['document_position']):
        _em = em_by_doc_idx['subspan_em'][idx]
        print(f'├── Support Document at {doc_pos} Supspan EM: {_em:.4f}')
        logging_metrics[f'Support at {doc_pos} EM'] = _em

    # Log metrics
    if wandb is not None:
        prefix = 'best_val'
        wandb.log({f'{prefix}/{k}': v for k, v in logging_metrics.items()})
    
    print_header(f'*** Finished ***')
    if not args.eval_only:
        print('├── Find checkpoint at:', trainer.best_val_checkpoint_path)


if __name__ == '__main__':
    main()
    