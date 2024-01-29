"""
Script to evaluate trained (or zero-shot) model across various top-k document settings
"""
import os
from os.path import join

import argparse
from omegaconf import OmegaConf

import torch
import numpy as np
import pandas as pd
from logging_utils import print_header, print_config

from dataloaders import load_data
from model.pretrained import get_pretrained_loader
from evaluate.subspan_em import evaluate_mqa, plot_lineplot_em
from main import create_peft_config
from main import set_eval_config_and_args

from setup import init_wandb, seed_everything, flatten_config


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_name", type=str, default='scratchpad_rag_eval')
    parser.add_argument("--model_config", type=str, default=None)
    parser.add_argument("--peft_config", type=str, default=None)
    parser.add_argument("--load_checkpoint", default=False, action='store_true')
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--eval_start", type=int, default=0)
    parser.add_argument("--eval_end", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_wandb", action='store_true', default=None)
    parser.add_argument("--wandb_entity", type=str, default='aunell')
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--last_answer_only", action='store_true', default=None)
    
    args = parser.parse_args()
    args.run_name = f'nq_lim_eval-m={args.model_config}-p={args.peft_config}-s={args.seed}'
    for k in ['eval_start', 'eval_end', 'max_new_tokens', 'last_answer_only']:
        if getattr(args, k) is not None:
            args.run_name += f'-{k}={getattr(args, k)}'
    args.run_name = args.run_name.replace('True', '1').replace('False', '0')  # concise hacks
    return args


def main():
    args = get_args()
    args.output_dir = join('./results', args.model_config)
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    seed_everything(args.seed)
    device = torch.device('cuda:0')  # Assume one GPU

    # Load model
    model_config_path = join('./configs/model', f'{args.model_config}.yaml')
    model_config = OmegaConf.load(model_config_path)
    print_header('Model Config')
    print_config(OmegaConf.create(model_config))

    # Get pretrained model
    model_loader = get_pretrained_loader(**model_config['pretrained_config'])
    tokenizer = model_loader.load_tokenizer()
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'left'
    model = model_loader.load()

    # Load checkpoint if applicable
    if args.load_checkpoint and args.checkpoint_path is not None:
        # Add LoRA params (we didn't merge)
        model.to(device)
        # Initialize PEFT configs
        model.train()
        peft_config = OmegaConf.load(join('./configs/peft', f'{args.peft_config}.yaml'))
        model, lora_config = create_peft_config(model, peft_config)
        
        state_dict = torch.load(args.checkpoint_path)['model_state_dict']
        model.load_state_dict(state_dict, strict=False)
        # Override default run name with checkpoint
        args.run_name = args.checkpoint_path.split('/')[-1].split('.pt')[0]
        for k in ['eval_start', 'eval_end', 'max_new_tokens', 'last_answer_only']:
            if getattr(args, k) is not None:
                args.run_name += f'-{k}={getattr(args, k)}'
        args.run_name = args.run_name.replace('True', '1').replace('False', '0')  # concise hacks
    
    eval_config = f"""
max_new_tokens: {args.max_new_tokens}
max_samples: 1000
negative_sample: false
print_outputs: true
eval_start: {args.eval_start}
eval_end: {args.eval_end}
"""
    eval_config = OmegaConf.create(eval_config)
    all_eval_metrics = {}

    # WandB
    wandb = init_wandb(args)

    results_avg = {'n_docs': [],
                   'subspan_em': []}
    results_lim = {'n_docs': [],
                   'document_position': [],
                   'subspan_em': []}

    for n_docs in [10,20]: #[1, 5, 10, 20]:
        if n_docs == 1:
            config_path = f'./configs/experiment/nq_lim_5_docs.yaml'
        else:
            config_path = f'./configs/experiment/nq_lim_{n_docs}_docs.yaml'
        config = OmegaConf.load(config_path)
        # Update tokenizer to match model
        for k in ['pretrained_model_name_or_path', 'cache_dir']:
            config.dataset.pretrained_model_config[k] = model_config.pretrained_config[k]
        dataloaders = load_data(config['dataset'], config['dataloader'])

        print_header(f'*** Evaluating top-{n_docs} doc retrieval ***')
        
        eval_loader = dataloaders['val_anc' if n_docs != 1 else 'val_pos']
        eval_metrics = evaluate_mqa(model, eval_loader, tokenizer, **eval_config)
    
        mean_em = np.mean(eval_metrics['subspan_em'])
        print(f'-> Overall Subspan Exact Match: {mean_em:.4f}')
    
        all_eval_metrics[n_docs] = eval_metrics

        results_avg['n_docs'].append(n_docs)
        results_avg['subspan_em'].append(mean_em)
        
        # Slice by supporting document index
        em_by_doc_idx = plot_lineplot_em(eval_metrics, 'support_doc_index',
                                         f'well well well... lost in the middle?', 
                                         f'{n_docs} docs', show_plot=False)
        logging_metrics = {}
        for k, v in em_by_doc_idx.items():
            logging_metrics[f'n_docs={n_docs}/{k}'] = v

        for ix, em in enumerate(em_by_doc_idx['subspan_em']):
            results_lim['n_docs'].append(n_docs)
            results_lim['document_position'].append(em_by_doc_idx['document_position'][ix])
            results_lim['subspan_em'].append(em)
        wandb.log(logging_metrics)
            
        for idx, doc_pos in enumerate(em_by_doc_idx['document_position']):
            _em = em_by_doc_idx['subspan_em'][idx]
            print(f'-> Support Document at {doc_pos} Supspan EM: {_em:.4f}')

        # Save data (do for each # of docs just in case job cancels)
        pd.DataFrame(results_avg).to_csv(join(args.output_dir, f'results_avg_{args.run_name}.csv'))
        pd.DataFrame(results_lim).to_csv(join(args.output_dir, f'results_lim_{args.run_name}.csv'))

    logging_metrics = {}
    for k, v in all_eval_metrics.items():
        logging_metrics[str(k)] = v
    wandb.log(logging_metrics)


if __name__ == '__main__':
    main()
    