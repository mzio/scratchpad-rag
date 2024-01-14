"""
Scratchpad RAG demo

python demo.py --experiment_config nq_lim_20_docs --model_config mistral_7b_instruct

(cmd 5)
python demo.py --experiment_config nq_lim_20_docs --model_config mistral_7b

(cmd 3)
python demo.py --experiment_config nq_lim_20_docs --model_config llama2_7b_32k_instruct

(cmd 4)
python demo.py --experiment_config nq_lim_20_docs --model_config mistral_7b --checkpoint_path ./checkpoints/scratchpad_rag.pt
"""
from os.path import join
from xopen import xopen
import json

import argparse
from omegaconf import OmegaConf
# from dataloaders import load_data

import torch
from transformers import AutoTokenizer

from dataloaders.utils import train_test_split, subsample_split, convert_to_hf_dataset
from setup import seed_everything
from model.pretrained import get_pretrained_loader
from logging_utils import print_header


def load_data(data_config):
    """
    Load dataloaders for Natural Questions for Lost-in-the-middle
    """
    name = f"{data_config['name']}_gold_at"
    dataset_config = data_config['dataset_config']
    pretrained_model_config = data_config['pretrained_model_config']
    preprocess_config = data_config['preprocess_config']
    tokenizer_name = pretrained_model_config['pretrained_model_name_or_path']
    tokenizer_name = tokenizer_name.split('/')[-1]

    if dataset_config['num_train_samples'] is not None:
        name += f"-nts={dataset_config['num_train_samples']}"
    if dataset_config['num_val_samples'] is not None:
        name += f"-nvs={dataset_config['num_val_samples']}"
    if dataset_config['include_support']:
        name += f'-is=1'

    # Misc. setup
    cache_dir   = dataset_config['cache_dir']
    total_docs  = dataset_config['total_docs']
    train_ratio = dataset_config['train_ratio']
    seed = dataset_config['seed']

    # Get initial data
    dataset = []
    for gold_at in dataset_config['gold_at']:
        name += f'_{gold_at}'
        data_file = f'nq-open-{total_docs}_total_documents_gold_at_{gold_at}.jsonl.gz'
        with xopen(join(cache_dir, data_file)) as f:
            for line in f:
                dataset.append(json.loads(line))

    # Split into train and val sets
    train_size = int(len(dataset) * train_ratio)
    val_size = len(dataset) - train_size
    train_set, val_set = train_test_split(dataset, train_size, val_size, seed)
    
    if dataset_config['num_val_samples'] is not None:
        val_set = subsample_split(val_set, dataset_config['num_val_samples'], seed)

    # Consistency with preprocessing and tokenizing maps later
    val_set = convert_to_hf_dataset(val_set, cache_dir)
    val_set = val_set.map(process_sample, 
                          remove_columns=list(val_set.features),
                          load_from_cache_file=False)
    return val_set


def process_sample(sample: dict):
    """
    Convert original data from file to consistent dictionary
    """
    support = []
    context = []
    support_indices = []  # Which result they showed up in

    for ix, ctx in enumerate(sample['ctxs']):
        text = ctx['text']
        title = ctx['title']
        if ctx['hasanswer'] or ctx['isgold']:
            support.append(
                {'title': ctx['title'], 'text': ctx['text']}
            )
            support_indices.append(ix)
        context.append(
            {'title': ctx['title'], 'text': ctx['text']}
        )
    sample = {
        'question': sample['question'],
        'answer': sample['answers'][0],
        'context': context,
        'support': support,
        'support_indices': support_indices,
    }
    return sample


def template_and_tokenize(sample, tokenizer, instruct_tune):
    """
    Convert RAG training samples into a single input text and tokenize
    """
    template = f"""Write a high-quality answer for the given question using only the provided context (some of which might be irrelevant).

Question: {{question}}

Context:
{{context}}
    
Question: {{question}}

Answer:"""
    if instruct_tune:
        template = '[INST] ' + template + ' [/INST]'
    context = []
    for ix, c in enumerate(sample['context']):
        context.append(f"Document (Title: {c['title']}) {c['text']}")
    context = '\n\n'.join(context)
    
    prompt = template.format(context=context, question=sample['question'].capitalize())
    prompt = f'{tokenizer.bos_token}{prompt}'
    prompt = tokenizer.encode(prompt, add_special_tokens=False, return_tensors='pt')
    
    target = tokenizer.encode(f'{sample["answer"]}{tokenizer.eos_token}', add_special_tokens=False,
                              return_tensors='pt')

    attn_mask = torch.ones(len(prompt))

    sample =  {
        "input_ids": prompt,
        "attention_mask" : attn_mask,
        "labels": target,
    }
    return sample


def generate_response(model, prompt, tokenizer, max_new_tokens: int):
    model.eval()
    with torch.no_grad():
        model_input = tokenizer(prompt, return_tensors='pt').to(model.device)
        output = tokenizer.decode(model.generate(**model_input, max_new_tokens=max_new_tokens)[0], 
                                  skip_special_tokens=True)
        output = output.split('Answer:')[-1]
        return output
    # model_input_args = ['input_ids', 'attention_mask', 'labels']
    # model.eval()
    # with torch.no_grad():
    #     _input_ids = data['input_ids']
    #     breakpoint()
    #     outputs = model.generate(**data,
    #         # **{k: v.to(model.device) for k, v in data.items()
    #            # if k in model_input_args},
    #         max_new_tokens=max_new_tokens,
    #         pad_token_id=tokenizer.eos_token_id,
    #     )
    #     for sample_idx in range(len(_input_ids)):
    #         outputs = outputs[sample_idx:sample_idx+1, len(_input_ids[sample_idx]):]
    #     outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    # return outputs


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
            model.to(dtype=torch.bfloat16)  # hardcode hack
        return model, peft_config
    else:
        raise NotImplementedError(f"Sorry PEFT method {peft_config['method']} not implemented yet.")

def get_args():
    parser = argparse.ArgumentParser()
    # Specify configs
    parser.add_argument("--experiment_config", type=str, default='nq_lim_20_docs')
    parser.add_argument("--model_config", type=str, default='mistral_7b')
    parser.add_argument("--peft_config", type=str, default='lora_r8_a16_qv')
    parser.add_argument("--max_samples", type=int, default=10)
    parser.add_argument("--train_method", type=str, default=None)
    parser.add_argument("--retrieve_topk", type=int, default=None)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    seed_everything(args.seed)
    device = torch.device('cuda:0')  # Assume one GPU

    # Load model
    model_config = OmegaConf.load(join('./configs/model', f'{args.model_config}.yaml'))
    model_loader = get_pretrained_loader(**model_config['pretrained_config'])
    tokenizer = model_loader.load_tokenizer()
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token
    print(f'Setting tokenizer.pad_token to {tokenizer.pad_token}')
    tokenizer.padding_side = 'left'  # for decoder-only generation

    print_header(f'Loading {args.model_config} model...')
    model = model_loader.load()

    # Load checkpoint 
    if args.checkpoint_path is not None:
        model.to(device)
        # Initialize PEFT configs
        model.train()
        peft_config = OmegaConf.load(join('./configs/peft', f'{args.peft_config}.yaml'))
        model, lora_config = create_peft_config(model, peft_config)
        state_dict = torch.load(args.checkpoint_path)['model_state_dict']
        model.load_state_dict(state_dict, strict=False)
        model.eval()

    # Load evaluation data
    experiment_config = OmegaConf.load(join('./configs/experiment', f'{args.experiment_config}.yaml'))
    print('-> Loading RAG eval dataset...')                          
    val_set = load_data(experiment_config.dataset)

    tokenizer = AutoTokenizer.from_pretrained(**experiment_config.dataset.pretrained_model_config)
    instruct_tune = True if 'instruct' in args.model_config else False

    # Load samples
    start_ix = 0
    for ix in range(args.max_samples):
        sample = val_set.__getitem__(start_ix)
        question = sample['question']
        if question[-1] != '?': 
            question += '?'  # Add punctuation
        print_header('Question 1:') 
        print(question)
        
        topk = int(input(f'>> To answer, how many docs to retrieve? (max 20) '))
        if topk == 1:  # only bring up support
            context = sample['support']
        else:
            context = sample['context']
        sample['context'] = [c for ix, c in enumerate(context) if ix < topk]
    
        print_header('Model input prompt')
        tokenized_sample = template_and_tokenize(sample, tokenizer, instruct_tune)
        decoded_sample = tokenizer.batch_decode(tokenized_sample['input_ids'])[0]
        print(decoded_sample)
    
        print_header("Model response")
        max_new_tokens = input(f'>> To answer, how many tokens to generate? (default 128) ')
        if max_new_tokens == '':
            max_new_tokens = 128
        # _ = input(f'>> (Press enter to display) ')
        output = generate_response(model, decoded_sample, tokenizer, int(max_new_tokens))
        print('Model response:')
        print(output)
    
        _ = input(f'>> (Press enter to display true answer) ')
        true_answer = tokenizer.batch_decode(tokenized_sample['labels'], skip_special_tokens=True)[0]
        print('True answer:', true_answer)

        move_to_next = input(f'>> Move to next question? (y / n) ')
        if move_to_next == 'y':
            start_ix += 1
        

if __name__ == '__main__':
    main()

