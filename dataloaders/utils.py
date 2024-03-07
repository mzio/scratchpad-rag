"""
Shared dataset helper functions
"""
import os
from os.path import join, isdir
from typing import Callable
from functools import partial

import numpy as np
import pandas as pd

from datasets import load_from_disk
from datasets import Dataset as HFDataset
from huggingface_hub import hf_hub_download

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, LlamaTokenizer
from transformers import DataCollatorForSeq2Seq, DefaultDataCollator


def convert_to_hf_dataset(dataset, cache_dir: str):
    def gen():
        for idx in range(len(dataset)): yield dataset[idx]

    return HFDataset.from_generator(gen, cache_dir=cache_dir)


def get_tokenizer_from_config(model_config):
    # Get tokenizer
    if 'LLaMA' in model_config['pretrained_model_name_or_path']:
        model_path = join(model_config['cache_dir'], 
                          model_config['pretrained_model_name_or_path'])
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
        # LlamaTokenizer.from_pretrained(model_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(**model_config)
    return tokenizer


def train_test_split(samples: any, train_size: int, test_size: int, seed: int):
    try:
        assert len(samples) == train_size + test_size
    except Exception as e:
        print(len(samples), train_size + test_size)
        raise e
    arange = np.arange(len(samples))
    np.random.seed(seed)
    test_idx = np.random.choice(arange, size=test_size, replace=False)
    train_idx = np.setdiff1d(arange, test_idx)
    return ([samples[_idx] for _idx in train_idx], 
            [samples[_idx] for _idx in test_idx])


def subsample_split(samples: any, num_samples: int, seed: int):
    arange = np.arange(len(samples))
    np.random.seed(seed)
    idx = np.random.choice(arange, size=num_samples, replace=False).tolist()
    try:
        # return samples[idx]
        return [samples[int(_idx)] for _idx in idx]
    except Exception as e:
        print(e)
        breakpoint()


def get_seq2seq_loader(dataset: Dataset, tokenizer: AutoTokenizer, 
                       split: str, **loader_kwargs: any):
    """
    Get dataloader for seq2seq tasks (evaluation)
    """
    collate_fn = DataCollatorForSeq2Seq(
        tokenizer, label_pad_token_id=-100, return_tensors='pt')
    return DataLoader(
        # dataset, shuffle='train' in split, collate_fn=collate_fn, **loader_kwargs)
        dataset, shuffle=True, collate_fn=collate_fn, **loader_kwargs)  # for evals


def get_lm_loader(dataset: Dataset, tokenizer: AutoTokenizer, 
                  split: str, **loader_kwargs: any):
    """
    Get dataloader for language modeling (training)
    """
    collate_fn = DefaultDataCollator(return_tensors='pt')
    return DataLoader(
        dataset, shuffle=True,  # 'train' in split, 
        collate_fn=collate_fn, **loader_kwargs)
    

def get_target_index(context: list[int], target: list[int]):
    """
    Helper function to find where a target occurs in a context
    """
    for ix in range(len(context) - len(target) + 1):
        if context[ix: ix + len(target)] == target:
            return ix, ix + len(target)
    return None


def tokenize_dataset(dataset: Dataset, split_name: str,                      
                     tokenize_func: Callable,
                     cache_dir: str,
                     tokenizer_name: str,
                     **tokenize_kwargs: any):
    """
    Apply prompt formatting and tokenize dataset
    """
    # Note that we have robust_context as a hack for sharing dataset path (saving memory from prior project)
    save_path = join(cache_dir, f'robust_context.{tokenizer_name}_{split_name}')  # 'train_anc'
    try:    
        dataset = load_from_disk(save_path)
        print(f'Tokenized dataset loaded from {save_path}!')
    except Exception as e:
        print(e)
        if not isdir(save_path):
            os.makedirs(save_path)
            print(f'-> Created {save_path}')
            print(f'-> Tokenizing {split_name} dataset...')
            #for some reason, no adjustments to tokenizer previously change this tokenizer in tokenize_kwargs
            dataset = dataset.map(partial(tokenize_func, **tokenize_kwargs),
                                  remove_columns=list(dataset.features),
                                  load_from_cache_file=False)
            # count_discard = sum(1 for item in dataset if item.get('discard_sample', False))
            data = [x for x in dataset if x["discard_sample"] == False]
            dataset = HFDataset.from_pandas(pd.DataFrame(data=data))
            # dataset = dataset.filter(lambda x: x['discard_sample']==False) #broken??
            # print('🪿dataset',dataset)
            try:
                dataset.save_to_disk(save_path)
                print(f'Tokenized {split_name} dataset saved to {save_path}!')
            except Exception as e:
                print(e)
                print('Not saving tokenized dataset...')        
    return dataset


def tokenize_add_label(sample: dict, tokenizer: AutoTokenizer, 
                       context_source: str='context', 
                       eval_type: str = 'em',
                       include_label: bool=True,
                       instruct_tune: bool=False,
                       include_support: bool=False,
                       truncation: bool=False,
                       padding: bool=False, 
                       max_length: int=3584):
    """
    Convert RAG training samples into a single input text and tokenize
    """
    #flag from checking if negative context is empty, so we need to discard sample
    discard_early=False 

    question = sample['question']
    if question[-1] != '?': question += '?'  # Add punctuation
    template = f"""Write a high-quality answer for the given question using only the provided context (some of which might be irrelevant).

Question: {{question}}

Context:
{{context}}
    
Question: {{question}}

Answer:"""
    if instruct_tune:
        template = '[INST] ' + template + ' [/INST]'
    context = []
    if context_source != 'val_closed_book':
        for ix, c in enumerate(sample[context_source]):
            context.append(f"Document (Title: {c['title']}) {c['text']}")
        if (context_source=='context' and len(sample[context_source])==1):
            discard_early=True
        context = '\n\n'.join(context)
    else:
        context = ''
    
    prompt = template.format(context=context, question=sample['question'].capitalize())
    prompt = f'{tokenizer.bos_token}{prompt}'
    prompt = tokenizer.encode(prompt, add_special_tokens=False, truncation = truncation, max_length=max_length)
    final_answer= sample["answer"]
    if include_support:  # include support in answer to complete
        sample["answer"] = f"{sample['question'].capitalize()}\n\n" + sample["answer"]

    # Add supporting context positions -> position where they end in context
    support_token_indices = []
    support_token_start, support_token_end, support_doc= [], [], []
    for ix, c in enumerate(sample['support']):  # Get positions of 
        support = f"\nDocument (Title: {c['title']}) {c['text']}\n"
        try: #for TriviaQA case
            for ans in sample['all_answers']:
                if ans in support:
                    discard_early=False
                    break
                discard_early=True
        except:
            pass
        support = tokenizer.encode(support, add_special_tokens=False, truncation = truncation, max_length=max_length)[1:]
        try:
            start, end = get_target_index(prompt, support)
        except:
            start, end = 0, 0
        support_token_indices.append((start, end))
        support_token_start.append(start)
        support_token_end.append(end)
        support_doc.append(support)
        if include_support:  # include support in answer to complete
            sample["answer"] = f"Document (Title: {c['title']}) {c['text']}\n\n" + sample["answer"]
        if eval_type == 'doc_retrieval':
            if len(sample['support'])>1:
                # print('hotpotQA data')
                sample["answer"]=""
                for ix, c in enumerate(sample['support']):
                    sample["answer"]+=f"Document (Title: {c['title']}) {c['text']}\n\n"
            else:
                sample["answer"] = f"Document (Title: {c['title']}) {c['text']}\n\n"
        if eval_type == 'multi':
            if len(sample['support'])>1:
                print('hotpotQA data')
                sample["answer"]=""
                for ix, c in enumerate(sample['support']):
                    sample["answer"]+=f"Document (Title: {c['title']}) {c['text']}\n\n"
            else:
                sample["answer"] = f"Document (Title: {c['title']}) {c['text']}\n\n"
            # print('final answer', final_answer)
            sample["answer"]= sample["answer"] + final_answer
    if include_label:
        answer = tokenizer.encode(f'{sample["answer"]}{tokenizer.eos_token}', add_special_tokens=False, truncation = truncation, max_length=max_length) 
    else:
        answer = []
        # if eval_type=='doc_retrieval':
        #     target = tokenizer.encode(f'{sample["answer"]}{tokenizer.eos_token}', add_special_tokens=False, truncation = truncation, max_length=max_length)
        # else:
        try: #for trivia QA case where many answer targets
                all_answers_string = ','.join(sample["all_answers"])
                if eval_type=='multi':
                    target_pre_tokenized = sample['answer']+all_answers_string
                    target = tokenizer.encode(f'{target_pre_tokenized}{tokenizer.eos_token}', add_special_tokens=False, truncation = truncation, max_length=max_length)
                else:
                    target = tokenizer.encode(f'{all_answers_string}{tokenizer.eos_token}', add_special_tokens=False, truncation = truncation, max_length=max_length)
        except:
                target = tokenizer.encode(f'{sample["answer"]}{tokenizer.eos_token}', add_special_tokens=False, truncation = truncation, max_length=max_length)
    if eval_type== 'doc_retrieval':
        target = tokenizer.encode(f'{sample["answer"]}{tokenizer.eos_token}', add_special_tokens=False, truncation = truncation, max_length=max_length)
    input_ids = prompt + answer
    if len(input_ids) == max_length or discard_early:
        discard=True
    else:
        discard=False
    attn_mask = [1] * len(input_ids)
    # Negative sample -> just mask out supporting context
    negative_ids = prompt + answer
    negative_attn_mask = [1] * len(negative_ids)
    for indices in support_token_indices:
        neg_len = indices[1] - indices[0]
        negative_ids[indices[0]:indices[1]] = [tokenizer.pad_token_id] * neg_len
        negative_attn_mask[indices[0]:indices[1]] = [0] * neg_len
    
    sample =  {
        "input_ids": input_ids,
        "attention_mask" : attn_mask,
        "negative_ids": negative_ids,
        "negative_attention_mask": negative_attn_mask,
        "labels": [-100] * len(prompt) + answer if include_label else target,
        "support_indices": sample['support_indices'],
        "support_token_indices": support_token_indices,
        "support_token_start": support_token_start,
        "support_token_end": support_token_end,
    }
    return {**sample, "discard_sample": discard}