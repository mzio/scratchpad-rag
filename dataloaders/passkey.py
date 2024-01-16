"""
Implement passkey retrieval
"""
"""
Passkey Retrieval dataset. 

Originally from "Landmark Attention: Random-Access Infinite Context Length for Transformers" 
by Amirkeivan Mohtashami, Martin Jaggi. https://arxiv.org/abs/2305.16300
"""

from functools import partial

import copy
import numpy as np

import torch
from torch.utils.data import Dataset

from tqdm import tqdm
from datasets import load_metric

from .utils import *
# from .base.train import get_lm_loader
# from .base.eval import get_seq2seq_loader, preprocess_seq2seq
# from .utils import convert_to_hf_dataset, train_test_split
# from .utils import get_tokenizer_from_config, add_special_tokens_to_dataset
# from .utils import download_scrolls_metric as download_metric


def load_data(data_config: dict, loader_config: dict):
    """
    Load pretraining, finetuning, and evaluation data for passkey retreival
    """
    name = data_config['name']
    dataset_config = data_config['dataset_config']
    pretrained_model_config = data_config['pretrained_model_config']
    preprocess_config = data_config['preprocess_config']

    # Misc. setup
    cache_dir = {'cache_dir': dataset_config['cache_dir']}
    input_len = dataset_config['chunk_size']
    
    # Setup tokenizer
    if 'tokenizer' not in pretrained_model_config:
        tokenizer = get_tokenizer_from_config(pretrained_model_config)
    else:
        tokenizer = pretrained_model_config['tokenizer']  # hack
        
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f'Setting tokenizer.pad_token to {tokenizer.pad_token}')

    tokenizer.padding_side = 'left'  # for decoder-only generation
    # ^ But does this impact impact attention sink stuff?

    _dataset = PassKeyRetrieval(**dataset_config)
    # Convert to HF dataset for consistency with tokenizing
    train_set = convert_to_hf_dataset(_dataset._train_set(), **cache_dir)
    val_set   = convert_to_hf_dataset(_dataset._val_set(), **cache_dir)
    test_set  = convert_to_hf_dataset(_dataset._test_set(), **cache_dir)

    # Add prompt formatting
    train_set = train_set.map(partial(apply_prompt_template, tokenizer=tokenizer), 
                              remove_columns=list(train_set.features),
                              load_from_cache_file=False)
    val_set   = val_set.map(partial(apply_prompt_template, tokenizer=tokenizer),
                            remove_columns=list(val_set.features),
                            load_from_cache_file=False)
    test_set  = test_set.map(partial(apply_prompt_template, tokenizer=tokenizer),
                             remove_columns=list(test_set.features),
                             load_from_cache_file=False)

    # # Make special tokens accessible later on
    # train_set = add_special_tokens_to_dataset(train_set, tokenizer)
    # val_set   = add_special_tokens_to_dataset(val_set, tokenizer)
    # test_set  = add_special_tokens_to_dataset(test_set, tokenizer)
    # # Actually probably just referencing the tokenizer is better
    # train_set.tokenizer = tokenizer
    # val_set.tokenizer   = tokenizer
    # test_set.tokenizer  = tokenizer
    
    # Get dataloaders    
    ft_train_loader = get_lm_loader(train_set, tokenizer, 'train',
                                    **loader_config)
    ft_val_loader   = get_lm_loader(val_set, tokenizer, 'val',
                                    **loader_config)

    val_loader = get_seq2seq_loader(val_set, tokenizer, 'val',
                                   **loader_config)
    
    test_loader = get_seq2seq_loader(test_set, tokenizer, 'test',
                                    **loader_config)
    dataloaders = {
        'train': ft_train_loader,
        'val_xent': ft_val_loader,
        'val': val_loader,
        'test': test_loader
    }
    
    # Finishing touches
    for k, v in dataloaders.items():  # Make tokenizer accessible
        dataloaders[k].dataset.tokenizer = tokenizer
    return dataloaders


def visualize_data():
    raise NotImplementedError


def apply_prompt_template(sample, tokenizer):
    """
    Format dataset context and answers into single-sequence prompts
    """
    prompt = f"{{bos_token}}{{input}}"
    prompt = prompt.format(bos_token=tokenizer.bos_token, input=sample['input'])
    prompt = tokenizer.encode(prompt, add_special_tokens=False)
    labels = tokenizer.encode(f'{sample["output"]}', add_special_tokens=False)
    # {tokenizer.eos_token}

    # Add supporting context positions -> position where they end in context
    support_token_indices = []
    support_token_start, support_token_end = [], []
    # Get positions of 
    try:
        start, end = get_target_index(prompt, labels)
    except:
        start, end = 0, 0
    support_token_indices.append((start, end))
    support_token_start.append(start)
    support_token_end.append(end)

    attn_mask = [1] * len(prompt)
    return {
        "input_ids": prompt,
        "attention_mask" : attn_mask,
        "labels": labels,
        "support_token_indices": support_token_indices,
        "support_token_start": support_token_start,
        "support_token_end": support_token_end,
    }
    # return {
    #     "text": prompt.format(
    #         input=sample["input"],
    #         output=sample["output"],
    #         eos_token=tokenizer.eos_token,
    #     )
    # }




class PassKeyRetrievalDataset(Dataset):
    def __init__(self, inputs, outputs, locations):
        self.inputs    = inputs
        self.outputs   = outputs
        self.locations = locations

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx: int):
        return {'input': self.inputs[idx],
                'output': self.outputs[idx],
                'location': self.locations[idx]}


class PassKeyRetrieval():
    """
    Passkey retrieval task from 
    Landmark Attention: Random-Access Infinite Context Length for Transformers
    - Amirkeivan Mohtashami, Martin Jaggi, https://arxiv.org/abs/2305.16300
    """
    def __init__(self, 
                 n_garbage: int = 10000, 
                 passkey_len: int = 5,
                 passkey_location: int = None,
                 train_samples: int = 10000,
                 val_samples: int = 2000,
                 test_samples: int = 2000,
                 split_passkeys: bool = True,
                 seed: int = 42,
                 **kwargs: any):
        self.n_garbage = n_garbage
        self.passkey_len = passkey_len
        self.passkey_location = passkey_location

        self.split_passkeys = split_passkeys
        self.seed = seed

        # Setup data
        if passkey_location is None:  # where passkey is located
            self.location_range = [0, n_garbage]
        else:
            idx = n_garbage // 3
            thirds = [[0, idx], [idx, 2 * idx], [2 * idx, n_garbage]]
            self.location_range = thirds[passkey_location]

        self.passkey_range = [0, 10 ** self.passkey_len]  # values of passkey
        if self.split_passkeys:  # Unique passkeys across train and test
            self.train_val_range = [0, self.passkey_range[1] // 2]
            self.test_range = [self.passkey_range[1] // 2, self.passkey_range[1]]

            train_val_data = self.init_data(n_samples=train_samples + val_samples,
                                            passkey_range=self.train_val_range)
            test_data = self.init_data(n_samples=test_samples,
                                       passkey_range=self.test_range)
            
            train_idx, val_idx = train_test_split(np.arange(train_samples + val_samples),
                                                  train_samples, val_samples,
                                                  self.seed)
            train_data = {k: [v[_ix] for _ix in train_idx] for k, v in train_val_data.items()}
            val_data   = {k: [v[_ix] for _ix in val_idx] for k, v in train_val_data.items()}
        
        else:
            self.train_val_range = self.passkey_range
            self.test_range = self.passkey_range
            
            train_val_test_data = self.init_data(
                n_samples=train_samples + val_samples + test_samples,
                passkey_range=self.passkey_range)

            train_val_idx, test_idx = train_test_split(
                np.arange(train_samples + val_samples + test_samples),
                train_samples + val_samples, test_samples,
                self.seed
            )
            # train_val_data, test_data = train_test_split(train_val_test_data,
            #                                              train_samples + val_samples,
            #                                              test_samples, 
            #                                              self.seed)
            train_idx, val_idx = train_test_split(train_val_idx,
                                                  train_samples,
                                                  val_samples,
                                                  self.seed)
            train_data = {k: [v[_ix] for _ix in train_idx] for k, v in train_val_test_data.items()}
            val_data   = {k: [v[_ix] for _ix in val_idx] for k, v in train_val_test_data.items()}
            test_data  = {k: [v[_ix] for _ix in test_idx] for k, v in train_val_test_data.items()}

        self.data = {'train': train_data,
                     'val': val_data,
                     'test': test_data}

    def _train_set(self):
        return PassKeyRetrievalDataset(**self.data['train'])

    def _val_set(self):
        return PassKeyRetrievalDataset(**self.data['val'])

    def _test_set(self):
        return PassKeyRetrievalDataset(**self.data['test'])
                                                    
    # def __len__(self):
    #     return len(self.data['input'])

    # def __getitem__(self, idx):
    #     return {'input': self.data['input'][idx],
    #             'output': self.data['output'][idx],
    #             'location': self.data['location'][idx]}

    def init_data(self, n_samples, passkey_range):
        data = {'inputs': [], 'outputs': [], 'locations': []}
        np.random.seed(self.seed)
        
        # Get passkeys
        data['outputs'] = np.random.choice(np.arange(*passkey_range),
                                          size=n_samples, replace=True)
        data['outputs'] = [f'{p:0>{self.passkey_len}d}' for p in data['outputs']]

        # Get contexts
        passkey_locations = np.random.choice(np.arange(*self.location_range), 
                                             size=n_samples)
        data['locations'] = passkey_locations
        
        n_garbage_prefix = passkey_locations
        n_garbage_suffix = self.n_garbage - n_garbage_prefix
        # print(n_garbage_prefix, n_garbage_suffix)

        task_desc = "There is an important piece of info hidden inside a lot of irrelevant text. Find it and memorize it. I will quiz you about the important information there."
        garbage = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
        garbage_inf = " ".join([garbage] * self.n_garbage * 2)
        assert len(garbage_inf) >= self.n_garbage
        final_question = "What is the pass key? The pass key is"

        for ix, passkey in tqdm(enumerate(data['outputs']), desc='Generating data', 
                                leave=False, colour='yellow'):
            garbage_prefix = garbage_inf[:n_garbage_prefix[ix]]
            garbage_suffix = garbage_inf[:n_garbage_suffix[ix]]
        
            information_line = f"The pass key is {passkey}. Remember it. {passkey} is the pass key."
        
            data['inputs'].append("\n".join([
                task_desc, garbage_prefix, information_line, garbage_suffix, final_question
            ]))
        return data