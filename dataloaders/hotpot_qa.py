"""
In-context Hotpot QA dataset
"""
from datasets import load_dataset

from dataloaders.utils import get_tokenizer_from_config, convert_to_hf_dataset
from dataloaders.utils import train_test_split, subsample_split

from .contrastive import ContrastiveContextDataset
from .utils import *


def load_data(data_config: dict, loader_config: dict):
    """
    Load dataloaders for Hotpot-QA
    """
    name = data_config['name']
    dataset_config = data_config['dataset_config']
    pretrained_model_config = data_config['pretrained_model_config']
    preprocess_config = data_config['preprocess_config']
    tokenizer_name = pretrained_model_config['pretrained_model_name_or_path']
    tokenizer_name = tokenizer_name.split('/')[-1]

    if dataset_config['include_support']:
        name += f'-is=1'

    # Misc. setup
    cache_dir = dataset_config['cache_dir']
    
    # Setup tokenizer
    if 'tokenizer' not in pretrained_model_config:
        tokenizer = get_tokenizer_from_config(pretrained_model_config)
    else:
        tokenizer = pretrained_model_config['tokenizer']  # hack
        
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f'Setting tokenizer.pad_token to {tokenizer.pad_token}')

    tokenizer.padding_side = 'left'  # for decoder-only generation

    # Get initial data
    dataset_kwargs = ['path', 'name', 'cache_dir']
    dataset = load_dataset(
        **{k: v for k, v in dataset_config.items() if k in dataset_kwargs})

    train_set = dataset['train']
    val_set = dataset['validation']

    # Convert to question, answer, context, support format
    train_set = train_set.map(process_sample, remove_columns=list(train_set.features),
                              load_from_cache_file=False)
    val_set   = val_set.map(process_sample, remove_columns=list(val_set.features),
                            load_from_cache_file=False)

    seed = dataset_config['seed']

    # Tokenize and prepare different datasets from source data
    tokenize_kwargs = {
        'tokenizer': tokenizer,
        'tokenizer_name': f'{name}_{tokenizer_name}',
        'tokenize_func': tokenize_add_label,
        'context_source': 'support',
        'include_label': True,
        'cache_dir': cache_dir,
        'instruct_tune': 'instruct' in tokenizer_name.lower(),
        'include_support': dataset_config['include_support'],
    }

    # 1. SFT datasets on entire context (baseline)
    tokenize_kwargs['context_source'] = 'context'
    train_set_lm_anc = tokenize_dataset(train_set, 'train_lm_anc',  **tokenize_kwargs)
    val_set_lm_anc   = tokenize_dataset(val_set, 'val_lm_anc', **tokenize_kwargs)

    # 2. Positive datasets (upper baseline, or for contrastive)
    tokenize_kwargs['context_source'] = 'support'
    train_set_lm_pos = tokenize_dataset(train_set, 'train_lm_pos',  **tokenize_kwargs)
    val_set_lm_pos   = tokenize_dataset(val_set, 'val_lm_pos', **tokenize_kwargs)

    # 3. Evaluation dataloaders
    tokenize_kwargs['include_label'] = False
    tokenize_kwargs['context_source'] = 'context'
    val_set_anc = tokenize_dataset(val_set, 'val_anc', **tokenize_kwargs)
    tokenize_kwargs['context_source'] = 'support'
    val_set_pos = tokenize_dataset(val_set, 'val_pos', **tokenize_kwargs)
    
    # Get dataloaders
    datasets_lm = {
        'train_lm_anc': train_set_lm_anc, 'val_lm_anc': val_set_lm_anc,
        'train_lm_pos': train_set_lm_pos, 'val_lm_pos': val_set_lm_pos,
    }
    datasets_seq2seq = {
        'val_anc': val_set_anc, 'val_pos': val_set_pos,
    }
    dataloaders = {
        k: get_lm_loader(v, tokenizer, k, **loader_config)
        for k, v in datasets_lm.items()
    }
    for k, v in datasets_seq2seq.items():
        dataloaders[k] = get_seq2seq_loader(v, tokenizer, k, **loader_config)

    # Finishing touches
    for k, v in dataloaders.items():  # Make tokenizer accessible
        if v is not None:
            dataloaders[k].dataset.tokenizer = tokenizer

    return dataloaders


def process_sample(sample: dict):
    """
    Preprocess data into question, answer, full context, support
    -> Include full paragraphs for supporting contexts
    """
    support = []
    context = []
    support_indices = []  # Which result they showed up in

    context_titles = sample['context']['title']
    support_titles = sample['supporting_facts']['title']

    context_sentences = sample['context']['sentences']
    support_sent_ids  = sample['supporting_facts']['sent_id']

    # Add contexts
    for cix, sentences in enumerate(context_sentences):
        _context = {'title': context_titles[cix], 'text': ''.join(sentences)}
        context.append(_context)
        #  Add supporting facts
        if context_titles[cix] in support_titles:
            support.append(_context)
            support_indices.append(cix)
            
    sample = {
        'question': sample['question'],
        'answer': sample['answer'],
        'context': context,
        'support': support,
        'support_indices': support_indices,
    }
    return sample