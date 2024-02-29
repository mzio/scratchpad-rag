"""
In-context TriviaQA dataset
"""
from datasets import load_dataset

from dataloaders.utils import get_tokenizer_from_config, convert_to_hf_dataset
from dataloaders.utils import train_test_split, subsample_split

from .contrastive import ContrastiveContextDataset
from .utils import *
import random
import gzip
import json

def load_data(data_config: dict, loader_config: dict):
    """
    Load dataloaders for Trivia-QA
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

    #hack to subtract max new token from maximum model length
    max_length=dataset_config["context_window"]-dataset_config["max_new_tokens"]

    # Get initial data
    dataset_kwargs = ['path', 'name', 'cache_dir']
    # dataset = load_dataset(
    #     **{k: v for k, v in dataset_config.items() if k in dataset_kwargs})
    file_path='/scr-ssd/aunell/scratchpad-rag/data/qa_data/biencoder-trivia-dev.json.gz'
    with gzip.open(file_path, 'rt', encoding='utf-8') as file:
        dataset = json.load(file)

    total_samples = len(dataset)
    train_set = dataset[:int(.8*total_samples)]
    val_set = dataset[int(.8*total_samples):]
    seed = dataset_config['seed']
    if dataset_config['num_train_samples'] is not None:
        train_set = subsample_split(train_set, dataset_config['num_train_samples'], seed)
    if dataset_config['num_val_samples'] is not None:
        val_set = subsample_split(val_set, dataset_config['num_val_samples'], seed)
    train_set = convert_to_hf_dataset(train_set, cache_dir)
    val_set = convert_to_hf_dataset(val_set, cache_dir)

    # Convert to question, answer, context, support format
    train_set = train_set.map(process_sample, remove_columns=list(train_set.features),
                              load_from_cache_file=False)
    train_set = train_set.filter((lambda x: len(x["support"])==1))
    val_set   = val_set.map(process_sample, remove_columns=list(val_set.features),
                            load_from_cache_file=False)
    val_set = val_set.filter((lambda x: len(x["support"])==1))

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
        'truncation': True,
        'max_length': max_length,
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
    # tokenize_kwargs['context_source'] = 'val_closed_book'
    # val_set_closed = tokenize_dataset(val_set, 'val_closed_book', **tokenize_kwargs)
    
    # Get dataloaders
    datasets_lm = {
        'train_lm_anc': train_set_lm_anc, 'val_lm_anc': val_set_lm_anc,
        'train_lm_pos': train_set_lm_pos, 'val_lm_pos': val_set_lm_pos,
    }
    datasets_seq2seq = {
        'val_anc': val_set_anc, 'val_pos': val_set_pos,
        # 'val_closed_book': val_set_closed,
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

    neg_context = sample['hard_negative_ctxs']+sample['negative_ctxs']
    pos_context = [sample['positive_ctxs'][0]] if sample['positive_ctxs'] else []

    try:
        pos_context = [sample['positive_ctxs'][0]]
    except:
        pos_context = []

    for doc in neg_context: doc['hasanswer']=False
    for doc in pos_context: doc['hasanswer']=True
    neg_context=neg_context[:19] #take only 20 docs total

    mixed_context=neg_context+pos_context
    random.shuffle(mixed_context)

    # Add contexts
    for ix, ctx in enumerate(mixed_context):
        if ctx['hasanswer']:
            support.append(
                {'title': ctx['title'], 'text': ctx['text']}
            )
            support_indices.append(ix)
        context.append(
            {'title': ctx['title'], 'text': ctx['text']}
        )
    answer = None
    
    for answer_alpha in sample['answers']:
        if answer_alpha.isalpha():
            answer = answer_alpha
            break
    if answer==None:
        answer=sample['answers'][0]
    sample = {
        'question': sample['question'],
        'answer': answer,
        'all_answers': sample['answers'],
        'context': context,
        'support': support,
        'support_indices': support_indices,
    }
    return sample