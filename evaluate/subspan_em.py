from typing import List

import string
import regex

import torch
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from logging_utils import print_header


# Copied from https://github.com/nelson-liu/lost-in-the-middle/blob/main/src/lost_in_the_middle/metrics.py
def normalize_answer(s: str) -> str:
    """
    Normalization from the SQuAD evaluation script.

    See https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/
    """

    def remove_articles(text):
        return regex.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    def remove_stop_token(text):  # because some tokenizers don't behave...
        return text.replace('</s>', '')

    return white_space_fix(remove_articles(remove_punc(remove_stop_token(lower(s)))))

# Copied from https://github.com/nelson-liu/lost-in-the-middle/blob/main/src/lost_in_the_middle/metrics.py
def best_subspan_em(prediction: str, ground_truths: List[str]) -> float:
    normalized_prediction = normalize_answer(prediction)

    for ground_truth in ground_truths:
        normalized_ground_truth = normalize_answer(ground_truth)
        if normalized_ground_truth.lower() in normalized_prediction.lower():
            return 1.0
    return 0.0


def evaluate_mqa(model, eval_loader, tokenizer, 
                 negative_sample: bool = False,
                 max_new_tokens: int = 20, 
                 max_samples: int = None,
                 print_outputs: bool = False,
                 last_answer_only: bool = False,
                 eval_start: int = None,
                 eval_end: int = None):
    """Evaluate with subspan exact-match"""
    eval_metrics = {
        'length': [],
        'support_doc_index': [],
        'support_token_idx': [],
        'subspan_em': [],
    }
    model_input_args = ['input_ids', 'attention_mask', 'labels']

    if eval_start is None:
        eval_start = 0
    if eval_end is None or eval_end > len(eval_loader):
        eval_end = len(eval_loader)
    eval_indices = range(eval_start, eval_end)
        
    if max_samples is None: max_samples = len(eval_loader)
    pbar = tqdm(range(eval_start, eval_end), desc='Evaluating', colour='blue')
    
    print(f'-> Only evaluating last part of answer:', last_answer_only)
    
    model.eval()
    # Handle the out of range IDs or skip this batch
    context_token_length = 850
    with torch.no_grad():
        for ix, data in enumerate(eval_loader):
            if ix in eval_indices:
                if negative_sample:
                    data['input_ids'] = data['negative_ids']
                    data['attention_mask'] = data['negative_attention_mask']
                if data['input_ids'].shape[1]>context_token_length:
                    continue
                _input_ids = data['input_ids']
                print('MNT',1024-data['input_ids'].shape[1])
                outputs = model.generate(
                    **{k: v.to(model.device) for k, v in data.items()
                       if k in model_input_args},
                    max_new_tokens=1024-data['input_ids'].shape[1],
                    pad_token_id=tokenizer.eos_token_id,
                )
                for sample_idx in range(len(_input_ids)):
                    outputs = outputs[sample_idx:sample_idx+1, len(_input_ids[sample_idx]):]
                outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                try:
                    targets = tokenizer.batch_decode(data['labels'], skip_special_tokens=True)
                except:
                    breakpoint()
                for sample_idx in range(len(_input_ids)):
                    if last_answer_only:
                        # _outputs = outputs[sample_idx].split('\n')[-1]  # outputs[sample_idx]
                        try:
                            if '</s>' in outputs[sample_idx]:
                                _outputs = outputs[sample_idx].split('</s>')[-2].split('\n')[-1]
                            else:  # hacks because Llama2-7b-32k tokenizer messed up
                                _outputs = outputs[sample_idx].split('\n')[-1]  # outputs[sample_idx]
                        except:
                            _outputs = outputs[sample_idx].split('\n')[-1]  # outputs[sample_idx]
                    else:
                        _outputs = outputs[sample_idx]
                    em = best_subspan_em(_outputs, [targets[sample_idx]])
                    if print_outputs:
                        print_header('Prompt:')
                        print(tokenizer.decode(_input_ids[sample_idx], skip_special_tokens=True))
                        print_header('Model output:')
                        
                        print(outputs[sample_idx])
                        print_header('Ground-truth:')
                        print(targets[sample_idx])
                        print_header(f'Sample subspan exact match: {em:.3f}')
                        
                    for _idx, support_idx in enumerate(data['support_indices'][sample_idx]):
                        token_idx = data['support_token_end'][sample_idx][_idx]
                        eval_metrics['length'].append(_input_ids.shape[-1])
                        eval_metrics['support_doc_index'].append(support_idx.item())
                        eval_metrics['support_token_idx'].append(token_idx.item())
                        eval_metrics['subspan_em'].append(em)

                correct = int(sum(eval_metrics['subspan_em']))
                total = len(eval_metrics['subspan_em'])
                _em = correct / total * 100
                pbar.set_description(f'Evaluating from {eval_start} to {eval_end} | subspan_exact_match: {_em} ({correct}/{total})')
                pbar.update(1)
                if ix == max_samples:
                    break
    pbar.close()
    return {k: np.array(v) for k, v in eval_metrics.items()}


def plot_histogram_em(eval_metrics: dict, title: str = None,
                      show_plot: bool = True):
    """
    Plot histogram of supporting token indices for 
    correct and incorrect samples
    """
    all_indices = np.arange(len(eval_metrics['subspan_em']))
    # Correct, subspan_em = 1
    correct_idx = np.where(eval_metrics['subspan_em'] > 0)[0]
    # Incorrect, subspan_em = 0
    mask = np.ones(len(eval_metrics['subspan_em']), dtype=bool)
    mask[correct_idx] = False
    incorrect_idx = all_indices[mask]
    
    token_pos = eval_metrics['support_token_idx']
    plt.hist(token_pos[correct_idx], label='Correct', alpha=0.7)
    plt.hist(token_pos[incorrect_idx], label='Incorrect', alpha=0.7)
    if show_plot:
        plt.grid()
        plt.xlabel('Supporting token index (end)')
        plt.ylabel('Subspan exact match (0 or 1)')
        plt.legend()
        plt.title(title)
        plt.show()


# Another view
def plot_lineplot_em(eval_metrics: dict, 
                     position_feature: str = 'support_doc_index',
                     title: str = None,
                     label: str = None, show_plot: bool = True):
    """
    Plot lineplot comparing supporting doc index vs 
    exact-match accuracy
    """
    # Another view
    doc_indices = []
    subspan_ems = []
    for doc_idx in np.unique(eval_metrics['support_doc_index']):
        _idx = np.where(eval_metrics['support_doc_index'] == doc_idx)[0]
        doc_indices.append(doc_idx)
        subspan_ems.append(np.mean(eval_metrics['subspan_em'][_idx]))

    plt.plot(doc_indices, subspan_ems, marker='o', label=label)
    if show_plot:
        plt.grid()
        plt.title(title)
        plt.xlabel('Support doc position')
        plt.ylabel('Subspan exact match')
        if label is not None:
            plt.legend()
        plt.show()

    return {'document_position': doc_indices, 'subspan_em': subspan_ems}