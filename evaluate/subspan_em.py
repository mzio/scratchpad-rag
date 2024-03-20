from typing import List

import string
import regex

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
import re

# from logging_utils import print_header


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


def best_subspan_em(prediction: str, ground_truths: List[str], data_type: str, eval_type: str) -> float:
    if data_type == 'nq':
        return best_subspan_em_nq(prediction, ground_truths, eval_type)
    if data_type == 'hqa':
        return best_subspan_em_hqa(prediction, ground_truths, eval_type)
    if data_type == 'tqa':
        return best_subspan_em_tqa(prediction, ground_truths, eval_type)
    
def em_metric(prediction: str, ground_truths: List[str]):
    '''
    prediction is model output, sliced to be LAO
    ground_truths is list of ground truth answers, typically is length 1 but more for TQA
    '''
    normalized_prediction = normalize_answer(prediction)
    for ground_truth in ground_truths:
        normalized_ground_truth = normalize_answer(ground_truth)
        #need only one answer to be in list of total correct answer options
        if normalized_ground_truth.lower() in normalized_prediction.lower() or normalized_prediction.lower() in normalized_ground_truth.lower():
            return 1.0
    return 0.0

def document_metric(prediction: str, ground_truths: List[str]):
    '''
    prediction is first part of model output, sliced to just be documents
    ground_truths is list of ground truth documents, typically is length 1 but more for HQA
    '''
    normalized_prediction = normalize_answer(prediction)
    for i in range(len(ground_truths)):
        ground_truths[i]=normalize_answer(ground_truths[i])
    for ground_truth in ground_truths:
        normalized_ground_truth = normalize_answer(ground_truth)
        #need all documents to be represented in prediction
        if not normalized_ground_truth.lower() in normalized_prediction.lower() and not normalized_prediction.lower() in normalized_ground_truth.lower():
            return 0.0
    return 1.0

def multi_metric(prediction_split: List[str], ground_truths: List[str]):
    '''
    prediction_split is model output, split into doc(s), question, answer
    ground_truths is list of ground truth answers, typically length 2--> doc(s), answer(s)
    '''
    em = em_metric(prediction_split[-1], [ground_truths[-1]])
    dr = document_metric(prediction_split[0], ground_truths[:-1])
    for i in range(len(ground_truths)):
        ground_truths[i]=normalize_answer(ground_truths[i])
    return 1 if em and dr else 0


def best_subspan_em_nq(prediction: str, ground_truths: List[str], eval_type: str) -> float:
    """ Compute best subspan exact match for NQ """
    prediction_split= prediction.split('\n') #split into doc, question, answer
    if eval_type=='em':
        return em_metric(prediction_split[-1], ground_truths)
    elif eval_type =='doc_retrieval':
        return document_metric(prediction_split[0], ground_truths)
    elif eval_type=='multi':
        prediction_split= prediction.split('\n') #split into doc, question, answer
        ground_truths=ground_truths[0].split('\n\n') #split into doc, answer
        return multi_metric(prediction_split, ground_truths)
    else:
        raise Exception("Evaluation metric not defined, received eval_type: ", eval_type, "Options are: em, doc_retrieval, multi")     

def best_subspan_em_hqa(prediction: str, ground_truths: List[str], eval_type: str) -> float:
    """ Compute best subspan exact match for HQA """
    prediction_split= prediction.split('\n\n') #split into doc, question, answer
    if eval_type=='em':
        return em_metric(prediction_split[-1], ground_truths)
    elif eval_type=='doc_retrieval':
        ground_truths = ground_truths[0].split('\n\n') 
        ground_truths=ground_truths[:-1]
        #pass in string of all documents associated w HQA
        return document_metric("".join(prediction_split[:-2]), ground_truths)
    elif eval_type=='multi':
        ground_truths=ground_truths[0].split('\n\n') #split into doc, answer
        hqa_multi_predictions=[]
        hqa_multi_predictions.append("".join(prediction_split[:-2]))  #add both docs in one string to predictions
        hqa_multi_predictions.append(prediction_split[-1])  #add answer to predictions
        return multi_metric(hqa_multi_predictions, ground_truths)
    else:
        raise Exception("Evaluation metric not defined, received eval_type: ", eval_type, "Options are: em, doc_retrieval, multi")

def best_subspan_em_tqa(prediction: str, ground_truths: List[str], eval_type: str) -> float:
    """ Compute best subspan exact match for TQA """
    prediction_split= prediction.split('\n\n') #split into doc, question, answer
    if eval_type=='em':
        ground_truths = ground_truths[0].split(',')
        return em_metric(prediction_split[-1], ground_truths)
    elif eval_type=='doc_retrieval':
        return document_metric(prediction_split[0], ground_truths)
    elif eval_type=='multi':
        ground_truths=ground_truths[0].split('\n\n') #split into doc, answer where answer is string of all possibilities
        return multi_metric(prediction_split, ground_truths)
    else:
        raise Exception("Evaluation metric not defined, received eval_type: ", eval_type)
    
def extract_titles(prompt, tokenizer):
    """Extract titles from document"""
    print('extracting titles')
    prompt = tokenizer.decode(prompt[0], skip_special_tokens=True)
    # Regex pattern to match "Document (Title: ____)"
    pattern = r"Document \(Title: ([^\)]+)\)"

    # Find all matches in the original string
    titles = re.findall(pattern, prompt)
    return [f"Document (Title: {title})" for title in titles]

def extract_documents(prompt, tokenizer):
    # Regular expression pattern to match documents
    prompt = tokenizer.decode(prompt[0], skip_special_tokens=True)
    pattern = r"Document \(Title: .+?\) .+?(?=\n\nDocument \(Title: |$)"

    # Find all matches of the pattern in the original string
    documents = re.findall(pattern, prompt, flags=re.DOTALL)
    documents[-1]=documents[-1].split('Question')[0]
    return documents

def extract_question(prompt, tokenizer):
    """Extract question from prompt"""
    prompt = tokenizer.decode(prompt[0], skip_special_tokens=True)
    # Regex pattern to match "Question: ____"
    pattern = r"Question: ([^\n]+)"

    # Find all matches in the original string
    question = re.findall(pattern, prompt)
    return question[0]

def calculate_perplexity(model, prompt, document_titles, tokenizer):
    """Calculate normalized perplexity of model on prompt for each document title."""
    print('calculating perplexity')
    prompt_decoded = tokenizer.batch_decode(prompt, skip_special_tokens=True)[0]
    doc_scores = {}
    start_length = tokenizer.encode(prompt_decoded + " Answer: ", return_tensors='pt').to(model.device).size(1)+3
    for title in tqdm(document_titles):
        # Concatenate the prompt with the current title
        new_answer = prompt_decoded + " Answer: " + title
        # Encode the concatenated string
        encoded_input = tokenizer.encode(new_answer, return_tensors='pt').to(model.device)
        # Calculate the number of tokens, subtracting 1 to not count the EOS token
        curr=np.mean(perplexity_token(model, encoded_input, start_length))
        doc_scores[title] = curr
    return sorted(doc_scores, key=doc_scores.get)[:3]

def perplexity_token(model, input_ids, start_length):
    nlls = []
    seq_len = input_ids.size(1)
    for end_loc in range(start_length, seq_len):
        input_ids_new = input_ids[:, :end_loc]
        # target_ids = input_ids_new.clone()
        # target_ids[:, :-1] = -100
        labels = torch.full_like(input_ids_new, -100)
        labels[:, -1] = input_ids_new[:, -1]

        with torch.no_grad():
            outputs = model(input_ids_new, labels=labels)
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)


    ppl = torch.exp(torch.stack(nlls).mean())
    return [ppl.item()]

def calculate_embeddings(model, documents, tokenizer, gold_index):
    """Calculate embeddings of model on prompt and every document in context"""
    # question_str = extract_question(prompt, tokenizer)
    doc=None
    gold_doc = documents[gold_index]
    documents.remove(gold_doc)
    encoded_document = tokenizer.encode(gold_doc, return_tensors='pt').to(model.device)
    with torch.no_grad():
        gold_doc_out = model(input_ids=encoded_document) #, labels=encoded_document)
        # print('gold_doc_out', len(gold_doc_out.hidden_states)) #33 items
        last_hidden_state_gold = gold_doc_out.hidden_states[-1]
        # print('last_hidden_state_gold', last_hidden_state_gold.size()) #size is [1, 157, 4096]
        gold_prediction = last_hidden_state_gold[:,-1,:] #.mean(dim=1) #[:,-1,:]  # Get the prediction for the last token of doc
        # print('gold_prediction', gold_prediction.size()) # size is [1, 4096] w taking mean
    distance = []
    for doc in documents:
        # prompt = prompt.to(model.device)
        encoded_document = tokenizer.encode(doc, return_tensors='pt').to(model.device)
        # encoded_question = tokenizer.encode(question_str, return_tensors='pt').to(model.device)
        with torch.no_grad():
            outputs_doc = model(input_ids=encoded_document, labels=encoded_document)
            last_hidden_state_doc = outputs_doc.hidden_states[-1]
            doc_prediction = last_hidden_state_doc.mean(dim=1)
            cosine_sim = F.cosine_similarity(doc_prediction, gold_prediction, dim=1)
        distance.append(cosine_sim.item())
    return distance

def create_new_prompt(top_k_docs, input_ids, tokenizer, model):
    question = extract_question(input_ids, tokenizer)
    top_k_docs = '\n\n'.join(top_k_docs)
    template = f"""Write a high-quality answer for the given question using only the provided context (some of which might be irrelevant).

Question: {{question}}

Context:
{{context}}
    
Question: {{question}}

Answer:"""
    prompt = template.format(context=top_k_docs, question=question.capitalize())
    prompt = f'{tokenizer.bos_token}{prompt}'
    new_prompt = tokenizer.encode(prompt, add_special_tokens=False, return_tensors='pt').to(model.device)
    attn_mask = [1] * len(new_prompt)
    # Negative sample -> just mask out supporting context
    negative_ids = new_prompt
    negative_attn_mask = [1] * len(new_prompt)
    # for indices in support_token_indices:
    #     neg_len = indices[1] - indices[0]
    #     negative_ids[indices[0]:indices[1]] = [tokenizer.pad_token_id] * neg_len
    #     negative_attn_mask[indices[0]:indices[1]] = [0] * neg_len
    
    sample =  {
        "input_ids": new_prompt,
        "attention_mask" : torch.tensor([1] * new_prompt.size(1)).unsqueeze(0),
        "negative_ids": negative_ids,
        "negative_attention_mask": negative_attn_mask,
        "labels": torch.tensor([-100] * new_prompt.size(1)).unsqueeze(0),
        "support_indices": torch.tensor([0]).unsqueeze(0),
        # "support_token_indices": support_token_indices,
        # "support_token_start": support_token_start,
        # "support_token_end": support_token_end,
    }
    return sample
def evaluate_mqa(model, eval_loader, tokenizer, 
                 negative_sample: bool = False,
                 max_new_tokens: int = 20, 
                 max_samples: int = None,
                 print_outputs: bool = False,
                 last_answer_only: bool = False,
                 eval_start: int = None,
                 eval_end: int = None, 
                 data_type: str = 'nq',
                 eval_type: str = 'em'):
    """Evaluate with subspan exact-match"""
    eval_metrics = {
        'length': [],
        'support_doc_index': [],
        'support_token_idx': [],
        'subspan_em': [],
        'embeddings': []
    }
    model_input_args = ['input_ids', 'attention_mask', 'labels']

    if eval_start is None:
        eval_start = 0
    if eval_end is None or eval_end > len(eval_loader):
        eval_end = len(eval_loader)
    eval_indices = range(eval_start, eval_end)
        
    if max_samples is None: max_samples = len(eval_loader)
    pbar = tqdm(range(eval_start, eval_end), desc='Evaluating', colour='blue')
    return_top_3 = False
    print(f'-> Returning top 3 documents:', return_top_3)

    model.eval()
    with torch.no_grad():
        for ix, data in enumerate(eval_loader):
            if ix in eval_indices:
            
                if negative_sample:
                    data['input_ids'] = data['negative_ids']
                    data['attention_mask'] = data['negative_attention_mask']
                _input_ids = data['input_ids']
                if return_top_3:
                    doc_strings= extract_documents(_input_ids, tokenizer)
                    top_k_docs = calculate_perplexity(model, _input_ids, doc_strings, tokenizer)
                    labels=data['labels']
                    data = create_new_prompt(top_k_docs, _input_ids, tokenizer, model)
                    data['labels']=labels
                ## if we are able to return top 3 docs from calculate_perplexity, we can use those as new prompt
                outputs = model.generate(
                    **{k: v.to(model.device) for k, v in data.items()
                       if k in model_input_args},
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.eos_token_id,
                )
                _input_ids = data['input_ids']
                for sample_idx in range(len(_input_ids)):
                    outputs = outputs[sample_idx:sample_idx+1, len(_input_ids[sample_idx]):]
                outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                try:
                    targets = tokenizer.batch_decode(data['labels'], skip_special_tokens=True)
                    print('target is', targets)
                except:
                    breakpoint()
                prompt = data['input_ids']
                documents = extract_documents(prompt, tokenizer)
                gold_index = data['support_indices']
                for sample_idx in range(len(_input_ids)):
                    _outputs = outputs[sample_idx]
                    em = best_subspan_em(_outputs, [targets[sample_idx]], data_type, eval_type)
                    if print_outputs:
                        print_header('Prompt:')
                        print(tokenizer.decode(_input_ids[sample_idx], skip_special_tokens=True))
                        print_header('Model output:')
                        
                        print(outputs[sample_idx])
                        print_header('Ground-truth:')
                        print(targets[sample_idx])
                        print_header(f'Sample subspan exact match: {em:.3f}')
                    embed_distance = calculate_embeddings(model, documents, tokenizer, gold_index)
                    eval_metrics['embeddings'].append(embed_distance)
                    for _idx, support_idx in enumerate(data['support_indices'][sample_idx]):
                        eval_metrics['subspan_em'].append(em)
                        if not return_top_3:
                            token_idx = data['support_token_end'][sample_idx][_idx]
                            eval_metrics['length'].append(_input_ids.shape[-1])
                            eval_metrics['support_doc_index'].append(support_idx.item())
                            eval_metrics['support_token_idx'].append(token_idx.item())

                correct = int(sum(eval_metrics['subspan_em']))
                total = len(eval_metrics['subspan_em'])
                _em = correct / total * 100
                pbar.set_description(f'Evaluating from {eval_start} to {eval_end} | subspan_exact_match: {_em} ({correct}/{total})')
                pbar.update(1)
                if ix == max_samples:
                    break
    pbar.close()
    print('eval metrics', eval_metrics)
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
                     label: str = None, show_plot: bool = True, output='plot.png'):
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
    # plt.savefig(output)
    if show_plot:
        plt.grid()
        plt.title(title)
        plt.xlabel('Support doc position')
        plt.ylabel('Subspan exact match')
        if label is not None:
            plt.legend()
        plt.show()

    return {'document_position': doc_indices, 'subspan_em': subspan_ems}

def plot_embeddings(eval_metrics: dict, show_plot: bool = True):
    # Simulated data: Replace these with your actual lists of lists and labels
    # Example lists and their labels
    distances =eval_metrics['embeddings'] 
    scores = eval_metrics['subspan_em']

    # Extracting features
    averages = [np.mean(lst) if lst else 0 for lst in distances]
    max_similarity = [np.max(lst) if lst else 0 for lst in distances] # or use max value for outlier emphasis
    labels = [label for label in scores]
    avg_var_0 = [(avg, var) for avg, var, label in zip(averages, max_similarity, labels) if label == 0]
    avg_var_1 = [(avg, var) for avg, var, label in zip(averages, max_similarity, labels) if label == 1]

    avg_0 = np.mean([avg for avg, var in avg_var_0])

    avg_1 = np.mean([avg for avg, var in avg_var_1])
    # Plotting
    plt.figure(figsize=(10, 6))

    # Scatter plot for label 0
    plt.scatter([avg for avg, label in zip(averages, labels) if label == 0],
                [var for var, label in zip(max_similarity, labels) if label == 0],
                color='red', label='Label 0', alpha=0.7, marker='^')

    # Scatter plot for label 1
    plt.scatter([avg for avg, label in zip(averages, labels) if label == 1],
                [var for var, label in zip(max_similarity, labels) if label == 1],
                color='blue', label='Label 1', alpha=0.7, marker='o')

    plt.text(0.02, 0.80, f'Label 0: avg={avg_0:.2f}', transform=plt.gca().transAxes)
    plt.text(0.02, 0.75, f'Label 1: avg={avg_1:.2f}', transform=plt.gca().transAxes)

    plt.title('Embedding Characteristics vs. Accuracy')
    plt.xlabel('Average Value of Embedding Similarity')
    plt.ylabel('Maximum Cosine Similarity of Embeddings')
    plt.legend()
    plt.grid(True)
    plt.savefig('embedding_plot_nq_Final.png')
    plt.show()
