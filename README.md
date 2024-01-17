# Scratchpad-RAG
Improving robustness of models over long(er) contexts

## Getting started

### Setup dependencies
Please see `environment.yaml` for dependencies. We can set them up with conda:
```
conda env create -f environment.yaml
conda activate scratchpad-rag
```

### Config organization
For running the demo and also training models with Scratchpad-RAG, we provide experiment and model configs in `./configs`. 

For data, please see `./configs/experiment/` for sample dataset configs. We provide sample starting files mimicking Nelson Liu's [Lost in the Middle](https://arxiv.org/abs/2307.03172) multi-document question-answering setting are in `./data/qa_data`. These are copied from the 20 document setting in the [paper repo](/juice2/scr2/mzhang/projects/scratchpad-rag/data/qa_data). 

For models, please see `./configs/model/` for sample configs to load models. Our scripts should automatically download the models from HuggingFace, but you should change the `cache_dir` to reflect where you want to save the weights.

For example:
```yaml
pretrained_config:
  pretrained_model_name_or_path: 'mistralai/Mistral-7B-v0.1'
  cache_dir: '/juice/scr/scr110/scr/nlp/data/neo/hub/'  # change this
  return_dict: true
  quantization: false
  device_map: auto
  low_cpu_mem_usage: true
  torch_dtype: bfloat16
  rope_theta: 10000.0
  attn_implementation: flash_attention_2
```

## Training  
Afterwards, we can run the main training script by specifying the experiment and model config as commandline arguments, e.g.,

**Standard RAG + finetune with LoRA**
```
python main.py --experiment_config nq_lim_20_docs --model_config mistral_7b --peft_config lora_r8_a16_qv --train_method sft --num_train_samples 1000 --num_val_samples 1000 --last_answer_only --max_new_tokens 512 --print_outputs
```

**Scratchpad-RAG with LoRA**
```
python main.py --experiment_config nq_lim_20_docs --model_config mistral_7b --peft_config lora_r8_a16_qv --train_method scratchpad --num_train_samples 1000 --num_val_samples 1000 --last_answer_only --max_new_tokens 512 --print_outputs
```

## Evaluation
After that, (or alternatively with some preloaded checkpoints), we can evaluate our LLMs in RAG-like settings either via an evaluation script or real-time demo.

### Single setting eval

For a single eval setting (e.g., RAG with 20 docs retrieved), we can modify the above training scripts to do evaluation only (adding `--eval_only --max_new_tokens <max_tokens_to_generate>`). For example:  

**Closed-book / Zero-shot** (`--eval_split val_closed_book`)
```
python main.py --experiment_config nq_lim_20_docs --model_config mistral_7b --eval_split val_closed_book --train_method sft --print_outputs --num_train_samples 1000 --num_val_samples 1000 --eval_only --max_new_tokens 128
```

**RAG**
```
python main.py --experiment_config nq_lim_20_docs --model_config mistral_7b --eval_split val_anc --train_method sft --print_outputs --num_train_samples 1000 --num_val_samples 1000 --eval_only --max_new_tokens 128
```

**Distractor-free / Gold-chunk RAG** (`--eval_split val_pos`)
```
python main.py --experiment_config nq_lim_20_docs --model_config mistral_7b_instruct --eval_split val_pos --train_method sft --print_outputs --num_train_samples 1000 --num_val_samples 1000 --eval_only --max_new_tokens 128
```

**RAG + Finetune LLM** (`--max_new_tokens 128 --load_checkpoint --checkpoint_path <path_to_ckpt>`)
```
python main.py --experiment_config nq_lim_20_docs --model_config mistral_7b --peft_config lora_r8_a16_qv --eval_split val_anc --train_method scratchpad --print_outputs --num_train_samples 1000 --num_val_samples 1000 --last_answer_only --eval_only --max_new_tokens 128 --load_checkpoint --checkpoint_path './checkpoints/mistral_7b/rag_sft.pt'
```

**Scratchpad-RAG** (`--max_new_tokens 512 --load_checkpoint --checkpoint_path <path_to_ckpt>`)
```
python main.py --experiment_config nq_lim_20_docs --model_config mistral_7b --peft_config lora_r8_a16_qv --eval_split val_anc --train_method scratchpad --print_outputs --num_train_samples 1000 --num_val_samples 1000 --last_answer_only --eval_only --max_new_tokens 512 --load_checkpoint --checkpoint_path './checkpoints/mistral_7b/scratchpad_rag.pt'
```

### Full sweep over docs retrieved eval
For running a full sweep eval over number of docs retrieved (used to make plots in writeup), check out `evaluate_nq_lim.py`. Examples: 

**RAG**
```
python evaluate_nq_lim.py --model_config mistral_7b --eval_start 0 --eval_end 1000
```

**RAG + Finetune LLM**
```
python evaluate_nq_lim.py --model_config mistral_7b --peft_config lora_r8_a16_qv --eval_start 0 --eval_end 1000 --max_new_tokens 128 --last_answer_only --load_checkpoint --checkpoint_path './checkpoints/mistral_7b/rag_sft.pt'
```

**Scratchpad-RAG**
```
python evaluate_nq_lim.py --model_config mistral_7b --peft_config lora_r8_a16_qv --eval_start 0 --eval_end 1000 --max_new_tokens 512 --last_answer_only --load_checkpoint --checkpoint_path './checkpoints/mistral_7b/scratchpad_rag.pt'
```

### Demo    
Alternatively, we can run an interactive RAG-like demo on the commandline with sample questions, retrievals, and model outputs. Example commands:  

**RAG**
```
python demo.py --experiment_config nq_lim_20_docs --model_config mistral_7b  
```

**RAG + Finetune LLM**
```
python demo.py --experiment_config nq_lim_20_docs --model_config mistral_7b --checkpoint_path './checkpoints/mistral_7b/rag_sft.pt'
```

**Scratchpad-RAG**
```
python demo.py --experiment_config nq_lim_20_docs --model_config mistral_7b --checkpoint_path './checkpoints/mistral_7b/scratchpad_rag.pt'
```

Note currently `demo.py` is hardcoded for lost in the middle NaturalQuestions datasets. We can change that later.

More example commands with additional argparse args below. So far we've got configs for testing on Mistral-7B, Mistral-7B-Instruct, Llama2-7B-32K, and Llama2-7B-32K-Instruct models.


## Creating new Scratchpad-RAG training sets
Please see `./notebooks/dev-2.0-dataset_construct-hotpot_qa.ipynb` for a notebook on scratchpad RAG dataset creation. This amounts to simply creating text samples of the format:
```
<system prompt>

<question>

<all context>

<question>

<supporting context>

<question>

<answer>
```  
We then train models over these entire samples with a standard next-token prediction cross-entropy loss. 

Just for comparison, a standard RAG + finetuning LLM sample would look like:
```
<system prompt>

<question>

<all context>

<question>

<answer>
```


## More example commands

RAG + Finetuning Mistral-7B
```
python main.py --experiment_config nq_lim_20_docs --model_config mistral_7b --peft_config lora_r8_a16_qv --eval_split val_anc --train_method sft --print_outputs --num_train_epochs 5 --lr 1e-4 --weight_decay 0.01 --scheduler none --num_train_samples 1000 --num_val_samples 1000 --last_answer_only --max_new_tokens 128
```

RAG + Finetuning Mistral-7B-Instruct
```
python main.py --experiment_config nq_lim_20_docs --model_config mistral_7b_instruct --peft_config lora_r8_a16_qv --eval_split val_anc --train_method sft --print_outputs --num_train_epochs 5 --lr 1e-4 --weight_decay 0.01 --scheduler none --num_train_samples 1000 --num_val_samples 1000 --last_answer_only --max_new_tokens 128
```

Scratchpad-RAG Mistral 7B (`--train_method scratchpad`)  
```
python main.py --experiment_config nq_lim_20_docs --model_config mistral_7b --peft_config lora_r8_a16_qv --eval_split val_anc --train_method scratchpad --print_outputs --num_train_epochs 5 --lr 1e-4 --weight_decay 0.01 --scheduler none --num_train_samples 1000 --num_val_samples 1000 --last_answer_only --max_new_tokens 512
```

Zero-shot Llama2-32K-Instruct
```
python main.py --experiment_config nq_lim_20_docs --model_config llama2_7b_32k_instruct --eval_split val_anc --print_outputs --num_train_samples 1000 --num_val_samples 1000 --eval_only --max_new_tokens 128
```

Finetuning on gold context (only supporting doc) (`--eval_split val_pos`)  
```
python main.py --experiment_config nq_lim_20_docs --model_config mistral_7b --peft_config lora_r8_a16_qv --eval_split val_pos --train_method sft --print_outputs --num_train_epochs 5 --lr 1e-4 --weight_decay 0.01 --scheduler none --num_train_samples 1000 --num_val_samples 1000 --last_answer_only --max_new_tokens 128
```

Zero-shot on gold context (only supporting doc) (`--eval_split val_pos`)  
```
python main.py --experiment_config nq_lim_20_docs --model_config mistral_7b --eval_split val_pos --print_outputs --num_train_samples 1000 --num_val_samples 1000 --eval_only --max_new_tokens 128
```
