# Scratchpad-RAG
Improving robustness of models over long(er) contexts

## Known dependencies
```
pytorch
omegaconf
transformers==4.36
datasets
tqdm
numpy
matplotlib
```

## Getting started
For running the demo and also training models with Scratchpad-RAG, we provide experiment and model configs in `./configs`. 

For data, sample starting files mimicking Nelson Liu's [Lost in the Middle](https://arxiv.org/abs/2307.03172) multi-document question-answering setting are in `./data/qa_data`. These are copied from the 20 document setting in the [paper repo](/juice2/scr2/mzhang/projects/scratchpad-rag/data/qa_data). 

For models, please see `/configs/model` for sample configs to load models. Our scripts should automatically download the models from HuggingFace, but you should change the `cache_dir` to reflect where you want to save the weights.

#### Training  
Afterwards, we can run the main training script by specifying the experiment and model config as commandline arguments, e.g.,

**Standard RAG + finetune with LoRA**
```
python main.py --experiment_config nq_lim_20_docs --model_config mistral_7b --peft_config lora_r8_a16_qv --train_method sft --num_train_samples 1000 --num_val_samples 1000 --last_answer_only --max_new_tokens 512 --print_outputs
```

**Scratchpad-RAG with LoRA**
```
python main.py --experiment_config nq_lim_20_docs --model_config mistral_7b --peft_config lora_r8_a16_qv --train_method scratchpad --num_train_samples 1000 --num_val_samples 1000 --last_answer_only --max_new_tokens 512 --print_outputs
```

#### Demo    
Alternatively, we can run a RAG-like demo with the following commands:

**Zero-shot RAG**
```
python demo.py --experiment_config nq_lim_20_docs --model_config mistral_7b --checkpoint_path ./checkpoints/mistral_7b/scratchpad_rag.pt
```

**RAG + finetuned LLM**
```
python demo.py --experiment_config nq_lim_20_docs --model_config mistral_7b --checkpoint_path <path_to_checkpoint>.pt
```

**Scratchpad-RAG demo**
```
python demo.py --experiment_config nq_lim_20_docs --model_config mistral_7b --checkpoint_path ./checkpoints/mistral_7b/scratchpad_rag.pt
```

#### Full split evaluation  

We can finally run an subspan exact match evaluation over an entire split with a pretrained checkpoint as follows:

**Scratchpad-RAG**
```
python main.py --experiment_config nq_lim_20_docs --model_config mistral_7b --peft_config lora_r8_a16_qv --eval_split val_anc --train_method scratchpad --print_outputs --num_train_samples 1000 --num_val_samples 1000 --last_answer_only --eval_only --max_new_tokens 512 --load_checkpoint --checkpoint_path './checkpoints/mistral_7b/scratchpad_rag.pt'
```

More example commands with additional argparse args below in Example commands.

## Creating a Scratchpad-RAG training set
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


## More example commands

Standard training (SFT)
```
python main.py --experiment_config nq_lim_20_docs --model_config mistral_7b --peft_config lora_r8_a16_qv --eval_split val_anc --train_method sft --print_outputs --num_train_epochs 3 --lr 1e-4 --weight_decay 0.01 --scheduler none --num_train_samples 1000 --num_val_samples 1000 --last_answer_only --max_new_tokens 128
```

Scratchpad (`--train_method scratchpad`)  
```
python main.py --experiment_config nq_lim_20_docs --model_config mistral_7b --peft_config lora_r8_a16_qv --eval_split val_anc --train_method scratchpad --print_outputs --num_train_epochs 3 --lr 1e-4 --weight_decay 0.01 --scheduler none --num_train_samples 1000 --num_val_samples 1000 --last_answer_only --max_new_tokens 512
```

Zero-shot
```
python main.py --experiment_config nq_lim_20_docs --model_config mistral_7b --eval_split val_anc --print_outputs --num_train_samples 1000 --num_val_samples 1000 --eval_only --max_new_tokens 128
```

SFT on gold context (only supporting doc) (`--eval_split val_pos`)  
```
python main.py --experiment_config nq_lim_20_docs --model_config mistral_7b --peft_config lora_r8_a16_qv --eval_split val_pos --train_method sft --print_outputs --num_train_epochs 3 --lr 1e-4 --weight_decay 0.01 --scheduler none --num_train_samples 1000 --num_val_samples 1000 --last_answer_only --max_new_tokens 128
```

Zero-shot on gold context (only supporting doc) (`--eval_split val_pos`)  
```
python main.py --experiment_config nq_lim_20_docs --model_config mistral_7b --eval_split val_pos --print_outputs --num_train_samples 1000 --num_val_samples 1000 --eval_only --max_new_tokens 128
```

Sample scratchpad-finetuned Mistral (`--load_checkpoint --checkpoint_path`)
```
python main.py --experiment_config nq_lim_20_docs --model_config mistral_7b --peft_config lora_r8_a16_qv --eval_split val_anc --train_method scratchpad --print_outputs --num_train_epochs 3 --lr 1e-4 --weight_decay 0.01 --scheduler none --num_train_samples 1000 --num_val_samples 1000 --last_answer_only --eval_only --max_new_tokens 512 --load_checkpoint --checkpoint_path './checkpoints/mistral_7b/sample_finetuned_mistral_20docs.pt'
```




