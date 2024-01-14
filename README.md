# scratchpad-rag
Improving robustness of models over long(er) contexts

## Known dependencies
```
pytorch 2
omegaconf
transformers
datasets
tqdm
numpy
matplotlib
```

## Example commands

Standard training (ERM)
```
python main.py --experiment_config nq_lim_20_docs --model_config mistral_7b --peft_config lora_r8_a16_qv --eval_split val_anc --train_method sft --print_outputs --num_train_epochs 3 --lr 1e-4 --weight_decay 0.01 --scheduler none --num_train_samples 1000 --num_val_samples 1000 --last_answer_only --max_new_tokens 512
```

Scratchpad (`--train_method scratchpad`)  
```
python main.py --experiment_config nq_lim_20_docs --model_config mistral_7b --peft_config lora_r8_a16_qv --eval_split val_anc --train_method scratchpad --print_outputs --num_train_epochs 3 --lr 1e-4 --weight_decay 0.01 --scheduler none --num_train_samples 1000 --num_val_samples 1000 --last_answer_only --max_new_tokens 512
```

Zero-shot
```
python main.py --experiment_config nq_lim_20_docs --model_config mistral_7b --eval_split val_anc --print_outputs --num_train_samples 1000 --num_val_samples 1000 --last_answer_only --eval_only --max_new_tokens 512
```

ERM on gold context (only supporting doc) (`--eval_split val_pos`)  
```
python main.py --experiment_config nq_lim_20_docs --model_config mistral_7b --peft_config lora_r8_a16_qv --eval_split val_pos --train_method sft --print_outputs --num_train_epochs 3 --lr 1e-4 --weight_decay 0.01 --scheduler none --num_train_samples 1000 --num_val_samples 1000 --last_answer_only --max_new_tokens 512
```

Zero-shot on gold context (only supporting doc) (`--eval_split val_pos`)  
```
python main.py --experiment_config nq_lim_20_docs --model_config mistral_7b --eval_split val_pos --print_outputs --num_train_samples 1000 --num_val_samples 1000 --last_answer_only --eval_only --max_new_tokens 512 
```

Sample scratchpad-finetuned Mistral (`--load_checkpoint --checkpoint_path`)
```
python main.py --experiment_config nq_lim_20_docs --model_config mistral_7b --peft_config lora_r8_a16_qv --eval_split val_anc --train_method scratchpad --print_outputs --num_train_epochs 3 --lr 1e-4 --weight_decay 0.01 --scheduler none --num_train_samples 1000 --num_val_samples 1000 --last_answer_only --eval_only --max_new_tokens 512 --load_checkpoint --checkpoint_path './checkpoints/mistral_7b/sample_finetuned_mistral_20docs.pt'
```
