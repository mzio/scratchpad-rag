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






## Random notes / gibberish thoughts
How can we improve the performance of models to use their context more effectively?  

On the flipside, once we do this, can we then use these models in RAG-like pipelines more effectively / with less worry?

Need to show:  
1. (Problem): Models may be good at certain evaluations (needle-in-haystack), but still fall short on things like NaturalQuestions lost in the middle 
2. (Why interesting / important) Talk about why this is troubling. If we can't use contexts, then what good are these models? Also interesting to see that the extra information can be distracting and hurt performance. As a (GPU-poor) proxy we can study this with things like Mistral, trying to answer questions over 4K context. (Even with models that claim shorter context, can they still be distracted?) 
3. (Why challenging / hard) Frame as a robustness problem. Usually the best thing to do with these is finetune / train on more balanced data (finetune on examples where the answer is in the middle). (Ideally we can do this, but with these big LLMs in the pipeline we might not be able to finetune. So we'll use LoRa.). But find that this doesn't always help?
4. (Our approach): So we'll instead show two different approaches that seem promising. First is to show that these models can often answer the question if they have the gold information. This motivates being careful with the retrievers. But can we cheaply train these models to pick out this information / focus on it? Two approaches  
  * First: simply train model to output the relevant / supporting information.
  * Second: do a contrastive approach inspired by what we did before with robustness.
5. Together, show that both of these can improve.


Show further that this works with longer context? Can fill in the gaps for stuff going up to 16k in context length?  

Finally show that we can apply the reverse to this too to make more effective systems. We don't need a large long-context model all the time (though the effect may compound?). Instead we can combine a RAG / retrieval system over a large document to get chunks. Train the LLM to output the relevant chunks, and train over this to get the outputs. This is kinda like process supervision / scratchpads. 

After training on this, does this let our model use the context more effectively?



TODO:
Do we have the SFT results for


How to make these things not distracted by long contexts? 

As context-size grows, does the performance go down? Get more and more distracted? 

Show that as the number of distractors increases, the performance of the model goes down?

On the flipside, can we show that by doing our procedure, if we increase the number of documents / chunks retrieved the performance goes up?
* Can evaluate this on something like SCROLLs? Long-context benchmark? Retrieve chunks that fit up to 4k (mistral). And do training over this?

As we retrieve more documents / evaluate on more documents, the performance of the trained RAG model should increase?

* Show curve where x-axis is number of documents presented (can be top X; random / could be missing) to reflect reality. Performance increases as documents go up?
* Or its a plot to show robustness. As the number of documents increases, the performance still says the same (the document is somewhere in there). 

* Also should compare the finetuned performance vs not on just the positive selection?
