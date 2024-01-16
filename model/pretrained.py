"""
Classes for loading pretrained models
"""
import torch
from os.path import join
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import MistralForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_pretrained_loader(pretrained_model_name_or_path: str, 
                          **model_kwargs: any):
    if 'llama' in pretrained_model_name_or_path:
        return PretrainedLlamaLoader(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            **model_kwargs,
        )
    elif 'mistral' in pretrained_model_name_or_path:
        return PretrainedMistralLoader(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            **model_kwargs,
        )
    else:
        return PretrainedModelLoader(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            **model_kwargs,
        )


class PretrainedModelLoader():
    def __init__(self,
                 pretrained_model_name_or_path: str,
                 cache_dir: str = None,
                 return_dict: bool = True,  # False
                 quantization: bool = False,
                 device_map: str = 'auto',
                 low_cpu_mem_usage: bool = True,
                 torch_dtype: str = 'bfloat16',
                 rope_theta: float = 10000.,
                 attn_implementation: str = 'flash_attention_2',  # eager
                 **other_kwargs: any):

        print(f'-> Using {attn_implementation} attention')
        
        self.loading_kwargs = {
            'pretrained_model_name_or_path': pretrained_model_name_or_path,
            'cache_dir': cache_dir,
            'return_dict': return_dict,
            'load_in_8bit': quantization,
            'device_map': device_map,
            'low_cpu_mem_usage': low_cpu_mem_usage,
            'torch_dtype': getattr(torch, torch_dtype),
            'rope_theta': rope_theta,
            'attn_implementation': attn_implementation,
        }
        for k, v in other_kwargs.items():
            self.loading_kwargs[k] = v
        
    def load(self):
        return AutoModelForCausalLM.from_pretrained(**self.loading_kwargs)

    def load_tokenizer(self):
        return AutoTokenizer.from_pretrained(**self.loading_kwargs)


class PretrainedLlamaLoader(PretrainedModelLoader):
    """
    Load pretrained LLaMA or Llama 2 for causal language modeling

    Currently assume weights are locally stored in a directory like
        `/juice5/scr5/nlp/llama_model/llama-2-7b-hf`
        
    We can then load by specifying:
    - pretrained_model_name_or_path = 'llama-2-7b-hf'
    - cache_dir = '/juice5/scr5/nlp/llama_model/'
    """
    def __init__(self, 
                 pretrained_model_name_or_path: str,
                 cache_dir: str = None,
                 rope_scaling: dict = None, 
                 *args, **kwargs):
        """
        Same as parent class other than above + support rope_scaling
        """
        if cache_dir is not None:
            pretrained_model_name_or_path = join(
                cache_dir, pretrained_model_name_or_path)
            cache_dir = None
        super().__init__(pretrained_model_name_or_path=pretrained_model_name_or_path,
                         cache_dir=cache_dir, *args, **kwargs)
        self.loading_kwargs['rope_scaling'] = rope_scaling

    def load(self):
        return LlamaForCausalLM.from_pretrained(**self.loading_kwargs)

    def load_tokenizer(self):
        return LlamaTokenizer.from_pretrained(**self.loading_kwargs)


class PretrainedMistralLoader(PretrainedModelLoader):
    def load(self):
        return MistralForCausalLM.from_pretrained(**self.loading_kwargs)
        
