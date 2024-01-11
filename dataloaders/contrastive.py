"""
Contrastive Dataset for robust sequence modeling
"""
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import default_data_collator


class ContrastiveContextDataset(Dataset):
    def __init__(self, 
                 anc_dataset: Dataset, 
                 pos_dataset: Dataset, 
                 num_samples: int = None, 
                 seed: int = None):
        super().__init__()
        self.anc_dataset = anc_dataset
        self.pos_dataset = pos_dataset
            
    def __len__(self):
        return len(self.anc_dataset)

    def __getitem__(self, idx: int):
        anc_data = self.anc_dataset[idx]
        pos_data = self.pos_dataset[idx]
        data = {k: v for k, v in anc_data.items()}
        data['pos_input_ids'] = pos_data['input_ids']
        data['pos_attention_mask'] = pos_data['attention_mask']
        data['pos_labels'] = pos_data['labels']
        return  data
