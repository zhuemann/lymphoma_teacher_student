import torch
import os
from os.path import exists
from torch.utils.data import Dataset
from PIL import Image
import io
import numpy as np

class TextImageDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len, truncation=True, dir_base='/home/zmh001/r-fcb-isilon/research/Bradshaw/', mode="train"): # data_path = os.path.join(dir_base,'Lymphoma_UW_Retrospective/Data/mips/')
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.text
        self.targets = self.data.label
        self.row_ids = self.data.index
        self.max_len = max_len

        self.df_data = dataframe.values
        # self.data_path = data_path
        self.mode = mode
        self.data_path = os.path.join(dir_base, 'Lymphoma_UW_Retrospective/Data/mips/')

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):

        # text extraction
        text = str(self.text[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float),
            'row_ids': self.row_ids[index],
        }