import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
import pandas as pd

class dataset_adapter(data_utils.Dataset):
    
    def __init__(self,datapath,tokenizer):
        self.data=pd.read_csv(datapath,sep='\t',names=['source','acceptibility','originality','sentence'])[['acceptibility','sentence']]
        self.tokenizer=tokenizer
    
    
    def __len__(self,data):
        return self.data.shape[0]
    
    
    def __getitem__(self,idx):
        row=self.data.iloc[idx]
        
        tokenised_sentence=self.tokenizer(row['sentence'],return_tensors="pt")
        label=row['acceptibility']
        
        return {"acceptibility":label,"sentence":tokenised_sentence}