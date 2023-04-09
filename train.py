import torch
from config import *
from torch import nn
from dataset import dataset_adapter
from torch.utils.data import DataLoader

import wandb
from tqdm import tqdm, trange
from engine import BertClassifier
from transformers import BertTokenizer, AdamW , get_linear_schedule_with_warmup

class Trainer:
    def __init__(self):
        self.model = BertClassifier(num_labels = 1)
        self.gpu_present = torch.cuda.is_available()
        wandb.init(project="c4ai-adapter-bert")

        if self.gpu_present:
            self.model = self.model.cuda()

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.train_dataset = dataset_adapter('/Users/yashsurange/adapter_bert/cola_dataset/raw/in_domain_train.tsv', tokenizer)
        self.val_dataset = dataset_adapter('/Users/yashsurange/adapter_bert/cola_dataset/raw/in_domain_dev.tsv', tokenizer)

        self.loss_fct = nn.BCEWithLogitsLoss()
        
    def configure_optimizers(self):
        layers = ["adapter", "LayerNorm"]
        params = [p for n, p in self.model.named_parameters() \
                        if any([(nd in n) for nd in layers])]
        
        self.optimizer = AdamW(params, lr=learning_rate)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,num_training_steps = len(self.train_dataloader) * epochs, num_warmup_steps = int(0.1 * len(self.train_dataset))*epochs
        )

    def compute_loss(self, output, labels):
        return self.loss_fct(output, labels)
        

    def train(self):
        
        self.train_dataloader=DataLoader(self.train_dataset,batch_size,shuffle=True)
        self.val_dataloader=DataLoader(self.val_dataset,batch_size,shuffle=False)
        self.configure_optimizers()
        
        # Training step
        for i in trange(epochs):
            training_loss=0.0
            
            for batch in tqdm(self.train_dataloader):
                
                input,labels,mask=batch['input_ids'].squeeze(1),batch['acceptibility'],batch['attention_mask']
                
                if self.gpu_present:
                    input,labels,mask=input.cuda(),labels.cuda(),mask.cuda()
                
                labels=labels.float()
                outputs=self.model(input)
                
                
                loss=self.compute_loss(outputs, labels)
                
                
                training_loss+=loss.item()
              
                self.optimizer.zero_grad()
                loss.backward()
                
                self.optimizer.step()
                self.scheduler.step()
                
            # Validation Step
            valid_loss = 0.0
            with torch.no_grad():
                for batch in tqdm(self.valloader):
                    inputs, labels,mask = batch['input_ids'].squeeze(1), batch['labels'],batch['attention_mask']
                    if self.gpu_present:
                        inputs, labels,mask = inputs.cuda(), labels.cuda(),mask.cuda()
                    
                    outputs = self.model(inputs)
                    labels=labels.float()
                    loss = self.compute_loss(outputs, labels)
                    valid_loss += loss.item()
                
            wandb.log({
                    'epoch': e,
                    'train_loss': train_loss/len(trainloader),
                    'val_loss': valid_loss/len(valloader)
                })
            print(f'Epoch {e}\t\tTraining Loss: {train_loss/len(trainloader)}\t\tValidation Loss: {valid_loss/len(valloader)}')
            wandb.finish()
                
            
if __name__=='__main__':
    Trainer().train()