from config import parsers
from transformers import BertTokenizer
from torch.utils.data import Dataset,DataLoader
import numpy as np
import torch

def read_data(file):
    all_data = open(file,"r",encoding="utf-8").read().split("\n")
    texts,labels = [],[]
    max_length = 0
    for data in all_data:
        if data:
            #print(data)
            text,label = data.split("_separator_")[1],data.split("_separator_")[3]
            #if len(text) > 60:
                #print(text)
             #   continue
            texts.append(text)
            max_length = max(max_length,len(text))
            labels.append(label)
    return texts,labels
class EarlyStopping:
    def __init__(self,patience=7,verbose=False,delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0
    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss
def adjust_learning_rate(optimizer,epoch,learning_rate = 0.02):
    lr_adjust = {epoch: learning_rate if epoch < 3 else learning_rate*(0.8 ** ((epoch - 3) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))
class MyDataset(Dataset):
    def __init__(self,texts,labels=None,with_label=True):
        self.all_text = texts
        self.all_label = labels
        self.max_len = parsers().max_len
        self.with_label = with_label
        self.tokenizer = BertTokenizer.from_pretrained(parsers().bert_pred)
        
    def __getitem__(self,index):
        text = self.all_text[index]
        encode_pair = self.tokenizer(text,padding='max_length',truncation = True,max_length = self.max_len,return_tensors = 'pt')
        token_ids = encode_pair['input_ids'].squeeze(0)
        attn_masks = encode_pair['attention_mask'].squeeze(0)
        token_type_ids = encode_pair['token_type_ids'].squeeze(0)
        
        if self.with_label:
            label = int(self.all_label[index])
            return token_ids,attn_masks,token_type_ids,label
        else:
            return token_ids,attn_masks,token_type_ids
    def __len__(self):
        return len(self.all_text)
if __name__ == "__main__":
    train_text, train_label,max_length = read_data("./data/train.txt")
    print(train_text[0], train_label[0],max_length,len(train_text[0]))
    trainDataset = MyDataset(train_text, labels=train_label, with_label=True)
    trainDataloader = DataLoader(trainDataset, batch_size=3, shuffle=False)
    #for i, batch in enumerate(trainDataloader):
    #    print(batch[0], batch[1], batch[2], batch[3])