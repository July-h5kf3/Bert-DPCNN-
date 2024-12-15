from module import BertTextModel_encoder_layer,BertTextModel_last_layer
from utils import MyDataset
from transformers import BertTokenizer
from torch.utils.data import DataLoader
import torch
from config import parsers
import time
import numpy as np
import pandas as pd
import os


def load_model(model_path,device,args):
    if args.select_model_last:
        model = BertTextModel_last_layer().to(device)
    else:
        model = BertTextModel_encoder_layer().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def text_class_name(texts,pred,args):
    results = torch.argmax(pred,dim=1)
    results = results.cpu().numpy().tolist()
    classification = open(args.classification,"r",encoding="utf-8").read().split("\n")
    classification_dict = dict(zip(range(len(classification)),classification))
    if len(results) != 1:
        for i in range(len(results)):
            print(f"文本：{texts[i]}\t预测的类别为：{classification_dict[results[i]]}")
    else:
        print(f"文本：{texts}\t预测的类别为：{classification_dict[results[0]]}")
        
def pred_one(args, model, device,text):
    tokenizer = BertTokenizer.from_pretrained(parsers().bert_pred)
    encoded_pair = tokenizer(text, padding='max_length', truncation=True,  max_length=args.max_len, return_tensors='pt')
    token_ids = encoded_pair['input_ids']
    attn_masks = encoded_pair['attention_mask']
    token_type_ids = encoded_pair['token_type_ids']
    all_con = tuple(p.to(device) for p in [token_ids, attn_masks, token_type_ids])
    pred = model(all_con)
    result = torch.argmax(pred,dim = 1)
    result = result.cpu().numpy().tolist()
    #print(result[0])
    classification = open(args.classification,"r",encoding="utf-8").read().split("\n")
    classification_dict = dict(zip(range(len(classification)),classification))
    return classification_dict[result[0]]
    #text_class_name(text, pred, args)
    
if __name__ == "__main__":
    start = time.time()
    args = parsers()
    dic = {
        "entertainment":0,
        "world":1,
        "sports":2,
        "culture":3,
        "car":4,
        "edu":5,
        "story":6,
        "finance":7,
        "argiculture":8,
        "tech":9,
        "military":10,
        "travel":11,
        "game":12,
        "house":13,
        "stock":14
    }
    classification = open(args.classification,"r",encoding="utf-8").read().split("\n")
    classification_dict = dict(zip(range(len(classification)),classification))
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    root, name = os.path.split(args.save_model_last)
    save_best = os.path.join(root, str(args.select_model_last) + "_" +name)
    model = load_model(save_best, device, args)

    
    print("模型预测结果：")
    path = "./data/test1.txt"
    texts = []
    all_data = open(path,"r",encoding="utf-8").read().split('\n')
    for data in all_data:
        if data:
            text = data.split("_separator_")[1]
            texts.append(text)
    x = MyDataset(texts,with_label=False)
    xDataLoader = DataLoader(x,batch_size = 32,shuffle=False)
    ans = []
    for batch_index,batch_con in enumerate(xDataLoader):
        batch_con = tuple(p.to(device) for p in batch_con)
        pred = model(batch_con)
        result = torch.argmax(pred,dim = 1)
        result = result.cpu().numpy().tolist()
        if len(result) != 1:
            for i in range(len(result)):
                ans.append([result[i]])
    output = pd.DataFrame(ans,index=None,columns=None)
    output.index = np.arange(1,len(output)+1)
    output.to_csv("./data/submit.csv")

