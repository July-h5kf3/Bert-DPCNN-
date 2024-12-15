import os
import time
from tqdm import tqdm
import numpy as np
from config import parsers
from utils import read_data,MyDataset,EarlyStopping,adjust_learning_rate
from torch.utils.data import DataLoader
from module import BertTextModel_encoder_layer,BertTextModel_last_layer
import torch
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def train(model,device,trainLoader,opt,epoch):
    model.train()
    loss_sum,count = 0,0
    for batch_index,batch_con in enumerate(trainLoader):
        batch_con = tuple(p.to(device) for p in batch_con)
        pred = model(batch_con)
        
        opt.zero_grad()
        loss = loss_fn(pred,batch_con[-1])
        loss.backward()
        opt.step()
        loss_sum += loss
        count += 1
        
        if count % 100 == 0:
            print("epoch", epoch, end='  ')
            print("The loss is: %.5f" % (loss_sum/100))

            loss_sum = 0
            count = 0

def dev(model,device,devLoader,save_best):
    global acc_min
    model.eval()
    all_true,all_pred = [],[]
    total_loss = []
    for batch_con in devLoader:
        batch_con = tuple(p.to(device) for p in batch_con)
        pred = model(batch_con)
        total_loss.append(loss_fn(pred,batch_con[-1]).cpu().detach().numpy().item())
        pred = torch.argmax(pred,dim = 1)
        pred_label = pred.cpu().numpy().tolist()
        true_label = batch_con[-1].cpu().numpy().tolist()
        all_true.extend(true_label)
        all_pred.extend(pred_label)
    total_loss = np.average(total_loss)
    acc = accuracy_score(all_true,all_pred)
    print(f"dev acc:{acc:.4f}")
    
    #if acc > acc_min:
        #acc_min = acc
        #torch.save(model.state_dict(),save_best)
        #print(f"已保存最佳模型")
    return total_loss
    
if __name__ == "__main__":
    start = time.time()
    args = parsers()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    Data_text,Data_label = read_data(args.train_file)
    train_text,dev_text,train_label,dev_label = train_test_split(Data_text,Data_label,test_size=0.01,random_state=42)
    
    trainData = MyDataset(train_text,train_label,with_label=True)
    trainLoader = DataLoader(trainData,batch_size=args.batch_size,shuffle=True)
    
    devData = MyDataset(dev_text,dev_label,with_label=True)
    devLoader = DataLoader(devData,batch_size=args.batch_size,shuffle=True)
    
    root,name = os.path.split(args.save_model_best)
    save_best = os.path.join(root,name)
    root,name = os.path.split(args.save_model_last)
    save_last = os.path.join(root,name)
    
    if args.select_model_last:
        model = BertTextModel_last_layer().to(device)
    else:
        model = BertTextModel_encoder_layer().to(device)
        
    opt = SGD(model.parameters(),lr=args.learn_rate)
    loss_fn = CrossEntropyLoss()
    
    acc_min = float("-inf")
    early_stopping = EarlyStopping(patience=5,verbose=True)
    for epoch in range(args.epochs):
        train(model,device,trainLoader,opt,epoch)
        vail_loss = dev(model,device,devLoader,save_best)
        early_stopping(vail_loss,model,save_best)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        adjust_learning_rate(opt,epoch+1)
        model.load_state_dict(torch.load(save_best, map_location="cuda:0"))
        
    model.eval()
    torch.save(model.state_dict(),save_last)
    end = time.time()
    print(f"运行时间:{(end - start) / 60%60:.4f} min")
