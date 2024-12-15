### TexTCNN
import torch.nn as nn
from config import parsers
from transformers import BertModel
import torch
import torch.nn.functional as F

class DPCNN(nn.Module):
    def __init__(self):
        super(DPCNN,self).__init__()
        self.channel_size = 250
        self.conv_region_embedding = nn.Conv2d(1,self.channel_size,(3,parsers().hidden_size),stride=1)
        self.conv3 = nn.Conv2d(self.channel_size,self.channel_size,(3,1),stride=1)
        self.pooling = nn.MaxPool2d(kernel_size=(3,1),stride=2)
        self.padding_conv = nn.ZeroPad2d((0,0,1,1))
        self.padding_pool = nn.ZeroPad2d((0,0,0,1))
        self.linear_out = nn.Linear(2 * self.channel_size,parsers().class_num)
    def forward(self,x):
        x = x.unsqueeze(1)
        batch = x.shape[0]
        x = self.conv_region_embedding(x)
        x = self.padding_conv(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = self.padding_conv(x)
        x = F.relu(x)
        x = self.conv3(x)
        
        while x.size()[-2] > 2:
            x = self._block(x)
        
        x = x.view(batch,2 * self.channel_size)
        x = self.linear_out(x)
        
        return x
    def _block(self,x):
        x = self.padding_pool(x)
        px = self.pooling(x)
        
        x = self.padding_conv(px)
        x = F.relu(x)
        x = self.conv3(x)
        
        x = self.padding_conv(x)
        x = F.relu(x)
        x = self.conv3(x)
        
        x = x + px
        return x


class TextCNN(nn.Module):
    def __init__(self):
        super(TextCNN, self).__init__()
        self.num_filter = parsers().num_filters
        self.num_filter_total = parsers().num_filters * len(parsers().filter_sizes)
        self.weight = nn.Linear(self.num_filter_total, parsers().class_num, bias=False)
        self.bias = nn.Parameter(torch.ones([parsers().class_num]))
        self.filter_list = nn.ModuleList([
            nn.Conv2d(1, parsers().num_filters, kernel_size=(size, parsers().hidden_size)) for size in parsers().filter_sizes
        ])
    def forward(self,x):
        x = x.unsqueeze(1)
        pooled_outputs = []
        for i,conv in enumerate(self.filter_list):
            out = F.relu(conv(x))
            #一层卷积
            maxPool = nn.MaxPool2d(
                kernel_size=(parsers().encode_layer - parsers().filter_sizes[i] + 1,1)
            )
            pooled = maxPool(out).permute(0,3,2,1)
            pooled_outputs.append(pooled)
            #一层池化
            
        h_pool = torch.cat(pooled_outputs,len(parsers().filter_sizes))
        h_pool_flat = torch.reshape(h_pool,[-1,self.num_filter_total])
            
        output = self.weight(h_pool_flat) + self.bias
            
        return output
        
class BertTextModel_encoder_layer(nn.Module):
    def __init__(self):
        super(BertTextModel_encoder_layer, self).__init__()
        self.bert = BertModel.from_pretrained(parsers().bert_pred)
        
        for param in self.bert.parameters():
            param.requires_grad = True
        self.Linear = nn.Linear(parsers().hidden_size,parsers().class_num)
        self.TextCNN = DPCNN()
    
    def forward(self,x):
        input_ids,attention_mask,token_type_ids = x[0],x[1],x[2]
        outputs = self.bert(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids,output_hidden_states = True)
        hidden_states = outputs.hidden_states
        cls_embeddings = hidden_states[1][:,0,:].unsqueeze(1)
        for i in range(2,13):
            cls_embeddings = torch.cat((cls_embeddings,hidden_states[i][:,0,:].unsqueeze(1)),dim = 1)
        pred = self.TextCNN(cls_embeddings)
        return pred

class BertTextModel_last_layer(nn.Module):
    def __init__(self):
        super(BertTextModel_last_layer,self).__init__()
        self.bert = BertModel.from_pretrained(parsers().bert_pred)  
        for param in self.bert.parameters():
            param.requires_grad = True
        
        self.convs = nn.ModuleList(
            [nn.Conv2d(in_channels=1, out_channels=parsers().num_filters, kernel_size=(k, parsers().hidden_size),) for k in parsers().filter_sizes]
        )
        self.dropout = nn.Dropout(parsers().dropout)
        self.fc = nn.Linear(parsers().num_filters * len(parsers().filter_sizes),parsers().class_num)
    
    def conv_pool(self,x,conv):
        x = conv(x)
        x = F.relu(x)
        x = x.squeeze(3)
        size = x.size(2)
        x = F.max_pool1d(x,size)
        x = x.squeeze(2)
        return x
    
    def forward(self,x):
        input_ids,attention_mask,token_type_ids = x[0],x[1],x[2]
        hidden_out = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,output_hidden_states=False)
        out = hidden_out.last_hidden_state.unsqueeze(1)
        out = torch.cat([self.conv_pool(out,conv) for conv in self.convs],1)
        out = self.dropout(out)
        out = self.fc(out)
        return out