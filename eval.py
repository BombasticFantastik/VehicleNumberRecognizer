from Dataset import NumberDataset
from torch.utils.data import DataLoader
from Model import CRNN
import torch
import os
from Dataset import collate_fn
from torch import nn
from torch.optim import AdamW
from Loop import Eval

device = 'cuda'

alphabet=[symb for symb in '_ABEKMHOPCTYX0123456789']
let2int={i:let for let,i in enumerate(alphabet)}
int2let={let:i for let,i in enumerate(alphabet)}

eval_number_data=NumberDataset(path='//home/artemybombastic/MyGit/VehicleNumberData/VNR_Data/val',number_len=9,let2int=let2int)
eval_number_dataloader=DataLoader(eval_number_data,batch_size=16,shuffle=False,drop_last=False,collate_fn=collate_fn)

model=CRNN(input_size=3,hidden_size=64,out_size=len(alphabet)).to(device)

if f'crnn_weights.pth' in os.listdir('../VehicleNumberData/VNR_Data/weights/'):
    weights_dict=torch.load(f'../VehicleNumberData/VNR_Data/weights/crnn_weights.pth',weights_only=True)
    model.load_state_dict(weights_dict)
    print('Веса обнаружены')

Eval(model=model,dataloader=eval_number_dataloader,device=device,blank=0,int2let=int2let)