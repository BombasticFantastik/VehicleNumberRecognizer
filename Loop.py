import torch
from tqdm import tqdm
import yaml

option_path='config.yaml'
with open(option_path,'r') as file_option:
    option=yaml.safe_load(file_option)

def Train(model,optimizer,loss_fn,dataloader,device='cpu'):
    model.train()

    losses=[]
    for batch in (pbar:=tqdm(dataloader)):
        optimizer.zero_grad()
        img,label,label_len=batch
        print(img.shape)
        pred=model(img.to(device))

        T = pred.size(0)
        N = pred.size(1)
        input_len = torch.full(size=(N,), fill_value=T, dtype=torch.int32)

        
        pred=pred.log_softmax(dim=2)
        loss=loss_fn(pred,label,input_len,label_len)
        loss.backward()
        loss_item=loss.item()
        losses.append(loss_item)
        optimizer.step()
        pbar.set_description(f"loss: {loss_item}")
        
        #pbar.set_descriptiont()
    try:
        torch.save(model.state_dict(),option['weights'])
    except:
        print('Ошибка загрузки')
    print(f'mean_loss: {sum(losses)/len(losses)}')
            



def ctc_decoder(pred_string,int2let):
    new_string=[]
    perv_symb=-1
    for symb in pred_string:
        if symb.item()!=perv_symb:
            if symb.item()!=0:
                new_string.append(int2let[symb.item()])
        perv_symb=symb
    return ''.join(new_string)


def Eval(model,dataloader,device='cpu',blank='_',int2let=None):
        
    model.eval()

    all_accuracy=[]
    for batch in (pbar:=tqdm(dataloader)):
        img,label,label_len=batch
        pred=model(img.to(device))

        #форматирование label
        label=[num.item() for num in label]
        corected_label=[]
        for lenght in label_len:
            new_label=label[:lenght]
            new_label=[int2let[num] for num in new_label]
            new_label=''.join(new_label)
            corected_label.append(new_label)
            label=label[lenght:]

        #форматирование pred
        corected_pred=[ctc_decoder(word,int2let) for word in pred.argmax(dim=2).permute(1,0)]
        #print(len(corected_pred))
        accuracy=[corected_label[i]==corected_pred[i] for i in range(len(corected_pred))]
        accuracy=sum(accuracy)/len(accuracy)
        
        all_accuracy.append(accuracy)
        pbar.set_description(f"accuracy: {accuracy}")
    print(f'Средняя точность на тестовой выборке равна {sum(all_accuracy)/len(all_accuracy)}')