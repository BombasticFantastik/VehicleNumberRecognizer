import torch
from tqdm import tqdm

def Train(model,optimizer,loss_fn,dataloader,device='cpu'):
    model.train()
    for i in range(5):
        losses=[]
        for batch in (pbar:=tqdm(dataloader)):
            optimizer.zero_grad()
            img,label,label_len=batch
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
            torch.save(model.state_dict(),r'../VehicleNumberData/VNR_Data/weights/crnn_weights.pth')
        except:
            print('Ошибка загрузки')
        print(f'mean_loss: {sum(losses)/len(losses)}')
            
        