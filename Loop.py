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
            
def Eval(model,dataloader,device='cpu',blank='_',int2let=None):
    model.eval()
    for i in range(5):
        all_accuracy=[]
        for batch in (pbar:=tqdm(dataloader)):
            img,label,label_len=batch
            pred=model(img.to(device))
            
            label=[num.item() for num in label]
            corected_label=[]
            for lenght in label_len:
                #print(1)
                new_label=label[:lenght]
                while len(new_label)<16:
                    new_label.append(0)
                corected_label.append(torch.tensor(new_label))
                #print(label)
                label=label[lenght:]


            corected_pred=[]
            for word in pred.argmax(dim=2).permute(1,0):
                new_word=[let.item() for let in word if let.item()!=0]
                while len(new_word)<16:
                    new_word.append(0)
                corected_pred.append(torch.tensor(new_word))

            #accuracy=[(corected_pred[i]-corected_label[i]).abs().sum().item() for i in range(len(corected_pred))]
            accuracy=((corected_pred[i]-corected_label[i])==0).sum().item()/16
            all_accuracy.append(accuracy)
            #print(accuracy)
            # for word in pred:
            #     word_list=[int2let[let.item()] for let in word if let.item()!=blank]
            #     word=''.join(word_list)
            #     print(word)
            pbar.set_description(f"accuracy: {accuracy}")
        print(f'Средняя точность на тестовой выборке равна {sum(all_accuracy)/len(all_accuracy)}')
        break
            
        