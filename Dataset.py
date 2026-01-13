from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms
import torch
import json



class NumberDataset(Dataset):
    def __init__(self,path,number_len,let2int):
        super(NumberDataset,self).__init__()
        self.number_len=number_len
        img_path=os.path.join(path,'img')
        label_path=os.path.join(path,'ann')
        self.let2int=let2int

        #номера
        self.image_numbers=[img[:-4] for img in os.listdir(img_path)]
        self.label_numbers=[label[:-5] for label in os.listdir(label_path)]
        
        #изображения и лейблы 
        self.images=[os.path.join(img_path,img) for img in os.listdir(img_path) if img[:-4] in self.label_numbers]
        self.labels=[os.path.join(label_path,label) for label in os.listdir(label_path) if label[:-5] in self.image_numbers]
        
        self.images.sort(reverse=True)
        self.labels.sort(reverse=True)

        self.trans=transforms.Compose([
            transforms.Resize((64,128)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.images)
    def __getitem__(self,idx):
        idx_img=Image.open(self.images[idx]).convert('RGB')
        idx_label=self.labels[idx]
        
        with open(idx_label,'r') as file_option:
            jf=json.load(file_option)
            #return jf['name'][0:self.number_len]
            tensor_label=torch.tensor([self.let2int[let] for let in jf['description'][0:self.number_len] if let!='_'])
        tensor_img=self.trans(idx_img)
        return {
            'img':tensor_img,
            'label':tensor_label,
            'label_len':len(tensor_label)
        }
    
def collate_fn(batch):
    imgs = torch.stack([x['img'] for x in batch])
    labels=[x['label'] for x in batch]
    label_lens=torch.tensor([x['label_len'] for x in batch])
    label = torch.cat(labels)
    return imgs,label,label_lens