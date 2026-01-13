from torch import nn
from torch.nn import Module


class ResnetBlock(Module):
    def __init__(self,input_size,output_size,stride=1,downsample=None):#downsample нужно в случае понижения размерности блока):
        super().__init__()
        self.act=nn.ReLU(inplace=True)
        self.conv0=nn.Conv2d(input_size,output_size,kernel_size=3,stride=stride,padding=1)
        self.norm0=nn.BatchNorm2d(output_size)

        self.conv1=nn.Conv2d(output_size,output_size,kernel_size=3,stride=1,padding=1)
        self.norm1=nn.BatchNorm2d(output_size)

        self.downsample=downsample
    def forward(self,x):
        out=self.conv0(x)
        out=self.norm0(out)
        out=self.act(out)
        out=self.conv1(out)
        out=self.norm1(out)
        if self.downsample:
            x=self.downsample(x)
        out+=x
        out=self.act(out)

        return out
    

def make_layers(block,cnt,input_size,output_size,stride=1,downsample=False):
    blocks=[]

    if downsample or input_size!=output_size or stride!=1:
        downsample=nn.Sequential(
            nn.Conv2d(input_size,output_size,1,stride,bias=False),
            nn.BatchNorm2d(output_size)
        )

    blocks.append(block(input_size,output_size,stride,downsample))    
    for i in range(1,cnt):
        blocks.append(block(output_size,output_size))

    return nn.Sequential(*blocks)


class Resnet34(Module):
    def __init__(self,input_size,hidden_size):
        super().__init__()

        self.initial_lay=nn.Sequential(
            nn.Conv2d(input_size,hidden_size,kernel_size=7,stride=2,padding=3),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        )

        self.lay0=make_layers(block=ResnetBlock,cnt=3,input_size=hidden_size,output_size=hidden_size,downsample=False)
        self.lay1=make_layers(block=ResnetBlock,cnt=4,input_size=hidden_size,output_size=hidden_size*2,stride=(2,1),downsample=True)
        self.lay2=make_layers(block=ResnetBlock,cnt=6,input_size=hidden_size*2,output_size=hidden_size*4,stride=(2,1),downsample=True)
        self.lay3=make_layers(block=ResnetBlock,cnt=3,input_size=hidden_size*4,output_size=hidden_size*8,stride=(2,1),downsample=True)

        self.avg_pool=nn.AdaptiveAvgPool2d((1,None))
        
    def forward(self,x):
        #print(x.shape)
        out=self.initial_lay(x)
        #print(out.shape)
        #print('СЛОИ:')
        out=self.lay0(out)
        #print(out.shape)
        out=self.lay1(out)
        #print(out.shape)
        out=self.lay2(out)
        #print(out.shape)
        out=self.lay3(out)
        #print(out.shape)

        final_out=self.avg_pool(out)
        #print(final_out.shape)
        return final_out
    
        

class CRNN(Module):
    def __init__(self,input_size,hidden_size,out_size):
        super().__init__()

        self.cnn=Resnet34(3,64)
        self.rnn=nn.LSTM(hidden_size*8,hidden_size*4,num_layers=1,bidirectional=True)
        self.final_lay=nn.Sequential(    
            nn.Linear(512,out_size)
        )
                
    def forward(self,x):
        out=self.cnn(x)

        out=out.squeeze(2).permute(2,0,1)
        out,_=self.rnn(out)

        out=self.final_lay(out)

        #финальная размерность длина_строки*batch*размер_алфвавита

        return out
        