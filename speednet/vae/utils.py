import torch
import torch.nn as nn

class Flatten(nn.Module):
    def forward(self , input):
        return input.view(input.size(0) , -1)
    
    
class Unflatten(nn.Module):
    def __init__(self , channel , height , width):
        super(Unflatten , self).__init__()
        
        self.channel = channel
        self.height = height
        self.width = width
          
    def forward(self , input):
        return input.view(input.size(0) , self.channel , self.height , self.width)
    
    
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.uniform_(m.weight, -0.08 , 0.08)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.uniform_(m.weight, -0.08 , 0.08)
        torch.nn.init.zeros_(m.bias)
    if classname.find('ConvTranspose') != -1:
        torch.nn.init.uniform_(m.weight, -0.08 , 0.08)
        

def down_conv(in_chan , act_func , out_chan):
        conv_blocks = []
        conv_blocks.append(nn.Conv2d(in_chan , out_chan  , 3 , 1 , 1))

        if act_func == 'ReLU':
            conv_blocks.append(nn.ReLU())
            conv_blocks.append((nn.BatchNorm2d(out_chan)))
            conv_blocks.append(nn.Conv2d(out_chan , out_chan*2 , 4 , 2 , 1))
            conv_blocks.append(nn.ReLU())
            conv_blocks.append((nn.BatchNorm2d(out_chan*2)))

        elif act_func == 'LeakyReLU':
            conv_blocks.append(nn.LeakyReLU(0.02))
            conv_blocks.append((nn.BatchNorm2d(out_chan)))
            conv_blocks.append(nn.Conv2d(out_chan , out_chan*2 , 4 , 2 , 1))
            conv_blocks.append(nn.LeakyReLU(0.02))
            conv_blocks.append((nn.BatchNorm2d(out_chan*2)))

        conv_blocks = nn.Sequential(*conv_blocks)
        return conv_blocks
    
    
    
def up_conv(in_chan , act_func , out_chan ):
    up_conv_blocks = []
    up_conv_blocks.append(nn.ConvTranspose2d(in_chan , out_chan , 3 , 1 , 1))

    if act_func == 'ReLU':
        up_conv_blocks.append(nn.ReLU())
        up_conv_blocks.append((nn.BatchNorm2d(out_chan)))
        up_conv_blocks.append(nn.ConvTranspose2d(out_chan , out_chan//2 , 4 , 2 , 1))
        up_conv_blocks.append(nn.ReLU())
        up_conv_blocks.append((nn.BatchNorm2d(out_chan//2)))

    elif act_func == 'LeakyReLU':
        up_conv_blocks.append(nn.LeakyReLU(0.02))
        up_conv_blocks.append((nn.BatchNorm2d(out_chan)))
        up_conv_blocks.append(nn.ConvTranspose2d(out_chan , out_chan//2 , 4 , 2 , 1))
        up_conv_blocks.append(nn.LeakyReLU(0.02))
        up_conv_blocks.append((nn.BatchNorm2d(out_chan//2)))
        
    up_conv_blocks = nn.Sequential(*up_conv_blocks)
    return up_conv_blocks