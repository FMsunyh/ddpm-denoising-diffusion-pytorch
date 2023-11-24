from torch import nn
import torch
from typing import Optional, Tuple, Union,List
import torch.nn.functional as F

class TimeEmbedding(nn.Module):
    def __init__(self, n_channels) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.lin1 = nn.Linear(self.n_channels//4, self.n_channels)
        
        self.act = Swish()
        
        self.lin2 = nn.Linear(self.n_channels, self.n_channels)
        
        
        
        
        
class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_channels: int, n_groups: int = 32, dropout: float = 0.1) -> None:
        super().__init__()

        self.norm1 = nn.GroupNorm(n_groups, in_channels) 
        self.act1 = Swish()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), padding=(1,1))

        self.norm2 = nn.GroupNorm(n_groups, out_channels) 
        self.act2 = Swish()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3,3), padding=(1,1))
        
        if in_channels!= out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=(1,1))
        else:
            self.shortcut = nn.Identity()
            
        self.time_emb = nn.Linear(time_channels, out_channels)
        self.time_act = Swish()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x:torch.Tensor, t: torch.Tensor):
        """
        * `x` has shape `[batch_size, in_channels, height, width]`
        * `t` has shape `[batch_size, time_channels]`
        """
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act1(h)
        h += self.time_emb(self.time_act(t))[:, :, None, None]
        
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act2(h)
        
        return h+self.shortcut(x)
        
class DownBlock(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, time_channels:int,  has_attn: bool) -> None:
        super().__init__()

class Downsample(nn.Module):
    def __init__(self,n_channels) -> None:
        super().__init__()
        self.conv = nn.Conv2d(n_channels, n_channels, (3,3), (2,2), (1,1))    
    
    def forward(self,x:torch.Tensor, t:torch.Tensor):
        _ = t
        return self.conv(x)

class MiddleBlock(nn.Module):
    def __init__(self, n_channels:int, time_channels:int) -> None:
        super().__init__()
        
class UpBlock(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, time_channels:int, has_attn:bool) -> None:
        super().__init__()

class Upsample(nn.Module):
    def __init__(self, n_channels) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose2d(n_channels,n_channels,(4,4),(2,2),(1,1))       
        
    def forward(self, x:torch.Tensor, t:torch.Tensor) :
        _ =t
        return self.conv(x)
    
 
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
    
class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x)) 
    
class UNet(nn.Module):
    def __init__(self, image_channels:int=3,n_channels:int=64,
                ch_mults:Union[Tuple[int,...], List[int]]=(1,2,3,4),
                is_attn:Union[Tuple[bool, ...], List[bool]]=(False, False, True, True),
                n_blocks:int=2):
        
        super.__init__()
        
        n_resolutions = len(ch_mults)
        self.image_proj = nn.Conv2d(image_channels, n_channels, kernel_size=(3,3), padding=(1,1))
        self.time_emb = TimeEmbedding(n_channels*4) #64*4 = 256
        down = []
        
        out_channels = in_channels = n_channels
        for i in range(n_resolutions): # 64,126,256,512
            out_channels=in_channels * ch_mults[i]
            
            for _ in range(n_blocks):
                down.append(DownBlock(in_channels, out_channels, n_channels*4, is_attn[i]))
                in_channels = out_channels
      
            if i < n_resolutions-1:
                down.append(Downsample(in_channels))
                
        self.down = nn.ModuleList(down)
        
        self.middle = MiddleBlock(out_channels, n_channels*4,)
        
        up = []
        
        in_channels = out_channels
        for i in reversed(range(n_resolutions)):
            out_channels = in_channels
            for _ in range(n_blocks):
                up.append(UpBlock(in_channels, out_channels, n_channels*4, is_attn[i]))
            
            out_channels = in_channels //ch_mults[i]
            up.append(UpBlock(in_channels, out_channels, n_channels*4, is_attn[i]))
            in_channels = out_channels
            
            if i > 0:
                up.append(Upsample(in_channels))
        
        self.up = nn.ModuleList(up)
        
        self.norm  = nn.GroupNorm(8, n_channels)
        self.act = Swish()
        self.final = nn.Conv2d(in_channels, image_channels, kernel_size=(3,3), padding=(1,1))
        
        