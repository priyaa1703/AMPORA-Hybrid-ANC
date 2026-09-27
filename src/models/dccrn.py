from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, cin, cout, stride=(1,1)):
        super().__init__()
        self.block=nn.Sequential(
            nn.Conv2d(cin,cout,3,stride=stride,padding=1), nn.BatchNorm2d(cout), nn.PReLU(cout),
            nn.Conv2d(cout,cout,3,padding=1), nn.BatchNorm2d(cout), nn.PReLU(cout))
    def forward(self,x): return self.block(x)

class DecoderBlock(nn.Module):
    def __init__(self, cin, skip, cout):
        super().__init__()
        self.block=nn.Sequential(
            nn.Conv2d(cin+skip,cout,3,padding=1), nn.BatchNorm2d(cout), nn.PReLU(cout),
            nn.Conv2d(cout,cout,3,padding=1), nn.BatchNorm2d(cout), nn.PReLU(cout))
    def forward(self,x,skip):
        x=F.interpolate(x,size=(skip.shape[-2],x.shape[-1]),mode="bilinear",align_corners=False)
        return self.block(torch.cat([x,skip],dim=1))

class AMPORA_DCCRN(nn.Module):
    def __init__(self,base=16,gru_hidden=64):
        super().__init__()
        self.enc1=ConvBlock(2,base)
        self.enc2=ConvBlock(base,base*2,stride=(2,1))
        self.enc3=ConvBlock(base*2,base*3,stride=(2,1))
        self.enc4=ConvBlock(base*3,base*4,stride=(2,1))
        self.gru=nn.GRU(base*4,gru_hidden,num_layers=1,batch_first=True,bidirectional=False)
        self.gru_projection=nn.Linear(gru_hidden,base*4)
        self.dec3=DecoderBlock(base*4,base*3,base*3)
        self.dec2=DecoderBlock(base*3,base*2,base*2)
        self.dec1=DecoderBlock(base*2,base,base)
        self.output=nn.Conv2d(base,2,1)
    def forward(self,x):
        e1=self.enc1(x); e2=self.enc2(e1); e3=self.enc3(e2); e4=self.enc4(e3)
        seq=e4.mean(dim=2).transpose(1,2)
        seq,_=self.gru(seq); seq=self.gru_projection(seq)
        e4=e4+seq.transpose(1,2).unsqueeze(2)
        d3=self.dec3(e4,e3); d2=self.dec2(d3,e2); d1=self.dec1(d2,e1)
        d1=F.interpolate(d1,size=(x.shape[-2],x.shape[-1]),mode="bilinear",align_corners=False)
        return torch.tanh(self.output(d1))
