"""
realize the deconv of a given conv2d layer
"""
from .reverse_base import ReverseBase 
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
class deconv(nn.ConvTranspose2d):
    def __init__(self,insize,outsize,kernel_size,stride,padding,output_padding,weight,bias):
        assert output_padding>=0
        super(deconv,self).__init__(insize,outsize,kernel_size=kernel_size,stride=stride,padding=padding,output_padding=output_padding,bias=False)
        self.weight=weight
    def forward(self,input):
        print "forward deconv"
        #print input.size(),self.bb.size()
        out=super(deconv,self).forward(input)
        return out
class Reverse_conv2d(ReverseBase):
    def reverse(self,frame,arg):
        print "reverse conv2d"
        weight=frame.f_locals["weight"]
        bias=frame.f_locals["bias"]
        dim1=weight.size()[2]
        dim2=weight.size()[3]
        dim=(dim1,dim2)
        bias = frame.f_locals["bias"]
        input=0
        flag=1
        if frame.f_locals.has_key("self"):
            input=frame.f_locals["self"]
            flag=0
        if frame.f_locals.has_key("input"):
            input=frame.f_locals["input"]
            flag=0
        stride=frame.f_locals["stride"]
        padding=frame.f_locals["padding"]        
        if flag==1:
            raise Exception("no input get!")
        outdim=(arg.size()[2]-1)*stride[0]-2*padding[0]+dim[0]
        output_padding=input.size()[2]-outdim
        #bias=Variable(torch.zeros(input.size()[1]))
        downsample=deconv(arg.size()[1],input.size()[1],dim,stride,padding,output_padding,weight,bias)
        return downsample
