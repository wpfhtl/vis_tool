"""
realize the max_unpool of a given max_pool layer
"""
from .reverse_base import ReverseBase 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.autograd import Variable
class unpool(nn.MaxUnpool2d):
    def __init__(self,index,kernel_size,stride,output_size):
        super(unpool,self).__init__(kernel_size,stride)
        self.index=index
        self.output_size=output_size
    def forward(self, input):
        print "forward upool"
        out=super(unpool,self).forward(input,self.index,torch.Size(self.output_size))
        return out 
class Reverse_max_pool2d(ReverseBase):
    def __init__(self):
        self.index=[]
    def reverse(self,frame,arg):
        print "revese max_pool2d"
        if frame.f_locals.has_key("self"):
            input=frame.f_locals["self"]
            flag=0
        if frame.f_locals.has_key("input"):
            input=frame.f_locals["input"]
            flag=0
        if flag==1:
            raise Exception("no input get") 
        kernel_size=frame.f_locals["kernel_size"]
        stride=frame.f_locals["stride"]
        padding=frame.f_locals["padding"]
        dilation=frame.f_locals["dilation"]
        ceil_mode=frame.f_locals["ceil_mode"]
        if frame.f_locals["return_indices"]==False:
            downsample=nn.MaxPool2d(kernel_size,
                                    stride,
                                    padding,
                                    dilation,
                                    return_indices=True,
                                    ceil_mode=ceil_mode)
            out,self.index=downsample(input)
        else:
            self.index=arg[1]
        return unpool(self.index,kernel_size,stride,tuple(input.size()))
        






        
       
        
