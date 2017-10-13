"""
the back method of relu is also relu
"""
from .reverse_base import ReverseBase 
import torch
import torch.nn as nn
import torch.nn.functional as F
class Reverse_relu(ReverseBase):
    def __init__(self):
        pass
    def reverse(self,frame,arg):
       print "reverse relu"
       relu=nn.ReLU()
       return relu

