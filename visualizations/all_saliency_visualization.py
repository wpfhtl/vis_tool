"""
this visualization class realize the SaliencyMap
ref:https://arxiv.org/pdf/1312.6034.pdf
"""
from .base import BaseVisualization
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.autograd import Variable
from PIL import Image
import numpy as np
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.mlab as mlab    
import matplotlib.pyplot as plt  
from matplotlib import pyplot
class vis_data:
    def __init__(self,image,model,out_dir):
        self.image=image
        self.model=model
        self.output_dir=out_dir
    def GetSaliencyMap(self,number):
        image=Variable(self.image.data,requires_grad=True)
        output=self.model.forward(image)
        output[0][number].backward()
        grad_image=image.grad.data.abs().numpy()
        im_array=grad_image[0][0]
        dim=grad_image.shape[1]
        for i in range(1,dim):
            im_array+=grads_image[0][i]
        im = pyplot.imshow(im_array,cmap='OrRd_r')
        pyplot.axis('off')
        pyplot.figure(figsize=(10,10))
        im.axes.get_xaxis().set_visible(False)
        im.axes.get_yaxis().set_visible(False)
        name="saliencymap_"+str(number)+".png"
        im.figure.savefig(self.output_dir+"/"+name,transparent=True,bbox_inches='tight',pad_inches=0)
        namelist=name
        return namelist
class AllSaliencyMapVisualization(BaseVisualization):
    description = 'show all saliencymap'
    def make_visualization(self,model,image,output_dir):
        output=model(image)
        output=output.data.numpy()
        namelist=[]
        number=0
        print output.shape
        for out in output[0]:
            namelist.append((number,out))
            number=number+1
        self.data=vis_data(image,model,output_dir)
        return namelist
            

