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
class SaliencyMapVisualization(BaseVisualization):
    description = 'show saliencymap'
    def make_visualization(self,model,images,output_dir):
        classes=-1
        namelist=[]
        for i in range(0,1):
            rimage=Variable(images.data,requires_grad=True)
            output=model.forward(rimage)
            v,c=output[0].max(0)
            output[0][c].backward()
            classes=c.data.numpy()
            image=rimage.grad.data.abs()
            im_array = image[0][0].numpy()
            grads=image.numpy()
            for i in range(1,grads.shape[1]):
                im_array+=grads[0][i]
            im = pyplot.imshow(im_array,cmap='OrRd_r')
            pyplot.axis('off')
            pyplot.figure(figsize=(10,10))
            im.axes.get_xaxis().set_visible(False)
            im.axes.get_yaxis().set_visible(False)
            name="saliencymap"+str(classes)+".png"
            im.figure.savefig(output_dir+"/"+name,transparent=True,bbox_inches='tight',pad_inches=0)
            namelist.append((str(classes),name))
        return namelist
