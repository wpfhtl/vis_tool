#!usr/bin/env python
#-*-coding:utf-8-*-
from .base import BaseVisualization
import numpy as np    
import matplotlib.mlab as mlab    
import matplotlib.pyplot as plt  
#继承至BaseVislization类
class NiceVis(BaseVisualization):
    description = 'show nice vis!'
    def make_visualization(self,model,image,output_dir):
        #跑一次模型
        output=model(image)
        size=output.size()[-1]
        output=output.data.numpy()
        x=range(0,size)
        y=output[0]
        #将输出画成柱状图
        plt.bar(x,y,0.4,color="green")
        plt.show()
        name="nicevis.png"
        #将图片储存到指定目录
        plt.savefig(output_dir+"/"+"nicevis.png")
        plt.clf()
        #将图片名字保存到namelist中，以便传递到html文件
        namelist=name
        return namelist
