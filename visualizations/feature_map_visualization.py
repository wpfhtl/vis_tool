"""
this class realize the featuremap visualization
and the back mapping understanding of featuremap
ref:https://www.cs.nyu.edu/~fergus/papers/zeilerECCV2014.pdf
"""
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.autograd import Variable
from PIL import Image
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.mlab as mlab    
import matplotlib.pyplot as plt  
from allreverse import ReverseBase
from importlib import import_module
from PIL import Image
from .base import BaseVisualization
import inspect
import sys
reverse_attr = vars(import_module('allreverse'))
ALLMETHOD=[]
for x in reverse_attr:
    m = reverse_attr[x]
    if inspect.isclass(m) and issubclass(m,ReverseBase) and m is not ReverseBase:
        ALLMETHOD.append(m)
allmethods={}
for method in ALLMETHOD:
    vis =method()
    allmethods[vis.__class__.__name__]=vis
def save_png_RGB(tpic,schar,output_dir):
    plt.figure(figsize=(5,5))
    tpic=tpic.data.abs().numpy()
    pic=tpic[0][0]
    for i in range(1,tpic.shape[1]):
        pic=pic+tpic[0][i]
    plt.imshow(pic)
    nowname=schar
    plt.savefig(output_dir+"/"+nowname)
    plt.clf()
    plt.close()
def save_png(tpic,schar):
    pic=tpic.data.numpy()
    plt.imshow(pic[2])
    nowname="feature"+"_"+schar+".png"
    plt.savefig("./"+nowname)
    plt.clf()
    plt.close()
def save_feature(pic,schar,output_dir):
    plt.figure(figsize=(5,5))
    pic=pic.data.numpy()
    plt.imshow(pic)
    plt.savefig(output_dir+"/"+schar)
    plt.clf()
    plt.close()
def save_greyfeature(pic,schar,output_dir):
    preprocess = transforms.Compose([transforms.ToPILImage()])
    tpic=pic.data.unsqueeze(0)
    tpic=preprocess(tpic)
    tpic.save(output_dir+"/"+schar)
def save_figure(images,schar,output_dir):
    images=images.data.numpy()
    images=images[0]
    if images.shape[0]==3:
        images=np.transpose(images,[1,2,0])
    else:
        images=images[0]
    plt.figure(figsize=(5,5))
    plt.imshow(images)
    plt.savefig(output_dir+"/"+schar)
    plt.clf()
    plt.close()
def get_feature_max(allarg,choice):
    assert allarg.dim()==4
    arg=allarg[0][choice]
    array=arg.data.numpy()
    shape=arg.size()
    index=np.argmax(array)
    row=index/shape[1]
    col=index%shape[1]
    shape=allarg.size()
    newout=torch.zeros(shape[0],shape[1],shape[2],shape[3])
    value=arg.data[row][col]
    row=0
    for rowdata in arg.data:
        col=0
        for data in rowdata:
            if abs(data-value)<1e-3:
                newout[0][choice][row][col]=data
            col=col+1
        row=row+1
    return Variable(newout)
def get_feature(allarg,choice):
    arg=allarg[0][choice]
    shape=allarg.size()
    newout=torch.zeros(shape[0],shape[1],shape[2],shape[3])
    newout[0][choice]=arg.data
    return Variable(newout)
def get_max(arg):
    assert  arg.dim()==4
    array=arg.data.numpy()
    shape=array.sha2e
    index=np.argmax(array)
    position=[]
    num=shape[1]*shape[2]*shape[3]
    for i in range(0,4):
        dex=index/num
        index=index-dex*num
        if i<3:
            num=num/shape[i+1]
        position.append(dex)
    p=position
    newinput=torch.zeros(shape[0],shape[1],shape[2],shape[3])
    newinput[p[0]][p[1]][p[2]][p[3]]=arg.data[p[0]][p[1]][p[2]][p[3]]
    newinput=Variable(newinput)
    return newinput
all_type=["featuremaps","1Ddata"]
class transedata:
    def __init__(self,dtype,d,number):
        self.datatype=dtype
        self.data=d
        self.number=number
class pickle_data:
    def __init__(self,feature_list,module_list,images,out_dir):
        self.feature_list=feature_list
        self.module_list=module_list
        self.images=images
        self.out_dir=out_dir
    def backer(self,num,feature_num):
        data=[]
        if num>len(self.module_list):
            print "not implemented!!!"
        else:
            back_list=self.module_list[0:num]
            back_list.reverse()
            x=get_feature_max(self.feature_list[num-1][1],feature_num)
            y=get_feature(self.feature_list[num-1][1],feature_num)
            for model in back_list:
                x=model.forward(x)
            for model in back_list:
                y=model.forward(y)
            namex="feature_max"+str(num)+"_"+str(feature_num)+".png"
            namey="feature_back"+str(num)+"_"+str(feature_num)+".png"
            name_image="image.png"
            save_png_RGB(x,namex,self.out_dir)
            save_png_RGB(y,namey,self.out_dir)
            save_figure(self.images,name_image,self.out_dir)
            data.append(("feature max point back",namex))
            data.append(("feature back",namey))
            data.append(("input image",name_image))
        return data
class mytracer:
    def __init__(self):
        self.filter_list=[]
        self.module_list=[]
        self.feature_list=[]
        self.layer_list=[]
        self.corr=0
        self.out_dir=[]
        for obj in inspect.getmembers(F):
            self.filter_list.append(obj[0])
    def traceit(self,frame, event, arg):
            if event == 'return' and frame.f_back.f_code.co_name == 'forward':
                if frame.f_code.co_name in self.filter_list:
                    name=frame.f_code.co_name
                    if frame.f_code.co_name=="threshold":
                        name="relu"
                    self.layer_list.append(name)
                    self.feature_list.append((name,arg))
                    if allmethods.has_key('Reverse_'+name) and self.corr==0:
                        model=allmethods['Reverse_'+name].reverse(frame,arg)
                        self.module_list.append(model)
                    else:
                        print "broken in this layer"
                        self.corr=1
            return self.traceit
    def storefeatures(self,output_dir):
        number=0
        namelist=[]
        for featuredata in self.feature_list:
            if featuredata[1].dim()==4:
                layer_feature_namelist=[]
                innernum=0
                for feature in featuredata[1][0]:
                    feature_name=featuredata[0]+"_"+str(number)+"_"+str(innernum)+".png"
                    save_greyfeature(feature,feature_name,output_dir)
                    layer_feature_namelist.append((innernum,feature_name))
                    innernum+=1
                number+=1
                data=transedata(all_type[0],(featuredata[0],layer_feature_namelist),number)
                namelist.append(data)
        return namelist
def getfeaturemap(model,images,output_dir):
    tc=mytracer()
    tc.out_dir=output_dir
    model.eval()
    sys.settrace(tc.traceit)
    model(images)
    return tc.storefeatures(output_dir),pickle_data(tc.feature_list,tc.module_list,images,output_dir)
class FeatureMapVisualization(BaseVisualization):
    description = 'show featuremaps!'
    def make_visualization(self,model,image,output_dir):
        namelist,self.data=getfeaturemap(model,image,output_dir)
        return namelist
