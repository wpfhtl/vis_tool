# Pytorch Viewer(模型可视化工具)

实现以下可视化

1.saliency map. ref:https://arxiv.org/pdf/1312.6034.pdf

2.feature map.

3.feature map back mapping. ref:https://www.cs.nyu.edu/~fergus/papers/zeilerECCV2014.pdf

##  依赖项
    1.python2.7
    2.flask
    3.pytorch
    4.MatplotLib

## 使用方法（以lenet举例）
    1.在服务端启动服务器运行main.py
    2.在lenet.py文件中额外定义一个名字为get_model_and_input(image)的函数,image 为读入的图片
    函数负责建立模型，并且将图片处理成需要的Variable
```python
    def get_model_and_input(image):
        pth_name = "lenet.pth"
        pth_file = os.path.split(os.path.abspath(__file__))[0] +'/'+ pth_name
        model = LeNet()
        model.load_state_dict(torch.load(pth_file,map_location=lambda storage,loc: storage))
        preprocess = transforms.Compose([
                           transforms.ToTensor()])
        image=image.resize((28,28))
        im=preprocess(image)[0]
        im.unsqueeze_(0)
        im.unsqueeze_(0)
        im=Variable(im)
        return model,im
```

    3.进入主页上传lenet.py,lenet.pth以及图片文件
    
<img src="https://github.com/int2char/vis_tool/blob/master/images/step1.png?raw=true" width = "800" height = "400" alt="step1" align=center />

    4.选择你希望的可视化方法,点击go按钮,你将得到可视化结果
    
<img src="https://github.com/int2char/vis_tool/blob/master/images/step2.png?raw=true" width = "800" height = "400" alt="atep2" align=center />

<img src="https://github.com/int2char/vis_tool/blob/master/images/step3.png?raw=true" width = "800" height = "400" alt="atep2" align=center />

## 可视化方法介绍

    1.saliency map可视化将显示当前最大分类值对输入的导数，导数值越大点越亮
    
<img src="https://github.com/int2char/vis_tool/blob/master/images/step4.png?raw=true" width = "800" height = "400" alt="step1" align=center />

    2.feature map可视化将显示模型各个层的feature map灰度图
    
<img src="https://github.com/int2char/vis_tool/blob/master/images/step5.png?raw=true" width = "800" height = "400" alt="step1" align=center />

    3.点击指定feature map的profile 将会使得当前feature map反向到输入，最终得到可视化图。
    如下，第一张图是以feature map最大值点（其他点设置为零）作为反向的输入得到的特征可视化图  
    第二张图是以当前feature map为反响的输入得到的特征可视化图
    
<img src="https://github.com/int2char/vis_tool/blob/master/images/step6.png?raw=true" width = "800" height = "400" alt="step1" align=center />

## 扩展你自己的可视化方法

    1.假设你想添加名为NiceVis的可视化方法以显示模型输出的柱状图，在visualization目录下新建py文件nicevis.py
    新建类NiceVis继承BaseVisualization类，并重写make_visualization方法，实现自己的可视化逻辑，代码如下:

```python
#!usr/bin/env python
#-*-coding:utf-8-*-
from .base import BaseVisualization
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
#继承至BaseVisualization类
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
```

    其中，description是对方法的描述，这将显示到首页的选择栏目中，model是已经加载好的模型，image是模型的输入（Variable类型），output_dir是用户目录，
    每个用户有一个独立的目录，将你生成的图片放到这个目录中以便浏览器访问。
    每个方法都需要返回一个namelist，pytorch viewer将把namelist传递给你的HTML文件，这样就可以访问你的图片或者其他数据。
    上面代码完成一个简单的逻辑：先跑一次模型，把模型的输出画成柱状图并储存成图片文件，然后返回图片的名字。

    2.注册你的方法：如下，在visualization下的__init__.py添加你的类，以便pytorch viewer 识别你的方法

```python
from .base import BaseVisualization
from .feature_map_visualization import FeatureMapVisualization
from .saliency_map_visualization import SaliencyMapVisualization
#添加这一行，将NiceVis import 进来
from .nicevis import NiceVis
```
    3.在templates目录下编写你自己的HTML可视化格式，名字为NiceVis.html(与类名字一样)如下：

```html
{% extends "result.html" %}
{%block vis%}
 <div>
 <font size="30" style="font-weight:bold;float:left;margin-top:10px">bar of result:</font>
 <a href="outputs/{{namelist}}?a={{rand}}">
 <img  src="outputs/{{namelist}}?a={{rand}}" style="width:3000px;height:800px;"/>
 </a>
</div>
{%endblock%}

```

    4.重启服务器，运行你的方法

<img src="https://github.com/int2char/vis_tool/blob/master/images/step7.png?raw=true" width = "800" height = "400" alt="step1" align=center />

## 扩展反向方法

    现在只实现了论文 Visualizing and Understanding Convolutional Networks 中的反向方法，包括conv2d,relu,maxpool2d，如果模型中间出现其他层，那么模型将不能从更深层反向
    ，你可以自己实现相应层的反向方法，下面以relu为例:在allreverse目录下新建文件reverse_relu.py,创建类Reverse_relu继承至ReverseBase，重写方法reverse，此方法需要返回一个层对象，此对象实现relu的反向功能
    如下:
```python
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
```

    relu的反向层就是relu(具体见论文)，所以函数只是简单的返回relu.
    大部分反向层的构建需要依赖于正向层的参数，其中frame中含有正向层的所有参数，arg是正向层的输出（Variable）,你可以根据frame参数和arg来构建你需要的反向方法.
    如下是conv2d 反向层 deconv的构建过程，deconv的构建需要conv2d的大量参数才能构造出正确的反向方法。

```python
"""
realize the deconv of a given conv2d layer
"""
from .reverse_base import ReverseBase
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
class deconv(nn.ConvTranspose2d):
    def __init__(self,insize,outsize,kernel_size,stride,padding,output_padding,weight):
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
        #获得conv2d的weight
        weight=frame.f_locals["weight"]
        #获得conv2d卷积核的大小
        dim1=weight.size()[2]
        dim2=weight.size()[3]
        dim=(dim1,dim2)
        #获得原conv2d的input（Variable）
        input=frame.f_locals["input"]
        stride=frame.f_locals["stride"]
        padding=frame.f_locals["padding"]
        #计算conv_transpose后的输出大小
        outdim=(arg.size()[2]-1)*stride[0]-2*padding[0]+dim[0]
        #计算conv_transpose输出大小与原输入大小的差值，以便传入conv_transpose对output进行padding
        output_padding=input.size()[2]-outdim
        downsample=deconv(arg.size()[1],input.size()[1],dim,stride,padding,output_padding,weight)
        return downsample

```

## 添加新的视图逻辑

    如果你的可视化方法需要特定的与服务器的交互逻辑，你需要新建一个视图函数并在其中编写这些逻辑。
    比如你想编写一个页面显示所有的分类结果，并且当点击某一个分类时会显示此分类对应的saliency map

    1.首先在visualization 下添加py文件all_saliency_map.py,添加类继承至BaseVisualiztion,代码如下所示。
    由于不同视图是分离的，不能共享数据，所以如果你的视图逻辑需要用到model，image相关的数据，你需要在make_visualization中将数据存在硬盘，数据的结构由你自己定义。
    pytorch_viewer已经实现了储存部分，你只需要将你的数据赋值给self.data,那么这个data将存储为与你方法名相同的文件，比如这里文件名就是AllSaliencyMapVisualization

```python
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
#你根据自己的逻辑需求设计这个类，此类负责将make_visualization传递的数据记录下来
#此类将作为数据储存到硬盘，这样你的视图函数就可以访问这个类以及其中的数据
#你也可以在这个类中实现一些和数据相关的方法，以便你在视图中使用
class vis_data:
    def __init__(self,image,model,out_dir):
        #image和model需要记录下来，因为视图中要生成saliency map需要这两个数据
        self.image=image
        self.model=model
        self.output_dir=out_dir
    #此函数计算给定classes number 的saliency map结果
    #并将结果固化到输出目录，将结果的图片名字返回，以便html使用
    #此函数将在视图中被调用来产生saliency map
    def GetSaliencyMap(self,number):
        #将required_grad设置为True
        image=Variable(self.image.data,requires_grad=True)
        output=self.model.forward(image)
        #将指定class的输出反向求导，得到saliency map
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
        #生成saliency map的名字
        name="saliencymap_"+str(number)+".png"
        #储存saliency map到指定目录，此目录可以在你的视图中访问到
        im.figure.savefig(self.output_dir+"/"+name,transparent=True,bbox_inches='tight',pad_inches=0)
        namelist=name
        #返回namelist,最终会传递给html
        return namelist
class AllSaliencyMapVisualization(BaseVisualization):
    description = 'show all saliencymap'
    #此函数生成方法首页，运行前向，并且把namelist返回，最终将传递给方法首页的html
    def make_visualization(self,model,image,output_dir):
        output=model(image)
        output=output.data.numpy()
        namelist=[]
        number=0
        for out in output[0]:
            namelist.append((number,out))
            number=number+1
        #此处将你的视图中需要的数据复制给self.data,pytorch_viewer会将self.data固化到目录(文件名为是AllSaliencyMapVisualization)，以便你的视图函数使用
        self.data=vis_data(image,model,output_dir)
        return namelist
```

    2.注册你的可视化方法（在visualization/__init__.py 中import这个类AllSaliencyMapVisualization）

    3.编写视图函数，视图中show函数恢复之前固化的对象，调用方法生成saliency map，download_outputs函数将对图片的请求定位到
    原始的session['img_output_dir']目录，生成的所有图像都是储存在这里的。

```python
"""
    autor:zhangqian
    this is flask blueprint
    this is responsible for the back profile page
"""
from flask import Blueprint, render_template, abort
from flask import (
        g,
        Blueprint,
        render_template,
        request,
        session,
        send_from_directory,
        jsonify,
        flash
        )
import cPickle as pickle
import random
#新建视图函数
saliency_page= Blueprint('saliency_page', __name__,template_folder='templates')
#此函数得到请求后，生成saliency map
@saliency_page.route('/profile',methods=['GET'])
def show():
    #得到请求的class number
    number=request.args.get('number')
    #读取之前固化到硬盘的数据结构，得到一个vis_data的object
    object_file= file(session['img_output_dir']+"/"+session['vis_name'],'rb')
    data=pickle.load(object_file)
    #调用vis_data的GetSaliencyMap方法得到对应saliency map的name
    namelist=data.GetSaliencyMap(int(number))
    #返回saliency map 显示页面
    return render_template("saliency.html",namelist=namelist,rand=random.randint(0,100000),number=number)
#下载逻辑，浏览器对图片的请求对应到session['img_output_dir']目录
@saliency_page.route('/outputs/<filename>')
def download_outputs(filename):
    return send_from_directory(session['img_output_dir'],filename)

```

    4.注册你的视图到main.py中

```python
from flask import Flask ,session,url_for
from flask import app as bpp
from app_page import app
from feature_page import feature_page
from saliency_page import saliency_page
import os
main = Flask(__name__)
if main.debug:
    main.secret_key = '...' ##secret_key
else:
    main.secret_key = os.urandom(24)
"""
register your own blueprint here
"""
main.register_blueprint(app)
main.register_blueprint(feature_page,url_prefix='/feature_page')
#注册你的视图
main.register_blueprint(saliency_page,url_prefix='/saliency_page')
if __name__ == "__main__":
    main.run(host='0.0.0.0', debug=True, threaded=True)
```

    5.在templates下编写AllSaliencyMapVisualization 如下，该html主要用以显示方法的首页

```html
{%extends "result.html"%}
{%block vis%}
<table class="table table-sm table-striped">
    <caption><font size="5" style="font-weight:bold;float:left;margin-top:10px">all the classes value:</font></caption>
    <tbody>
           <tr>
         {% for name in namelist %}
          <td align="center">
                  <a href="{{url_for('saliency_page.show')}}?number={{name[0]}}" target="_blank">
                   <b>{{name[0]}}</b>
                  </a>
          </td>
         {%endfor%}
       </tr>
        <tr>
          {% for name in namelist %}
            <td align="center">
              <b>{{name[1]}}</b>
            </td>
          {% endfor %}
        </tr>

    </tbody>
</table>
{%endblock%}

```

    此时选择show all saliency map方法，点击go将出现下面页面

<img src="https://github.com/int2char/vis_tool/blob/master/images/step8.png?raw=true" width = "800" height = "400" alt="step1" align=center />

    6.在templates下新建saliency.html,这个html是视图中的show方法返回的html
```html
<h4><b>class {{number}} saliency map</b></h4>
<a href="outputs/{{namelist}}?a={{rand}}" target="_blank">
    <img src="outputs/{{namelist}}?a={{rand}}" style="float:left;width:244px;height:244px;"/>
</a>
```

    现在点击主页上的分类标签，你可以得到你想要分类的salency map图

<img src="https://github.com/int2char/vis_tool/blob/master/images/step9.png?raw=true" width = "600" height = "400" alt="step1" align=center />




