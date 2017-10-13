#!usr/bin/env python
#-*-coding:utf-8-*-

"""
    author:zhangqian
    this module is a flask blueprint
--------------------------------
    this is responsible for basic server logic,
    including allocating resources for new session,
    generating homepage
    processing visualization method choosing
"""

import os
import sys
import io
import time
import inspect
import shutil
from operator import itemgetter
from tempfile import mkdtemp
from importlib import import_module
from types import ModuleType
from PIL import Image
from flask import Flask
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
from visualizations import BaseVisualization
import threading
import time
visualization_attr = vars(import_module('visualizations'))
visualization_submodules = []
VISUALIZATION_CLASSES = []
TIMERLIST={}
RANDY=0
for x in visualization_attr:
    m = visualization_attr[x]
    if inspect.isclass(m) and issubclass(m, BaseVisualization) and m is not BaseVisualization:
        VISUALIZATION_CLASSES.append(m)
def get_app_state():
    app_state = {
        'app_title':"pytorch viewer",
        'backend':"pytorch",
    }
    return app_state
def get_visualizations():
    if not hasattr(g, 'visualizations'):
        g.visualizations = {}
        for VisClass in VISUALIZATION_CLASSES:
            vis = VisClass()
            g.visualizations[vis.__class__.__name__] = vis
    return g.visualizations


app=Blueprint('app', __name__,template_folder='templates')

@app.route("/site")
@app.route('/', methods=['GET'])
def landing():
    visualizations = get_visualizations()
    vis_desc = [{'name': vis,'description': visualizations[vis].description}
    for vis in visualizations]
    return render_template('select_visualization.html',
                           app_state=get_app_state(),
                           visualizations=sorted(vis_desc,key=itemgetter('name'))
                            )
@app.route('/process', methods=['POST'])
def process():
    session['vis_name'] = request.form.get('choice')
    vis = get_visualizations()[session['vis_name']]
    if "model" not in session:
        return render_template('alert.html',msg="please upload a model file !")
    if "image" not in session:
        return render_template('alert.html',msg="please upload a input image !")
    if "model_pth" not in session:
        return render_template('alert.html',msg="please upload a pth!")
    try:
        path=session['img_input_dir']+"/"+session["model"]
        sys.path.append(os.path.dirname(path))
        model_file = __import__(session["model"][:-3])
    except Exception as err:
        return render_template('alert.html',msg="model file is wrong!")
    model_file=__import__(session["model"][:-3])
    path=os.path.join(session['img_input_dir'],session["image"])
    try:
        Image.open(path)
    except Exception as err:
        return render_template('alert.html',msg="image file is wrong!")
    rowimage=Image.open(path)
    try:
        model,image=model_file.get_model_and_input(rowimage)
    except Exception as err:
        return render_template('alert.html',msg="no function get_model_and_input(image) or the function is wrong!")
    model,image=model_file.get_model_and_input(rowimage)
    output=vis.make_visualization(model,image,session['img_output_dir'])
    permenant_vis=file(session['img_output_dir']+"/"+session['vis_name'],'wb')
    pickler = pickle.Pickler(permenant_vis)  #把文件关联到pickle
    pickler.dump(vis.data)  #把数据存到文件
    return render_template('{}.html'.format(session['vis_name']),model=session["model"],
                               image=session["image"],
                               namelist=output,
                               current_vis=session['vis_name']
                               ,rand=random.randint(0,10000000))
@app.route('/inputs/<filename>')
def download_inputs(filename):
    return send_from_directory(session['img_input_dir'],
                               filename)
@app.route('/outputs/<filename>')
def download_outputs(filename):
    return send_from_directory(session['img_output_dir'],
                               filename)
@app.route('/select_model_pth', methods=['POST'])
def save_model_pth():
    if 'file[]' in request.files:
        inputs = []
        for file_obj in request.files.getlist('file[]'):
            session["model_pth"]=file_obj.filename
            filepy = open(os.path.join(session['img_input_dir'],file_obj.filename), "wb");
            try:
                filepy.write( io.BytesIO(file_obj.stream.getvalue()).getvalue())
            except AttributeError:
                filepy.write(io.BytesIO(file_obj.stream.read()).getvalue());
            except Exception as err:
                return render_template('alert.html',msg="wrong model pth file")
        return""
    else:
        return render_template('alert.html',msg="no model pth file selected")
@app.route('/select_model', methods=['POST'])
def save_model():
    if 'file[]' in request.files:
        inputs = []
        for file_obj in request.files.getlist('file[]'):
            session["model"]=file_obj.filename
            filepy = open(os.path.join(session['img_input_dir'],file_obj.filename), "wb");
            try:
                filepy.write( io.BytesIO(file_obj.stream.getvalue()).getvalue())
            except AttributeError:
                filepy.write(io.BytesIO(file_obj.stream.read()).getvalue());
            except Exception as err:
                return render_template('alert.html',msg='model file wrong!') 
            return""
    else:
        return render_template('alert.html',msg='no model file selected !')
@app.route('/select_image', methods=['POST'])
def save_image():
    if 'file[]' in request.files:
        inputs = []
        for file_obj in request.files.getlist('file[]'):
            session["image"]=file_obj.filename
            try:
                Image.open(io.BytesIO(file_obj.stream.getvalue())).save(os.path.join(session['img_input_dir'],file_obj.filename))
            except AttributeError:
                Image.open(io.BytesIO(file_obj.stream.read())).save(os.path.join(session['img_input_dir'],file_obj.filename))
            except Exception as err:
                return render_template('alert.html',msg='wrong image file')
            return""
    else:
        return render_template('alert.html',msg='no image file selected')
@app.before_request
def initialize_new_session():
    if 'set' not in session:
#session.permanent = True
        session['set']=' '
    if 'image_uid_counter' in session and 'image_list' in session:
        pass #app.logger.debug('images are already being tracked')
    else:
        session['image_uid_counter'] = 0
        session['image_list'] = []
    if 'img_input_dir' in session and 'img_output_dir' in session:
        pass #app.logger.debug('temporary image directories already exist')
    else:
        session['img_input_dir'] = mkdtemp(dir="./static")
        print session['img_input_dir']
        session['img_output_dir'] = mkdtemp(dir="./static")
    if 'uid' not in session:
        session['uid']=session['img_input_dir']
        timer=threading.Timer(300000,end_session,(session['uid'],session['img_input_dir'],session['img_output_dir']))
        TIMERLIST[session['uid']]=timer
        timer.start()
@app.route('/set_timer',methods=['GET'])
def  set_timer():
    t=TIMERLIST[session['uid']]
    t.cancel()
    t.join()
    t=threading.Timer(300000,end_session,(session['uid'],session['img_input_dir'],session['img_output_dir']))
    TIMERLIST[session['uid']]=t
    t.start()
    return "ok"
@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),'favicon.ico', mimetype='image/vnd.microsoft.icon')
def end_session(uid,inputd,outputd):
    del TIMERLIST[uid]
    shutil.rmtree(inputd)
    shutil.rmtree(outputd)
@app.errorhandler(500)
def internal_server_error(e):
    return "500"#render_template('500.html', app_state=get_app_state()), 500
@app.errorhandler(404)
def not_found_error(e):
    return "404"#,request.url #render_template('404.html', app_state=get_app_state()), 404
