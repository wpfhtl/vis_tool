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

feature_page= Blueprint('feature_page', __name__,template_folder='templates')

@feature_page.route('/profile',methods=['GET'])
def show():
    layer=request.args.get('layer')
    number=request.args.get('number')
    object_file= file(session['img_output_dir']+"/"+session['vis_name'],'rb')
    data=pickle.load(object_file)
    if(int(layer)>len(data.module_list)):
        return render_template("alert.html",msg="not implemeted!!!")
    x,y,z=data.backer(int(layer),int(number))
    print x,y,z
    return render_template("profile.html",namelist=(x,y,z),rand=random.randint(0,100000),layer=layer,number=number)
@feature_page.route('/outputs/<filename>')
def download_outputs(filename):
    return send_from_directory(session['img_output_dir'],filename)

