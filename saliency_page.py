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

saliency_page= Blueprint('saliency_page', __name__,template_folder='templates')

@saliency_page.route('/profile',methods=['GET'])
def show():
    number=request.args.get('number')
    object_file= file(session['img_output_dir']+"/"+session['vis_name'],'rb')
    data=pickle.load(object_file)
    namelist=data.GetSaliencyMap(int(number))
    return render_template("saliency.html",namelist=namelist,rand=random.randint(0,100000),number=number)
@saliency_page.route('/outputs/<filename>')
def download_outputs(filename):
    return send_from_directory(session['img_output_dir'],filename)

