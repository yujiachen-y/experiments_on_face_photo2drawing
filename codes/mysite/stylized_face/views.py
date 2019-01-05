import os
import time

import numpy as np
from django.http import HttpResponse, HttpResponseRedirect
from django.shortcuts import render, render_to_response, reverse
from PIL import Image

from .stylize import alignment, stylize

path = os.path.join('support_material', 'stylizer')


def IsValidImage(img_path):
    bValid = True
    try:
        Image.open(img_path).verify()
    except:
        bValid = False
    return bValid


# Create your views here.
def handle_upload_file(file, filename):
    if not os.path.exists(path):
        os.makedirs(path)
    file_path = os.path.join(path, filename)
    with open(file_path,'wb') as destination:
        for chunk in file.chunks():
            destination.write(chunk)
    if not IsValidImage(file_path):
        return None
    file_path_split = os.path.splitext(file_path)
    if file_path_split[1] != '.jpg':
        img = Image.open(file_path)
        file_path = file_path_split[0] + '.jpg'
        img.save(file_path)
    return file_path


def upload(request):
    if request.method=="POST":
        filenames = os.path.splitext(str(request.FILES['file']))
        filename = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())+str(np.random.randint(800820))+filenames[1]
        raw_file_path = handle_upload_file(request.FILES['file'], filename)
        if raw_file_path is None:
            return render_to_response('stylized_face/index.html')
        align_file_path = alignment(raw_file_path)
        align_file_name = os.path.split(align_file_path)[1]
        return HttpResponseRedirect(reverse('stylized_face:p2c_overview', args=(align_file_name,)))
 
    return render_to_response('stylized_face/index.html')


def overview(request, img_name):
    seed = np.random.randint(800820)
    img_path = os.path.join(path, img_name)
    sty_path = stylize(img_path, seed)
    sty_name = os.path.split(sty_path)[1]
    return render(request, 'stylized_face/p2c_overview.html', {
        'img_name': img_name,
        'sty_name': sty_name,
    })


def detail(request, img_name):
    img_path = os.path.join(path, img_name)
    image = open(img_path, 'rb').read()
    return HttpResponse(image, content_type="image/jpg")
