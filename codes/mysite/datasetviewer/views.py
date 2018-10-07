import os

from django.shortcuts import render

from . import config

# Create your views here.
def index(request):
    datasets = os.listdir(config.WC_datasets_dir)
    return render(request, 'datasetviewer/index.html', {'datasets': datasets})