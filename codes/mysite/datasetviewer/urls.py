from django.urls import path

from . import views

app_name = 'datasetviewer'
urlpatterns = [
    # ex: /datasetviewer/
    path('', views.index, name='index'),
]