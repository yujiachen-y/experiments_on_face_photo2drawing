from django.urls import path

from . import views

app_name = 'datasetviewer'
urlpatterns = [
    # ex: /datasetviewer/
    path('', views.index, name='index'),
    # ex: /datasetviewer/1/original_dataset/
    path('<int:show_landmarks>/<dataset_name>/', views.overview, name='overview'),
    # ex: /datasetviewer/1/original_dataset/Xiang_Liu/
    path('<int:show_landmarks>/<dataset_name>/<people_name>/', views.detail, name='detail'),
    # ex: /datasetviewer/1/original_dataset/Xiang_Liu/C00001
    path('<int:show_landmarks>/<dataset_name>/<people_name>/<image_name>', views.view_image, name='view_image'),
]