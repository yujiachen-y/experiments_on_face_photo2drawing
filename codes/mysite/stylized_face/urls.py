from django.urls import path

from . import views

app_name = 'stylized_face'
urlpatterns = [
    # ex: /stylized_face/
    path('', views.upload, name='index'),
    # ex: /stylized_face/p2c_overview/file_name.jpg/
    path('p2c_overview/<img_name>/', views.overview, name='p2c_overview'),
    # ex: /stylized_face/file_name.jpg
    path('<img_name>/', views.detail, name='detail')
]