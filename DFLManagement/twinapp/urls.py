from unicodedata import name
from django.contrib import admin
from django.urls import path
from . import views
urlpatterns = [
    path('home', views.home, name = 'home'),
    path('adddevice', views.adddevice, name = 'adddevice'),
    path("APIs", views.APIs, name="APIs"),
    path("initial_model_trasfer", views.initial_model_trasfer, name="initial_model_trasfer"),
    path("SLNode", views.SLNode, name="SLNode"),
    path("prepareModel", views.prepareModel, name="prepareModel"),
    path("designmodel", views.designmodel, name="designmodel"),
    path("topology", views.topology, name="topology"),
    path('gnntopology', views.gnntopology, name = 'gnntopology'),
    path("start_SL", views.start_SL, name="start_SL"),
    path("start_weight_learning", views.start_weight_learning, name="start_weight_learning"),
    
    path("prediction", views.prediction, name="prediction")
]

