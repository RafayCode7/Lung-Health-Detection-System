"""
URL configuration for registration project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from app2 import views

from django.conf.urls.static import static
from django.conf import settings

urlpatterns = [
    path('admin/', admin.site.urls),
    path('signup',views.SignupPage,name='signup'),
    path('login/',views.LoginPage,name='login'),
    path('home/',views.HomePage,name='home'),
    path('blog/',views.BlogPage,name='blog'),
    path('UserGuide/',views.UserGuidePage,name='userguide'),
    path('Health-Form/',views.HealthForm,name='health-form'),
    path('',views.LaunchPage,name='launch'),
     path('D-signup',views.D_SignupPage,name='d-signup'),
    path('P-signup',views.P_SignupPage,name='p-signup'),
    path('predictImage/',views.predict_pneumonia,name="predictImage"),
    path('predictLungCancer', views.predict_lung_cancer, name='predict_lung_cancer'),
    path('Doctors/',views.DoctorPage,name='doctors'),
    path('Pateint/',views.patients_list,name='patient'),
    path('DoctorPanel/',views.DoctorStartPage,name='DoctorPanel'),
    path('DoctorPanellogin/',views.DoctorLoginPage,name='DoctorPanelLogin'),
    
    path('PatientPanel/',views.PatientStartPage,name='PatientPanel'),
    path('PateintPanellogin/',views.PatientLoginPage,name='PatientPanelLogin'),
    
     
     path('result/', views.result, name='result'),
      # detection/urls.py


 

    

   
]
urlpatterns += static(settings.MEDIA_URL,document_root=settings.MEDIA_ROOT)




