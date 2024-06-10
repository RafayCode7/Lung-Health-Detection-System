
from ast import Load
from multiprocessing import context
import os
from turtle import pd
from django.shortcuts import render ,HttpResponse, redirect
import keras
from matplotlib import testing
from pandas import read_csv
from requests import request
from django.contrib.auth.models import User
from django.contrib.auth import authenticate,login
from django.core.files.storage import FileSystemStorage 
from django.conf import settings  # Import Django settings module
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# Create your views here.
def HomePage(request):
    return render(request,'start.html')

def SignupPage(request):
    if request.method=='POST':
        uname=request.POST.get('username')
        email=request.POST.get('email')
        password1=request.POST.get('password1')
        password2=request.POST.get('password2')
        
        if password1!=password2:
            return HttpResponse('Your password is not same')
        else:                    
            my_user=User.objects.create_user(uname,email,password1)
            my_user.save()
            return redirect('login')
    
    return render(request,'signup.html')

def LoginPage(request):
    if request.method=='POST':
        username=request.POST.get('username')
        pass1=request.POST.get('pass')
        user=authenticate(request,username=username,password=pass1)
        if user is not None:
            login(request,user)
            return redirect('home')
        else:
            return HttpResponse('Incorrect username or password!!')
        
    return render(request,'login.html')
def DoctorLoginPage(request):
    if request.method=='POST':
        username=request.POST.get('username')
        pass1=request.POST.get('pass')
        user=authenticate(request,username=username,password=pass1)
        if user is not None:
            login(request,user)
            return redirect('DoctorPanel')
        else:
            return HttpResponse('Incorrect username or password!!')
        
        
    return render(request,'DoctorLogin.html')
def PatientStartPage(request):
    return render(request,"PateintPanel.html")

def PatientLoginPage(request):
    if request.method=='POST':
        username=request.POST.get('username')
        pass1=request.POST.get('pass')
        user=authenticate(request,username=username,password=pass1)
        if user is not None:
            login(request,user)
            return redirect('PatientPanel')
        else:
            return HttpResponse('Incorrect username or password!!')
        
        
    return render(request,'PatientLogin.html')

def BlogPage(request):
    return render(request,'blog-single.html')

def UserGuidePage(request):
    return render(request,'user-guide.html')

def HealthForm(request):
    return render(request,'Health-form.html')

def LaunchPage(request):
    return render(request,"launch.html")

def DoctorStartPage(request):
    return render(request,"DoctorStartPage.html")
    
from django.shortcuts import render
from .models import Doctor

def DoctorPage(request):
    doctors = Doctor.objects.all()  # Retrieve all doctor objects from the database
    return render(request, 'doctors.html', {'doctors': doctors})
from app2.models import Doctor  # Import the Doctor model
from app2.models import Patient

from django.shortcuts import render
from .models import Patient

from django.shortcuts import render
from .models import Patient

def patients_list(request):
    # Fetch all patients from the database
    patients = Patient.objects.all()
    # Pass the patients queryset to the template
    return render(request, 'pateint.html', {'patients': patients})

def D_SignupPage(request):
    if request.method == 'POST':
        uname = request.POST.get('username')
        email = request.POST.get('email')
        password1 = request.POST.get('password1')
        password2 = request.POST.get('password2')
        specialty = request.POST.get('specialty')  # Additional field for doctors
        experience_years = request.POST.get('experience_years')  # Additional field for doctors
        
        if not (uname and email and password1 and password2 and specialty and experience_years):
            return HttpResponse('All fields are required.')
        
        if len(password1) < 8:
            return HttpResponse('Password must be at least 8 characters long.')

        
        if password1 != password2:
            return HttpResponse('Your password is not the same')
        else:
            # Create a new user
            my_user = User.objects.create_user(uname, email, password1)
            
            # Create a new doctor
            doctor = Doctor.objects.create(user=my_user, specialty=specialty, experience_years=experience_years)
            
            return redirect('DoctorPanelLogin')
    
    return render(request, 'doctor_signup.html')




def P_SignupPage(request):
    if request.method == 'POST':
        uname = request.POST.get('username')
        email = request.POST.get('email')
        password1 = request.POST.get('password1')
        password2 = request.POST.get('password2')
        age = request.POST.get('age')  # Additional field for patients
        gender = request.POST.get('gender')  # Additional field for patients
        
        if password1 != password2:
            return HttpResponse('Your password is not the same')
        else:
            # Create a new user
            my_user = User.objects.create_user(uname, email, password1)
            
            # Create a new patient
            patient = Patient.objects.create(user=my_user, age=age, gender=gender)
            
            return redirect('PatientPanelLogin')
    
    return render(request, 'patient_signup.html')
#  Pnemonia Detection
from django.shortcuts import render
from keras.models import load_model # type: ignore
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from keras.preprocessing import image # type: ignore
from keras.applications.vgg16 import preprocess_input # type: ignore
import numpy as np

# Load the trained model
model_path = os.path.join(settings.BASE_DIR, 'savedModel/chest_xray.h5')
try:
    model = load_model(model_path)
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

# Define the image dimensions expected by the model
img_height, img_width = 224, 224

def predict_pneumonia(request):
    if request.method == 'POST' and request.FILES.get('image'):
        # Save the uploaded image to the media directory
        uploaded_image = request.FILES['image']
        fs = FileSystemStorage()
        filename = fs.save(uploaded_image.name, uploaded_image)
        uploaded_image_path = fs.url(filename)

        try:
            # Preprocess the uploaded image using Keras preprocessing
            img_path = os.path.join(settings.MEDIA_ROOT, filename)
            img = image.load_img(img_path, target_size=(img_height, img_width))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            img_data = preprocess_input(x)

            if model is not None:
                # Make prediction
                prediction = model.predict(img_data)

                # Print the prediction
                prediction=np.min(prediction)
                print(f"Prediction array: {prediction}")
                

                # Determine the predicted label based on the prediction array value
                predicted_label = 'Pneumonia' if prediction >= 0.0050 else 'Normal'

                return render(request, 'P_detect.html', {
                    'predicted_label': predicted_label,
                    'uploaded_image_path': uploaded_image_path
                })
            else:
                return render(request, 'P_detect.html', {
                    'error': 'Model is not loaded properly.'
                })
        except Exception as e:
            print(f"Error processing image: {e}")
            return render(request, 'P_detect.html', {
                'error': 'Error processing image.'
            })
    else:
        return render(request, 'P_detect.html')



  
#   Lung Cancer detection 
from django.shortcuts import render
from tensorflow.keras.models import load_model # type: ignore
from django.conf import settings
from django.core.files.storage import FileSystemStorage
import os
import numpy as np
import cv2

# Load the trained model
model_path = os.path.join(settings.BASE_DIR, 'savedModel/lung_cancer_detection_model3.h5')
try:
    model = load_model(model_path)
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

# Define the image dimensions expected by the model
img_height, img_width = 256, 256

def predict_lung_cancer(request):
    if request.method == 'POST' and request.FILES.get('image'):
        # Save the uploaded image to the media directory
        uploaded_image = request.FILES['image']
        fs = FileSystemStorage()
        filename = fs.save(uploaded_image.name, uploaded_image)
        uploaded_image_path = fs.url(filename)

        try:
            # Preprocess the uploaded image using OpenCV
            img_path = os.path.join(settings.MEDIA_ROOT, filename)
            img = cv2.imread(img_path)

            if img is None:
                raise ValueError(f"Error loading image at {img_path}")

            # Resize image
            img = cv2.resize(img, (img_height, img_width))

            # Normalize image
            img = img / 255.0

            # Expand dimensions to create a batch of size 1
            img = np.expand_dims(img, axis=0)

            if model is not None:
                # Make prediction
                prediction = model.predict(img)
                
                # Print the prediction result
                print("Prediction:", prediction)
                
                class_label = np.max(prediction)
                print("Class Label:", class_label)

                # Determine the category based on the class_label value
                if class_label>= 0.34:
                    category = 'Malignant'
                elif class_label > 0.339:
                    category = 'Benign'
                elif class_label > 0.33:
                    category = 'Normal'
                

                print("Category:", category)

                # Render prediction result along with form
                return render(request, 'lungCancerDetect.html', {
                    'predicted_label': category,
                    'uploaded_image_path': uploaded_image_path
                })
            else:
                return render(request, 'lungCancerDetect.html', {
                    'error': 'Model is not loaded properly.'
                })
        except Exception as e:
            print(f"Error processing image: {e}")
            return render(request, 'lungCancerDetect.html', {
                'error': 'Error processing image.'
            })
    else:
        return render(request, 'lungCancerDetect.html')

 
# #  Lung Cancer Module_2
# def result(request):
    
#     data = read_csv(r"C:\Users\smabb\survey lung cancer.csv")
#     # Encode categorical variables
#     data['GENDER'] = data['GENDER'].map({'M': 1, 'F': 0})
#     data = data.replace({'YES': 1, 'NO': 0})

#     # Split features and target variable
#     x = data.drop('LUNG_CANCER', axis=1)
#     y = data['LUNG_CANCER']

#     # Split the dataset into training and testing sets
#     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#     # Initialize and train the logistic regression model
#     lr = LogisticRegression()
#     lr.fit(x_train, y_train)

#     # Get input values from the form
#     val1 = float(request.GET.get('n1', 0))
#     val2 = float(request.GET.get('n2', 1))  # Assuming 1 as default value for gender
#     val3 = float(request.GET.get('n6', 0))
#     val4 = float(request.GET.get('n7', 0))
#     val5 = float(request.GET.get('n8', 0))
#     val6 = float(request.GET.get('n9', 0))
#     val7 = float(request.GET.get('n10', 0))
#     val8 = float(request.GET.get('n11', 0))
#     val9 = float(request.GET.get('n12', 0))
#     val10 = float(request.GET.get('n13', 0))
#     val11 = float(request.GET.get('n14', 0))
#     val12 = float(request.GET.get('n15', 0))
#     val13 = float(request.GET.get('n16', 0))
#     val14 = float(request.GET.get('n17', 0))
#     val15 = float(request.GET.get('n18', 0))

#     # Make prediction using the logistic regression model
#     pred = lr.predict([[val1, val2, val3, val4, val5, val6, val7, val8, val9, val10, val11, val12, val13, val14, val15]])

#     # Interpret prediction
#     result = "Positive" if pred[0] == 1 else "Negative"

#     # Render the result in the HTML template
#     return render(request, "Health-form.html", {"result2": result})
# detection/views.py
from django.shortcuts import render
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from .models import Symptom

def result(request):
    data = read_csv(r"C:\Users\smabb\survey lung cancer.csv")
    # Encode categorical variables
    data['GENDER'] = data['GENDER'].map({'M': 1, 'F': 0})
    data = data.replace({'YES': 1, 'NO': 0})

    # Split features and target variable
    x = data.drop('LUNG_CANCER', axis=1)
    y = data['LUNG_CANCER']

    # Split the dataset into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Initialize and train the logistic regression model
    lr = LogisticRegression()
    lr.fit(x_train, y_train)

    # Get input values from the form
    val1 = int(request.GET.get('n1', 0))
    val2 = int(request.GET.get('n2', 1))  # Assuming 1 as default value for gender
    val3 = int(request.GET.get('n6', 0))
    val4 = int(request.GET.get('n7', 0))
    val5 = int(request.GET.get('n8', 0))
    val6 = int(request.GET.get('n9', 0))
    val7 = int(request.GET.get('n10', 0))
    val8 = int(request.GET.get('n11', 0))
    val9 = int(request.GET.get('n12', 0))
    val10 = int(request.GET.get('n13', 0))
    val11 = int(request.GET.get('n14', 0))
    val12 = int(request.GET.get('n15', 0))
    val13 = int(request.GET.get('n16', 0))
    val14 = int(request.GET.get('n17', 0))
    val15 = int(request.GET.get('n18', 0))

    # Store symptoms in the database
    symptom_entry = Symptom(
        age=val1, gender=val2, smoking=val3, yellow_fingers=val4, anxiety=val5,
        peer_pressure=val6, chronic_disease=val7, fatigue=val8, allergy=val9,
        wheezing=val10, alcohol_consumption=val11, coughing=val12,
        shortness_of_breath=val13, swallowing_difficulty=val14, chest_pain=val15
    )
    symptom_entry.save()

    # Make prediction using the logistic regression model
    pred = lr.predict([[val1, val2, val3, val4, val5, val6, val7, val8, val9, val10, val11, val12, val13, val14, val15]])

    # Interpret prediction
    result = "Positive" if pred[0] == 1 else "Negative"

    # Render the result in the HTML template
    return render(request, "Health-form.html", {"result2": result})

