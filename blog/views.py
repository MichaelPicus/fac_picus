from __future__ import unicode_literals
from django.shortcuts import render
from django.shortcuts import render_to_response
from django.utils import timezone
from .models import Post
from django.template import RequestContext
from django.http import HttpResponseRedirect
from django.core.urlresolvers import reverse

from blog.models import Document
from blog.forms import DocumentForm

from sklearn.externals import joblib

import numpy as np 
import pandas as pd 

import lightgbm as lgb
from random import randint
import copy
import os, sys
import shutil


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from django.contrib.auth.forms import UserCreationForm
from django.views.generic.edit import CreateView

class SignUpView(CreateView):
    template_name = 'signup.html'
    form_class = UserCreationForm

from django.contrib.auth.models import User
from django.http import JsonResponse

def validate_username(request):
    username = request.GET.get('username', None)
    data = {
        'is_taken': User.objects.filter(username__iexact=username).exists()
    }
    if data['is_taken']:
        data['error_message'] = 'A user with this username already exists.'
    return JsonResponse(data)


from .forms import NameForm
from .forms import ContactForm
from .forms import Jingbai
from django.core.mail import send_mail


def get_name(request):
    # if this is a POST request we need to process the form data
    if request.method == 'POST':
        # create a form instance and populate it with data from the request:
        form = ContactForm(request.POST)
        # check whether it's valid:
        if form.is_valid():
            # process the data in form.cleaned_data as required
            # ...
            # redirect to a new URL:
            
            subject = form.cleaned_data['subject']
            message = form.cleaned_data['message']
            sender = form.cleaned_data['sender']
            cc_myself = form.cleaned_data['cc_myself']

            recipients = ['michaelchen@factorypicus.com']
            if cc_myself:
                recipients.append(sender)

            send_mail(subject, message, sender, recipients)
            return HttpResponseRedirect('/thanks/')


    # if a GET (or any other method) we'll create a blank form
    else:
        form = ContactForm()
    athlete_list = ['michael chen', 'john song', 'david li']
    return render(request, 'name.html', {'form': form, 'athlete_list': athlete_list})

def thanks(request):
    return render(request, 'thanks.html')

# display soluton for onsite tuning
def display(request):

    return render(request, 'display.html')

def process(request):
    model = joblib.load(os.path.join(BASE_DIR, 'ml_models/model_gboost_jingbai.pkl'))

    results = []
    predict = []
    m = []
    if request.method == 'POST':
        form = Jingbai(request.POST)

        if form.is_valid():
            airouttemp = float(form.cleaned_data['airouttemp'])
            basepowdertemp = float(form.cleaned_data['basepowdertemp'])
            airintemp_1 = float(form.cleaned_data['airintemp_1'])
            slurrytemp = float(form.cleaned_data['slurrytemp'])
            towertopnegativepressure = float(form.cleaned_data['towertopnegativepressure'])
            agingtankflow = float(form.cleaned_data['agingtankflow'])
            secondinputairtemp = float(form.cleaned_data['secondinputairtemp'])
            slurrypipelinelowerlayerpressure = float(form.cleaned_data['slurrypipelinelowerlayerpressure'])
            outairmotorfreq = float(form.cleaned_data['outairmotorfreq'])
            secondairmotorfreq = float(form.cleaned_data['secondairmotorfreq'])
            highpressurepumpfreq = float(form.cleaned_data['highpressurepumpfreq'])
            gasflow = float(form.cleaned_data['gasflow'])

            train = np.array([[airouttemp,basepowdertemp, airintemp_1, slurrytemp, towertopnegativepressure, agingtankflow, secondinputairtemp, slurrypipelinelowerlayerpressure, outairmotorfreq, secondairmotorfreq, highpressurepumpfreq, gasflow ]])
            train_pred = model.predict(train)
            train_pred = np.expm1(train_pred)
            m =  train_pred

            if train_pred > 30:

                for x in range(3):

                    # AirOutTemp
                    if train[0][0] > 130:
                        train[0][0] = 129.99
                    elif train[0][0] < 76:
                        train[0][0] = 76.001
                    else:
                        train[0][0] = train[0][0]  * 0.993
                    
                    # BasePowderTemp
                    if train[0][1] > 166:
                        train[0][1] = 165.99
                    elif train[0][1] < 95:
                        train[0][1] = 95.001
                    else:
                        train[0][1] = train[0][1]  * 0.997

                    # AirInTemp_1
                    if train[0][2] > 302 :
                        train[0][2] = 301.999
                    elif train[0][2] < 238:
                        train[0][2] = 238.001
                        
                    else:
                        train[0][2] = train[0][2]  * 1.0152


                    # SlurryTemp
                    if train[0][3] > 894:
                        train[0][3] = 893.99
                    elif train[0][3] < 0:
                        train[0][3] = 0.001
                    else:
                        train[0][3] = train[0][3] * 1.0226

                    # TowerTopNegativePressure
                    if train[0][4] > 0:
                        train[0][4] = -0.000001
                    elif train[0][4] < -30.0:
                        train[0][4] = -29.988
                    else: 
                        train[0][4] = train[0][4] * 0.9959
                    
                    # AgingTankFlow
                    if train[0][5] > 27474:
                        train[0][5] = 27473.999
                    elif train[0][5] < 17451:
                        train[0][5] = 17451.02
                    else:
                        train[0][5] = train[0][5] * 1.02226
                    
                    # SecondInputAirTemp
                    if train[0][6] >68:
                        train[0][6] = 67.99
                    elif train[0][6] < 0:
                        train[0][6] = -0.00001
                    else:
                        train[0][6] = train[0][6] * 1.00078
                    
                    # SlurryPipelineLowerLayerPressure
                    if train[0][7] > 76:
                        train[0][7] = 75.999
                    elif train[0][7] < 42:
                        train[0][7] = 42.0009
                    else:
                        train[0][7] = train[0][7] * 1.0095
                    
                    # OutAirMotorFreq
                    if train[0][8] > 0.9:
                        train[0][8] = 0.899999
                    elif train[0][8] < 0.6:
                        train[0][8] = 0.6001
                    else:
                        train[0][8] = train[0][8] * 0.98817
                    
                    # SecondAirMotorFreq
                    if train[0][9] > 88:
                        train[0][9] = 87.99
                    elif train[0][9] < 53:
                        train[0][9] = 53.001
                    else:

                        train[0][9] = train[0][9] * 0.9941167
                    # HighPressurePumpFreq
                    if train[0][10] > 37.6:
                        train[0][10] = 37.59
                    elif train[0][10] < 8.6:
                        train[0][10] = 8.699
                    else:

                        train[0][10] = train[0][10] * 1.018

                    # GasFlow
                    if train[0][11] > 722:
                        train[0][11] = 721.99
                    elif train[0][11] < 500:
                        train[0][11] = 500.001
                    else:
                        train[0][11] = train[0][11] * 0.99857
                    # # AirOutTemp
                    # train[0][0] = train[0][0]  * 0.98
                    # # BasePowderTemp
                    # train[0][1] = train[0][1]  * 0.99123
                    # # AirInTemp_1
                    # train[0][2] = train[0][2]  * 1.0456
                    # # SlurryTemp
                    # train[0][3] = train[0][3] * 1.0678

                    # # TowerTopNegativePressure
                    # train[0][4] = train[0][4] * 0.9877
                    # # AgingTankFlow
                    # train[0][5] = train[0][5] * 1.09678
                    # # SecondInputAirTemp
                    # train[0][6] = train[0][6] * 1.00234
                    # # SlurryPipelineLowerLayerPressure
                    # train[0][7] = train[0][7] * 1.0285
                    # # OutAirMotorFreq
                    # train[0][8] = train[0][8] * 0.9645
                    # # SecondAirMotorFreq
                    # train[0][9] = train[0][9] * 0.98235
                    # # HighPressurePumpFreq
                    # train[0][10] = train[0][10] * 1.054
                    # # GasFlow
                    # train[0][11] = train[0][11] * 0.98667

                    modified_m = np.expm1(model.predict(train))
                    
                    tmp = np.append(train, modified_m)
                    results.append(tmp)

            else:
                
                tmp = np.append(train, train_pred)
                results.append(tmp)


    else:
        form = Jingbai()

    return render(request, 'process.html', {'form': form, 'results': results, 'm': m})    


# Create your views here.
def post_list(request):
	posts = Post.objects.filter(published_date__lte=timezone.now()).order_by('published_date')
	return render(request, 'blog/post_list.html', {'posts': posts})
	
def jingbai(request):
    # Handle file upload
    if request.method == 'POST':
        form = DocumentForm(request.POST, request.FILES)
        if form.is_valid():
            newdoc = Document(docfile=request.FILES['docfile'])
            newdoc.save()

            # Redirect to the document list after POST
            return HttpResponseRedirect(reverse('jingbai'))
    else:
        form = DocumentForm()  # A empty, unbound form

    # Load documents for the list page
    documents = Document.objects.all()

    # Render list page with the documents and the form
    return render(
        request,
        'blog/jingbai.html',
        {'documents': documents, 'form': form}
    )

def jingbai_two_vars(request):
    # Handle file upload
    if request.method == 'POST':
        form = DocumentForm(request.POST, request.FILES)
        if form.is_valid():
            newdoc = Document(docfile=request.FILES['docfile'])
            newdoc.save()

            # Redirect to the document list after POST
            return HttpResponseRedirect(reverse('jingbai_two_vars'))
    else:
        form = DocumentForm()  # A empty, unbound form

    # Load documents for the list page
    documents = Document.objects.all()

    # Render list page with the documents and the form
    return render(
        request,
        'blog/jingbai_two_vars.html',
        {'documents': documents, 'form': form}
    )

def tbo(request):
    # Handle file upload
    if request.method == 'POST':
        form = DocumentForm(request.POST, request.FILES)
        if form.is_valid():
            newdoc = Document(docfile=request.FILES['docfile'])
            newdoc.save()

            # Redirect to the document list after POST
            return HttpResponseRedirect(reverse('tbo'))
    else:
        form = DocumentForm()  # A empty, unbound form

    # Load documents for the list page
    documents = Document.objects.all()

    # Render list page with the documents and the form
    return render(
        request,
        'blog/tbo.html',
        {'documents': documents, 'form': form}
    )

def tbo_two_vars(request):
    # Handle file upload
    if request.method == 'POST':
        form = DocumentForm(request.POST, request.FILES)
        if form.is_valid():
            newdoc = Document(docfile=request.FILES['docfile'])
            newdoc.save()

            # Redirect to the document list after POST
            return HttpResponseRedirect(reverse('tbo_two_vars'))
    else:
        form = DocumentForm()  # A empty, unbound form

    # Load documents for the list page
    documents = Document.objects.all()

    # Render list page with the documents and the form
    return render(
        request,
        'blog/tbo_two_vars.html',
        {'documents': documents, 'form': form}
    )

def bilang(request):
    # Handle file upload
    if request.method == 'POST':
        form = DocumentForm(request.POST, request.FILES)
        if form.is_valid():
            newdoc = Document(docfile=request.FILES['docfile'])
            newdoc.save()

            # Redirect to the document list after POST
            return HttpResponseRedirect(reverse('bilang'))
    else:
        form = DocumentForm()  # A empty, unbound form

    # Load documents for the list page
    documents = Document.objects.all()

    # Render list page with the documents and the form
    return render(
        request,
        'blog/bilang.html',
        {'documents': documents, 'form': form}
    )

def bilang_two_vars(request):
    # Handle file upload
    if request.method == 'POST':
        form = DocumentForm(request.POST, request.FILES)
        if form.is_valid():
            newdoc = Document(docfile=request.FILES['docfile'])
            newdoc.save()

            # Redirect to the document list after POST
            return HttpResponseRedirect(reverse('bilang_two_vars'))
    else:
        form = DocumentForm()  # A empty, unbound form

    # Load documents for the list page
    documents = Document.objects.all()

    # Render list page with the documents and the form
    return render(
        request,
        'blog/bilang_two_vars.html',
        {'documents': documents, 'form': form}
    )


def list(request):
    # Handle file upload
    if request.method == 'POST':
        form = DocumentForm(request.POST, request.FILES)
        if form.is_valid():
            newdoc = Document(docfile=request.FILES['docfile'])
            newdoc.save()

            # Redirect to the document list after POST
            return HttpResponseRedirect(reverse('list'))
    else:
        form = DocumentForm()  # A empty, unbound form

    # Load documents for the list page
    documents = Document.objects.all()

    # Render list page with the documents and the form
    return render(
        request,
        'blog/list.html',
        {'documents': documents, 'form': form}
    )

def jingbai_ds(request):
    lasso_model = joblib.load(os.path.join(BASE_DIR, 'ml_models/model_gboost_jingbai.pkl'))
    df_ready = pd.read_csv(os.path.join(BASE_DIR, 'media/documents/jingbai_ready.csv'))

    train_y = df_ready.M.values

    del df_ready["Unnamed: 0"]
    del df_ready["M"]
    train = df_ready.values
    train_pred = lasso_model.predict(train)

    combine = np.column_stack((np.expm1(train_pred), train))
    
    # need to delete the origin upload file
    os.remove(os.path.join(BASE_DIR, 'media/documents/jingbai_ready.csv'))

    rows = combine.shape[0]
    cols = combine.shape[1]
    modified_res = copy.deepcopy(combine)
    for x in range(0, rows):
        # count = 0
        # while combine[x, 0] > 32.9:
        #     if count == 3: 
        #         break
        #     count = count + 1
        #     for y in range(1, cols-1):
        #         modified_res[x, y] = combine[x, y] * (1 - randint(9700, 10000) / 100000.0)
        #     # modified_res[x, 0] = lgb_model.predict(np.reshape(modified_res[x][1:], (-1, 12)))
        #     modified_res[x, 0] = combine[x, 0] * (1 - randint(9800, 10000) / 100000.0)

        #     modified_res[x, cols -1] = combine[x, cols-1] * 0.9994448 *randint(980, 999) / 1000
        if combine[x, 0] > 33 :
            # AirOutTemp
            modified_res[x, 1] = combine[x, 1]  * 0.98
            # BasePowderTemp
            modified_res[x, 2] = combine[x, 2]  * 0.99123
            # AirInTemp_1
            modified_res[x, 3] = combine[x, 3]  * 1.0456
            # SlurryTemp
            modified_res[x, 4] = combine[x, 4]  * 1.0678

            # TowerTopNegativePressure
            modified_res[x, 5] = combine[x, 5] * 0.9877
            # AgingTankFlow
            modified_res[x, 6] = combine[x, 6] * 1.09678
            # SecondInputAirTemp
            modified_res[x, 7] = combine[x, 7] * 1.00234
            # SlurryPipelineLowerLayerPressure
            modified_res[x, 8] = combine[x, 8] * 1.0285
            # OutAirMotorFreq
            modified_res[x, 9] = combine[x, 9] * 0.9645
            # SecondAirMotorFreq
            modified_res[x, 10] = combine[x, 10] * 0.98235
            # HighPressurePumpFreq
            modified_res[x, 11] = combine[x, 11] * 1.054
            # GasFlow
            modified_res[x, 12] = combine[x, 12] * 0.98667

            modified_res[x, 0] = np.expm1(lasso_model.predict(np.reshape(modified_res[x][1:], (-1, 12))))
            

    final_com = np.column_stack((combine, modified_res))
 
    return render(request, 'blog/jingbai_ds.html', {'final_com': final_com})



def tbo_ds(request):
    lasso_model = joblib.load(os.path.join(BASE_DIR, 'ml_models/model_gboost_tbo.pkl'))
    df_ready = pd.read_csv(os.path.join(BASE_DIR, 'media/documents/tbo_ready.csv'))
    train_y = df_ready.M.values
    del df_ready["Unnamed: 0"]
    del df_ready["M"]
    train = df_ready.values
    train_pred = np.expm1(lasso_model.predict(train))

    combine = np.column_stack((train_pred, train))
    
    # need to delete the origin upload file
    os.remove(os.path.join(BASE_DIR, 'media/documents/tbo_ready.csv'))

    # rows = combine.shape[0]
    # cols = combine.shape[1]
    # modified_res = copy.deepcopy(combine)
    # for x in range(0, rows):
    #     count = 0
    #     while combine[x, 0] > 32.9:
    #         if count == 3: 
    #             break
    #         count = count + 1
    #         for y in range(1, cols-1):
    #             modified_res[x, y] = combine[x, y] * (1 - randint(9700, 10000) / 100000.0)
    #         # # modified_res[x, 0] = lgb_model.predict(np.reshape(modified_res[x][1:], (-1, 12)))
    #         modified_res[x, 0] = combine[x, 0] * (1 - randint(9800, 10000) / 100000.0)

    #         modified_res[x, cols -1] = combine[x, cols-1] * 0.9994448 *randint(980, 999) / 1000

    # final_com = np.column_stack((combine, modified_res))
    rows = combine.shape[0]
    cols = combine.shape[1]
    modified_res = copy.deepcopy(combine)
    for x in range(0, rows):
        # count = 0
        # while combine[x, 0] > 32.9:
        #     if count == 3: 
        #         break
        #     count = count + 1
        #     for y in range(1, cols-1):
        #         modified_res[x, y] = combine[x, y] * (1 - randint(9700, 10000) / 100000.0)
        #     # modified_res[x, 0] = lgb_model.predict(np.reshape(modified_res[x][1:], (-1, 12)))
        #     modified_res[x, 0] = combine[x, 0] * (1 - randint(9800, 10000) / 100000.0)

        #     modified_res[x, cols -1] = combine[x, cols-1] * 0.9994448 *randint(980, 999) / 1000
        if modified_res[x, 0] > 32.5 :
            # AirOutTemp
            modified_res[x, 1] = combine[x, 1]  * 0.98
            # BasePowderTemp
            modified_res[x, 2] = combine[x, 2]  * 0.99123
            # AirInTemp_1
            modified_res[x, 3] = combine[x, 3]  * 1.0456
            # SlurryTemp
            modified_res[x, 4] = combine[x, 4]  * 1.0678

            # TowerTopNegativePressure
            modified_res[x, 5] = combine[x, 5] * 0.9877
            # AgingTankFlow
            modified_res[x, 6] = combine[x, 6] * 1.09678
            # SecondInputAirTemp
            modified_res[x, 7] = combine[x, 7] * 1.00234
            # SlurryPipelineLowerLayerPressure
            modified_res[x, 8] = combine[x, 8] * 1.0285
            # OutAirMotorFreq
            modified_res[x, 9] = combine[x, 9] * 0.9645
            # SecondAirMotorFreq
            modified_res[x, 10] = combine[x, 10] * 0.98235
            # HighPressurePumpFreq
            modified_res[x, 11] = combine[x, 11] * 1.054
            # GasFlow
            modified_res[x, 12] = combine[x, 12] * 0.98667

            modified_res[x, 0] = np.expm1(lasso_model.predict(np.reshape(modified_res[x][1:], (-1, 12))))

    final_com = np.column_stack((combine, modified_res))
    return render(request, 'blog/tbo_ds.html', {'final_com': final_com})


def bilang_ds(request):
    lasso_model = joblib.load(os.path.join(BASE_DIR, 'ml_models/model_gboost_bilang.pkl'))
    df_ready = pd.read_csv(os.path.join(BASE_DIR, 'media/documents/bilang_ready.csv')) 
    train_y = df_ready.M.values
    del df_ready["Unnamed: 0"]
    del df_ready["M"]
    train = df_ready.values
    train_pred = np.expm1(lasso_model.predict(train))

    combine = np.column_stack((train_pred, train))
	
	# need to delete the origin upload file
    os.remove(os.path.join(BASE_DIR, 'media/documents/bilang_ready.csv'))

    rows = combine.shape[0]
    cols = combine.shape[1]
    modified_res = copy.deepcopy(combine)
    for x in range(0, rows):
        # count = 0
        # while combine[x, 0] > 32.9:
        #     if count == 3: 
        #         break
        #     count = count + 1
        #     for y in range(1, cols-1):
        #         modified_res[x, y] = combine[x, y] * (1 - randint(9700, 10000) / 100000.0)
        #     # modified_res[x, 0] = lgb_model.predict(np.reshape(modified_res[x][1:], (-1, 12)))
        #     modified_res[x, 0] = combine[x, 0] * (1 - randint(9800, 10000) / 100000.0)

        #     modified_res[x, cols -1] = combine[x, cols-1] * 0.9994448 *randint(980, 999) / 1000
        if combine[x, 0] > 33 :
            # AirOutTemp
            modified_res[x, 1] = combine[x, 1]  * 0.98
            # BasePowderTemp
            modified_res[x, 2] = combine[x, 2]  * 0.99123
            # AirInTemp_1
            modified_res[x, 3] = combine[x, 3]  * 1.0456
            # SlurryTemp
            modified_res[x, 4] = combine[x, 4]  * 1.0678

            # TowerTopNegativePressure
            modified_res[x, 5] = combine[x, 5] * 0.9877
            # AgingTankFlow
            modified_res[x, 6] = combine[x, 6] * 1.09678
            # SecondInputAirTemp
            modified_res[x, 7] = combine[x, 7] * 1.00234
            # SlurryPipelineLowerLayerPressure
            modified_res[x, 8] = combine[x, 8] * 1.0285
            # OutAirMotorFreq
            modified_res[x, 9] = combine[x, 9] * 0.9645
            # SecondAirMotorFreq
            modified_res[x, 10] = combine[x, 10] * 0.98235
            # HighPressurePumpFreq
            modified_res[x, 11] = combine[x, 11] * 1.054
            # GasFlow
            modified_res[x, 12] = combine[x, 12] * 0.98667

            modified_res[x, 0] = np.expm1(lasso_model.predict(np.reshape(modified_res[x][1:], (-1, 12))))

            

    final_com = np.column_stack((combine, modified_res))
    return render(request, 'blog/bilang_ds.html', {'final_com': final_com})

def bilang_steps(request):
    model = joblib.load(os.path.join(BASE_DIR, 'ml_models/model_gboost_bilang.pkl'))
    df_ready = pd.read_csv(os.path.join(BASE_DIR, 'media/documents/bilang_ready.csv')) 
    train_y = df_ready.M.values
    del df_ready["Unnamed: 0"]
    del df_ready["M"]
    train = df_ready.values
    train_pred = np.expm1(model.predict(train))

    combine = np.column_stack((train_pred, train))
    
    # need to delete the origin upload file
    os.remove(os.path.join(BASE_DIR, 'media/documents/bilang_ready.csv'))

    rows = combine.shape[0]
    cols = combine.shape[1]
    modified_res = copy.deepcopy(combine)
    for x in range(0, rows):
      
        if combine[x, 0] > 33 :
            # AirOutTemp
            modified_res[x, 1] = combine[x, 1]  * 0.98
            # BasePowderTemp
            modified_res[x, 2] = combine[x, 2]  * 0.99123
            # AirInTemp_1
            modified_res[x, 3] = combine[x, 3]  * 1.0456
            # SlurryTemp
            modified_res[x, 4] = combine[x, 4]  * 1.0678

            # TowerTopNegativePressure
            modified_res[x, 5] = combine[x, 5] * 0.9877
            # AgingTankFlow
            modified_res[x, 6] = combine[x, 6] * 1.09678
            # SecondInputAirTemp
            modified_res[x, 7] = combine[x, 7] * 1.00234
            # SlurryPipelineLowerLayerPressure
            modified_res[x, 8] = combine[x, 8] * 1.0285
            # OutAirMotorFreq
            modified_res[x, 9] = combine[x, 9] * 0.9645
            # SecondAirMotorFreq
            modified_res[x, 10] = combine[x, 10] * 0.98235
            # HighPressurePumpFreq
            modified_res[x, 11] = combine[x, 11] * 1.054
            # GasFlow
            modified_res[x, 12] = combine[x, 12] * 0.98667

            modified_res[x, 0] = np.expm1(model.predict(np.reshape(modified_res[x][1:], (-1, 12))))

            

    final_com = np.column_stack((combine, modified_res))
    return render(request, 'blog/bilang_steps.html', {'final_com': final_com})





def delete_bilang(request):

    shutil.rmtree(os.path.join(BASE_DIR, 'media/documents/'))

    return render(request, 'blog/bilang.html')

def delete_jingbai(request):

    shutil.rmtree(os.path.join(BASE_DIR, 'media/documents/'))

    return render(request, 'blog/jingbai.html')

def delete_tbo(request):

    shutil.rmtree(os.path.join(BASE_DIR, 'media/documents/'))

    return render(request, 'blog/tbo.html')

def jingbai_gasflow_highppf(request):

    model = joblib.load(os.path.join(BASE_DIR, 'ml_models/model_gboost_jingbai_gasflow_highpressurepf.pkl'))
    df_ready = pd.read_csv(os.path.join(BASE_DIR, 'media/documents/jingbai_two_vars_ready.csv')) 
    train_y = df_ready.GasFlow.values
    del df_ready["Unnamed: 0"]
    del df_ready["GasFlow"]
    train = df_ready.values
    train_pred = np.expm1(model.predict(train))

    combine = np.column_stack((train_pred, train))
    
    # need to delete the origin upload file
    os.remove(os.path.join(BASE_DIR, 'media/documents/jingbai_two_vars_ready.csv'))

    # rows = combine.shape[0]
    # cols = combine.shape[1]
    # modified_res = copy.deepcopy(combine)

    # for x in range(0, rows):

    #     modified_res[x, 0] = np.expm1(lasso_model.predict(np.reshape(modified_res[x][1:], (-1, 1))))

    final_com = combine
    return render(request, 'blog/jingbai_gasflow_highppf.html', {'final_com': final_com})


def bilang_gasflow_highppf(request):

    model = joblib.load(os.path.join(BASE_DIR, 'ml_models/model_gboost_bilang_gasflow_highpressurepf.pkl'))
    df_ready = pd.read_csv(os.path.join(BASE_DIR, 'media/documents/bilang_two_vars_ready.csv')) 
    train_y = df_ready.GasFlow.values
    del df_ready["Unnamed: 0"]
    del df_ready["GasFlow"]
    train = df_ready.values
    train_pred = np.expm1(model.predict(train))

    combine = np.column_stack((train_pred, train))
    
    # need to delete the origin upload file
    os.remove(os.path.join(BASE_DIR, 'media/documents/bilang_two_vars_ready.csv'))

    # rows = combine.shape[0]
    # cols = combine.shape[1]
    # modified_res = copy.deepcopy(combine)

    # for x in range(0, rows):

    #     modified_res[x, 0] = np.expm1(lasso_model.predict(np.reshape(modified_res[x][1:], (-1, 1))))

    final_com = combine
    return render(request, 'blog/bilang_gasflow_highppf.html', {'final_com': final_com})


def tbo_gasflow_highppf(request):

    model = joblib.load(os.path.join(BASE_DIR, 'ml_models/model_gboost_tbo_gasflow_highpressurepf.pkl'))
    df_ready = pd.read_csv(os.path.join(BASE_DIR, 'media/documents/tbo_two_vars_ready.csv')) 
    train_y = df_ready.GasFlow.values
    del df_ready["Unnamed: 0"]
    del df_ready["GasFlow"]
    train = df_ready.values
    train_pred = np.expm1(model.predict(train))

    combine = np.column_stack((train_pred, train))
    
    # need to delete the origin upload file
    os.remove(os.path.join(BASE_DIR, 'media/documents/tbo_two_vars_ready.csv'))

    # rows = combine.shape[0]
    # cols = combine.shape[1]
    # modified_res = copy.deepcopy(combine)

    # for x in range(0, rows):

    #     modified_res[x, 0] = np.expm1(lasso_model.predict(np.reshape(modified_res[x][1:], (-1, 1))))

    final_com = combine
    return render(request, 'blog/tbo_gasflow_highppf.html', {'final_com': final_com})




from django.shortcuts import render

# Create your views here.
# from django.http import HttpResponse
from django.http import HttpResponse, JsonResponse
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response

from django.views.decorators.csrf import csrf_exempt
from rest_framework.renderers import JSONRenderer
from rest_framework.parsers import JSONParser
from blog.models import Snippet
from blog.serializers import SnippetSerializer, ValuedataSerializer

from django.http import Http404
from rest_framework.views import APIView
from rest_framework import mixins
from rest_framework import generics
import os
import json



@api_view(['GET', 'POST'])
def snippet_list(request, format=None):
    """
    List all code snippets, or create a new snippet.
    """
    if request.method == 'GET':
        snippets = Snippet.objects.all()
        serializer = SnippetSerializer(snippets, many=True)
        return Response(serializer.data)

    elif request.method == 'POST':
        serializer = SnippetSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)



@api_view(['GET', 'PUT', 'DELETE'])
def snippet_detail(request, pk, format=None):
    """
    Retrieve, update or delete a code snippet.
    """
    try:
        snippet = Snippet.objects.get(pk=pk)
    except Snippet.DoesNotExist:
        return Response(status=status.HTTP_404_NOT_FOUND)

    if request.method == 'GET':
        serializer = SnippetSerializer(snippet)
        return Response(serializer.data)

    elif request.method == 'PUT':
        serializer = SnippetSerializer(snippet, data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    elif request.method == 'DELETE':
        snippet.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)

from influxdb import InfluxDBClient
hostname = '127.0.0.1' #'143.35.218.66'
port_num = 8086
db_user_name = 'root'
db_password = 'root'
database_name = 'picus_db'
#setup the connection with influx db
con = InfluxDBClient(hostname, port_num, db_user_name, db_user_name, database_name)
# con.write_points(json_body)
# 
# 
# air_out_temp, base_powder_temp,  air_in_temp_1, slurry_temp, tower_top_negative_pressure, aging_tank_flow, second_input_air_temp, slurry_pipeline_lower_layer_pressure, out_air_motor_freq, second_air_motor_freq, high_pressure_pump_freq, gas_flow,p_slurry_pipeline_lower_layer_pressure, p_out_air_motor_freq, p_second_air_motor_freq, p_high_pressure_pump_freq, p_gas_flow,p_air_out_temp, p_base_powder_temp,  p_air_in_temp_1, p_slurry_temp, p_tower_top_negative_pressure, p_aging_tank_flow, p_second_input_air_temp

# 
# 
@api_view(['GET'])
def getlatest(request, format=None):
    if request.method == "GET":
        result = con.query("select f_m, modified_m, air_out_temp, base_powder_temp,  air_in_temp_1, slurry_temp, tower_top_negative_pressure, aging_tank_flow, second_input_air_temp, slurry_pipeline_lower_layer_pressure, out_air_motor_freq, second_air_motor_freq, high_pressure_pump_freq, gas_flow,p_slurry_pipeline_lower_layer_pressure, p_out_air_motor_freq, p_second_air_motor_freq, p_high_pressure_pump_freq, p_gas_flow,p_air_out_temp, p_base_powder_temp,  p_air_in_temp_1, p_slurry_temp, p_tower_top_negative_pressure, p_aging_tank_flow, p_second_input_air_temp  from value_data  order by desc limit 1")
        values = result.raw['series'][0]['values'][0]
        keys   = result.raw['series'][0]['columns']

        res = dict(zip(keys, values))
        con.close()
        # res =json.loads('{"one" : "111", "two" : "2", "three" : "3"}')
        return Response(res)
        


# process value data from OPC via restful API call
@api_view(['GET', 'POST'])
def value_data_process(request, format=None):

    # return Response("hello world!!! michael is here for you, haha ~~")
    if request.method == 'POST':
        serializer = ValuedataSerializer(data=request.data)
        if serializer.is_valid():
            # print serializer.data.get("aging_tank_flow")
            # print serializer.data.get("air_in_temp_1")
            d = {
                'air_out_temp': [serializer.data.get("air_out_temp")], 
                'base_powder_temp': [serializer.data.get("base_powder_temp")], 
                'air_in_temp_1': [serializer.data.get("air_in_temp_1")], 
                'slurry_temp': [serializer.data.get("slurry_temp")], 
                'tower_top_negative_pressure':[serializer.data.get("tower_top_negative_pressure")],
                'aging_tank_flow': [serializer.data.get("aging_tank_flow")], 
                'second_input_air_temp': [serializer.data.get("second_input_air_temp")], 
                'slurry_pipeline_lower_layer_pressure': [serializer.data.get("slurry_pipeline_lower_layer_pressure")], 
                'out_air_motor_freq': [serializer.data.get("out_air_motor_freq")], 
                'second_air_motor_freq': [serializer.data.get("second_air_motor_freq")], 
                'high_pressure_pump_freq': [serializer.data.get("high_pressure_pump_freq")], 
                'gas_flow':[serializer.data.get("gas_flow")],
                'brand' : [serializer.data.get("brand")],
                'f_m' : [serializer.data.get("f_m")],
                'density_checking_switch_2' : [serializer.data.get("density_checking_switch_2")],
            }
            
            data = pd.DataFrame(data=d, columns=['air_out_temp', 'base_powder_temp', 'air_in_temp_1', 'slurry_temp', 'tower_top_negative_pressure',
                    'aging_tank_flow', 'second_input_air_temp', 'slurry_pipeline_lower_layer_pressure', 
                    'out_air_motor_freq', 'second_air_motor_freq', 'high_pressure_pump_freq', 'gas_flow', 'brand', 'f_m', 'density_checking_switch_2'])
            pred_m = 0
            
            res, pred_m = data_process(data)
            
            measurement = "value_data"
            host_name = "127.0.0.1"
            region_value = "us_west"

            json_body = [
                {
                    "measurement": measurement,
                    "tags": {
                        "host": host_name,
                        "region": region_value
                    },
                    "fields": {
                        "brand": serializer.data.get("brand"),
                        "air_out_temp": round(float(serializer.data.get("air_out_temp")), 2),
                        "base_powder_temp": round(float(serializer.data.get("base_powder_temp")), 2),
                        "air_in_temp_1": round(float(serializer.data.get("air_in_temp_1")), 2),
                        "slurry_temp": round(float(serializer.data.get("slurry_temp")), 2),
                        "tower_top_negative_pressure": round(float(serializer.data.get("tower_top_negative_pressure")), 2),
                        "aging_tank_flow": round(float(serializer.data.get("aging_tank_flow")), 2),
                        "second_input_air_temp": round(float(serializer.data.get("second_input_air_temp")), 2),
                        "slurry_pipeline_lower_layer_pressure": round(float(serializer.data.get("slurry_pipeline_lower_layer_pressure")), 2),
                        "out_air_motor_freq": round(float(serializer.data.get("out_air_motor_freq")), 2),
                        "second_air_motor_freq": round(float(serializer.data.get("second_air_motor_freq")), 2),
                        "high_pressure_pump_freq": round(float(serializer.data.get("high_pressure_pump_freq")), 2),
                        "gas_flow": round(float(serializer.data.get("gas_flow")), 2),
                        "p_air_out_temp": round(float(res[0][1]), 2),
                        "p_base_powder_temp": round(float(res[0][2]), 2),
                        "p_air_in_temp_1": round(float(res[0][3]), 2),
                        "p_slurry_temp": round(float(res[0][4]), 2),
                        "p_tower_top_negative_pressure": round(float(res[0][5]), 2),
                        "p_aging_tank_flow": round(float(res[0][6]), 2),
                        "p_second_input_air_temp": round(float(res[0][7]), 2),
                        "p_slurry_pipeline_lower_layer_pressure": round(float(res[0][8]), 2),
                        "p_out_air_motor_freq": round(float(res[0][9]), 2),
                        "p_second_air_motor_freq": round(float(res[0][10]), 2),
                        "p_high_pressure_pump_freq": round(float(res[0][11]), 2),
                        "p_gas_flow": round(float(res[0][12]), 2),
                        "f_m" : round(float(serializer.data.get("f_m")), 2),
                        "pred_m" : round(float(pred_m[0]), 2),
                        "modified_m" :round(float(res[0][0]), 2),
                        "slurry_density" : round(float(serializer.data.get("slurry_density")), 4),
                        "host": serializer.data.get("host"),


                        "aging_tank_a_temp" : round(float(serializer.data.get("aging_tank_a_temp")), 2),
                        "aging_tank_b_temp" : round(float(serializer.data.get("aging_tank_b_temp")), 2),
                        "head_tank_liquid_level_low_setting" : round(float(serializer.data.get("head_tank_liquid_level_low_setting")), 2),
                        "head_tank_liquid_level_high_setting" : round(float(serializer.data.get("head_tank_liquid_level_high_setting")), 2),
                        "sulfate_silo_low_level" : round(float(serializer.data.get("sulfate_silo_low_level")), 2),
                        "sulfatesilo_high_level" : round(float(serializer.data.get("sulfatesilo_high_level")), 2),
                        "sulfate_silo_weightlessness_scale_setting" : round(float(serializer.data.get("sulfate_silo_weightlessness_scale_setting")), 2),
                        "sulfate_silo_weightlessness_scale_actual" : round(float(serializer.data.get("sulfate_silo_weightlessness_scale_actual")), 2),
                        "sulfate_silo_weightlessness_scale_motor_freq" : round(float(serializer.data.get("sulfate_silo_weightlessness_scale_motor_freq")), 2),
                        "minor_material_silo_low_level" : round(float(serializer.data.get("minor_material_silo_low_level")), 2),
                        "minor_material_silo_high_level" : round(float(serializer.data.get("minor_material_silo_high_level")), 2),
                        "brighter_minor_material_setting" : round(float(serializer.data.get("brighter_minor_material_setting")), 2),
                        "brighter_minor_material_actual" : round(float(serializer.data.get("brighter_minor_material_actual")), 2),
                        "brighter_minor_material_motor_freq" : round(float(serializer.data.get("brighter_minor_material_motor_freq")), 2),
                        "carbonate_silo_high_level" : round(float(serializer.data.get("carbonate_silo_high_level")), 2),
                        "carbonate_silo_low_level" : round(float(serializer.data.get("carbonate_silo_low_level")), 2),
                        "carbonate_silo_setting" : round(float(serializer.data.get("carbonate_silo_setting")), 2),
                        "carbonate_silo_actual" : round(float(serializer.data.get("carbonate_silo_actual")), 2),
                        "carbonate_silo_motor_freq" : round(float(serializer.data.get("carbonate_silo_motor_freq")), 2),

                        "hlas_mass_flow_meter_setting" : round(float(serializer.data.get("hlas_mass_flow_meter_setting")), 2),
                        "naoh_mass_flowm_eter_setting" : round(float(serializer.data.get("naoh_mass_flowm_eter_setting")), 2),
                        "aging_tank_a_flow" : round(float(serializer.data.get("aging_tank_a_flow")), 2),
                        "aging_tank_b_flow" : round(float(serializer.data.get("aging_tank_b_flow")), 2),
                        "aging_tank_a_outlet_valve" : round(float(serializer.data.get("aging_tank_a_outlet_valve")), 2),
                        "aging_tank_b_outlet_valve" : round(float(serializer.data.get("aging_tank_b_outlet_valve")), 2),
                        "air_in_temp_2" : round(float(serializer.data.get("air_in_temp_2")), 2),
                        "high_pressure_pump_a_freq" : round(float(serializer.data.get("high_pressure_pump_a_freq")), 2),
                        "high_pressure_pump_b_freq" : round(float(serializer.data.get("high_pressure_pump_b_freq")), 2),
                        "las_mass_flow_meter_actual" : round(float(serializer.data.get("las_mass_flow_meter_actual")),2),

                        "las_mass_flow_meter_setting" : round(float(serializer.data.get("las_mass_flow_meter_setting")), 2),
                        "rv_base_mass_flow_meter_setting" : round(float(serializer.data.get("rv_base_mass_flow_meter_setting")), 2),
                        "rv_base_mass_flow_meter_actual" : round(float(serializer.data.get("rv_base_mass_flow_meter_actual")), 2),
                        "ev_base_mass_flow_meter_acutal" : round(float(serializer.data.get("ev_base_mass_flow_meter_acutal")), 2),
                        "ev_base_mass_flow_meter_setting" : round(float(serializer.data.get("ev_base_mass_flow_meter_setting")), 2),
                        "silicate_nass_flow_meter_actual" : round(float(serializer.data.get("silicate_nass_flow_meter_actual")), 2),
                        "silicate_mass_flow_meter_setting" : round(float(serializer.data.get("silicate_mass_flow_meter_setting")), 2),
                        "processed_water_mass_flow_meter_setting" : round(float(serializer.data.get("processed_water_mass_flow_meter_setting")), 2),
                        "processed_water_mass_flow_meter_actual" : round(float(serializer.data.get("processed_water_mass_flow_meter_actual")), 2),
                        "remelt_water_mass_flow_meter_setting" : round(float(serializer.data.get("remelt_water_mass_flow_meter_setting")), 2),

                        "remelt_water_mass_flow_meter_actual" : round(float(serializer.data.get("remelt_water_mass_flow_meter_actual")), 2),
                        "sulfate_silo_high_level_outlet_valve" : round(float(serializer.data.get("sulfate_silo_high_level_outlet_valve")), 2),
                        "sulfate_silo_low_level_outlet_valve" : round(float(serializer.data.get("sulfate_silo_low_level_outlet_valve")), 2),
                        "minor_material_silo_high_level_outlet_valve" : round(float(serializer.data.get("minor_material_silo_high_level_outlet_valve")), 2),
                        "minor_material_silo_low_level_outlet_valve" : round(float(serializer.data.get("minor_material_silo_low_level_outlet_valve")), 2),
                        "carbonate_silo_high_level_outlet_valve" : round(float(serializer.data.get("carbonate_silo_high_level_outlet_valve")), 2),
                        "carbonate_silo_low_level_outlet_valve" : round(float(serializer.data.get("carbonate_silo_low_level_outlet_valve")), 2),
                        "hlas_mass_flow_meter_actual_value" : round(float(serializer.data.get("hlas_mass_flow_meter_actual_value")), 2),
                        "naoh_mass_flow_meter_actual_value" : round(float(serializer.data.get("naoh_mass_flow_meter_actual_value")), 2),
                        "slurry_pipeline_upper_layer_pressure" : round(float(serializer.data.get("slurry_pipeline_upper_layer_pressure")), 2),

                        "base_power_flow_setting_value" : round(float(serializer.data.get("base_power_flow_setting_value")), 2),
                        "base_power_flow_acutal_value" : round(float(serializer.data.get("base_power_flow_acutal_value")), 2),
                        "powder_motor_freq" : round(float(serializer.data.get("powder_motor_freq")), 2),
                        "slurry_pipe_temp" : round(float(serializer.data.get("slurry_pipe_temp")), 2),
                        "sulfate_weight" : round(float(serializer.data.get("sulfate_weight")), 2),
                        "carbonate_weight" : round(float(serializer.data.get("carbonate_weight")), 2),
                        "brighter_minor_material_weight" : round(float(serializer.data.get("brighter_minor_material_weight")), 2),
                        "out_air_motor_freq" : round(float(serializer.data.get("out_air_motor_freq")), 2),
                        "air_in_temp_4" : round(float(serializer.data.get("air_in_temp_4")), 2),
                        "base_powder_weight" : round(float(serializer.data.get("base_powder_weight")), 2),

                        "waste_water_actual" : round(float(serializer.data.get("waste_water_actual")), 2),
                        "waste_water_setting" : round(float(serializer.data.get("waste_water_setting")), 2),
                        "las_open" : round(float(serializer.data.get("las_open")), 2),
                        "base_powder_open" : round(float(serializer.data.get("base_powder_open")), 2),
                        "steam_flow" : round(float(serializer.data.get("steam_flow")), 2),
                        "density_checking_switch_1" : round(float(serializer.data.get("density_checking_switch_1")), 2),
                        "density_checking_switch_2" : round(float(serializer.data.get("density_checking_switch_2")), 2),
                        "high_pressure_pump_entry_pressure" : round(float(serializer.data.get("high_pressure_pump_entry_pressure")), 2),
                        "high_pressure_pump_entry_flow" : round(float(serializer.data.get("high_pressure_pump_entry_flow")), 2),
                        "high_pressure_pump_a_freq_new" : round(float(serializer.data.get("high_pressure_pump_a_freq_new")), 2),

                        "high_pressure_pump_b_freq_new" : round(float(serializer.data.get("high_pressure_pump_b_freq_new")), 2),
                        "exhaust_freq_new" : round(float(serializer.data.get("exhaust_freq_new")), 2),
                        
                        "flag_aging_tank_flow" : float(1),
                        "flag_air_in_temp_1" : float(1), 
                        "flag_air_out_temp" : float(1),
                        "flag_base_powder_temp" :float(1) , 
                        "flag_gas_flow" : float(1), 
                        "flag_high_pressure_pump_freq" : float(1),
                        "flag_out_air_motor_freq" : float(1),
                        "flag_second_air_motor_freq" : float(1),
                        "flag_second_input_air_temp" : float(1),
                        "flag_slurry_pipeline_lower_layer_pressure" : float(1),
                        "flag_slurry_temp" : float(1),
                        "flag_tower_top_negative_pressure" : float(1),
                        "flag_slurry_density" : float(1),
                        "flag_density_checking_switch_1" : float(1),
                        "flag_density_checking_switch_2" : float(1),

                    }
                }
            ]
            
            con.write_points(json_body)
            print "post sucessfully!"
            con.close()
            return Response(serializer.data, status=status.HTTP_201_CREATED)


    elif request.method == 'GET':
        pass

    else:
        print "post failure!!!"
        return Response("error! sorry!")

   

# 1 -- others
# 2 -- jingbai
# 3 -- bilang
# 4 -- tbo
def data_process(data):
    if data['brand'][0] == 2.0:
                print "processing jingbai!"
                res, pred_m= jingbai_process_v2(data)
    elif data['brand'][0] == 3.0:
                print "processing bilang!"
                res, pred_m = bilang_process(data)
                # res, pred_m =jingbai_process_v2(data) # for testing
    elif data['brand'][0] == 4.0:
                print "processing tbo or others!"
                res, pred_m= tbo_process(data)
    return res, pred_m


# jb_INTERVAL = 120
# jb_count = 0
# jb_tmp = ""

def jingbai_process(data):
    model = joblib.load(os.path.join(BASE_DIR, 'ml_models/model_gboost_jingbai.pkl'))
    density_checking_switch = data.iloc[0]['density_checking_switch_2']
    print "------------------------------"
    print data['density_checking_switch_2']
    print density_checking_switch
    print density_checking_switch > 504
    print "------------------------------"
    del data['density_checking_switch_2']
    del data['brand']
    df_ready = data 
   
    train = df_ready.values
    # train_pred = np.expm1(model.predict(train))
    train_pred = data['f_m'] * 0.993588
    print train_pred
    combine = np.column_stack((train_pred, train))
   
    rows = combine.shape[0]
    cols = combine.shape[1]
    modified_res = copy.deepcopy(combine)
    for x in range(0, rows):
        
        if ((combine[x, 0] > 33) and (combine[x, 2] >= 105) and (density_checking_switch > 580) and (density_checking_switch < 640)):
            # jb_count = jb_count + 1
            # AirOutTemp
            # if combine[x, 1] > 130 :
            #     modified_res[x, 1] = 129.99
            # elif combine[x, 1] < 76 :
            #     modified_res[x, 1] = 76.001
            # else:
            #     modified_res[x, 1] = combine[x, 1]  * 0.993
            if combine[x, 1] < 79 and combine[x, 1] > 70:
                modified_res[x, 1] = round(combine[x, 1]  * 0.981, 2)


            # BasePowderTemp 
            # if combine[x, 2] > 166 :
            #     modified_res[x, 2] = 165.99
            # elif combine[x, 2] < 95 :
            #     modified_res[x, 2] = 95.001
            # else :
            #     modified_res[x, 2] = combine[x, 2]  * 0.997
            modified_res[x, 2] = round(combine[x, 2]  * 1.0035, 2)
            if modified_res[x, 2] < 106:
                modified_res[x, 2] = 106


            # AirInTemp_1# 
            # if combine[x, 3] > 302 :
            #     modified_res[x, 3] = 301.999
            # elif combine[x, 3] < 238 :
            #     modified_res[x, 3] = 238.001
            # else:
            #     modified_res[x, 3] = combine[x, 3]  * 1.0152
            modified_res[x, 3] = round(combine[x, 3]  + randint(6, 15) * 1, 2)
            if modified_res[x, 3] > 279.1:
                modified_res[x, 3] = 279

            # SlurryTemp# 
            # if combine[x, 4] > 894:
            #     modified_res[x, 4] = 893.99
            # elif combine[x, 4] < 0:
            #     modified_res[x, 4] = 0.001
            # else:
            #     modified_res[x, 4] = combine[x, 4]  * 1.0226
            modified_res[x, 4] = round(combine[x, 4]  + randint(4, 9) * 1, 2)
            if modified_res[x, 4] > 71.0:
                modified_res[x, 4] = 71

            # TowerTopNegativePressure 
            # if combine[x, 5] > 0:
            #     modified_res[x, 5] = -0.000001
            # elif combine[x, 5] < -30.0:
            #     modified_res[x, 5] = -29.988
            # else:
            #     modified_res[x, 5] = combine[x, 5] * 0.9959
            modified_res[x, 5] = round(combine[x, 5] * 0.9907, 2)

            # AgingTankFlow 
            # if combine[x, 6] > 27474:
            #     modified_res[x, 6] = 27473.999
            # elif combine[x, 6] < 17451:
            #     modified_res[x, 6] = 17451.02
            # else:
            #     modified_res[x, 6] = combine[x, 6] * 1.02226
            # modified_res[x, 6] = round(combine[x, 6] * 1.0355, 2) + round(31.89 * randint(2, 4), 2)

            # SecondInputAirTemp 
            # if combine[x, 7] > 68:
            #     modified_res[x, 7] = 67.99
            # elif combine[x, 7] < 0:
            #     modified_res[x, 7] = -0.00001
            # else:
            #     modified_res[x, 7] = combine[x, 7] * 1.00078
            modified_res[x, 7] = round(combine[x, 7] * 1.00158, 2)

            # SlurryPipelineLowerLayerPressure
            # if combine[x, 8] > 76:
            #     modified_res[x, 8] = 75.999
            # elif combine[x, 8] < 42:
            #     modified_res[x, 8] = 42.0009
            # else:
            #     modified_res[x, 8] = combine[x, 8] * 1.0095
            modified_res[x, 8] = round(combine[x, 8] * 1.0195, 2)

            # OutAirMotorFreq#
            # if combine[x, 9] > 0.9:
            #     modified_res[x, 9] = 0.899999
            # elif combine[x, 9] < 0.6:
            #     modified_res[x, 9] = 0.6001
            # else:
            #     modified_res[x, 9] = combine[x, 9] * 0.98817
            modified_res[x, 9] = round(combine[x, 9] - randint(4, 11) * 0.2, 2)

            # SecondAirMotorFreq# 
            # if combine[x, 10] > 88:
            #     modified_res[x, 10] = 87.99
            # elif combine[x, 10] < 53:
            #     modified_res[x, 10] = 53.001
            # else:
            #     modified_res[x, 10] = combine[x, 10] * 0.9941167
            if combine[x, 10] > 61.0:
                modified_res[x, 10] = round(combine[x, 10] - randint(3, 8) * 0.5, 2)

            # HighPressurePumpFreq#
            # if combine[x, 11] > 37.6:
            #     modified_res[x, 11] = 37.59
            # elif combine[x, 11] < 8.6:
            #     modified_res[x, 11] = 8.699
            # else:
            #     modified_res[x, 11] = combine[x, 11] * 1.018

            RANDOM = randint(0, 1)
            if combine[x, 11] > 30 and combine[x, 11] < 35:
                modified_res[x, 11] = round(combine[x, 11] + 1, 2)

            modified_res[x, 6] = round(combine[x, 6] * 1.0355, 2) + round(299 * RANDOM, 2) + 99

            # GasFlow# 
            # if combine[x, 12] > 722:
            #     modified_res[x, 12] = 721.99
            # elif combine[x, 12] < 500:
            #     modified_res[x, 12] = 500.001
            # else:
            #     modified_res[x, 12] = combine[x, 12] * 0.99857
            if combine[x, 2] > 109 and density_checking_switch < 635 and density_checking_switch > 600:
                modified_res[x, 12] = round(combine[x, 12] - randint(1, 8) * 2, 2)

            # modified_res[x, 0] = np.expm1(model.predict(np.reshape(modified_res[x][1:], (-1, 12))))
            modified_res[x, 0] = train_pred[x] * 0.9669


            ##conditioins
            if combine[x, 2] < 108 and (combine[x, 11]==34 or combine[x, 11] == 35): 
                modified_res[x, 11] = combine[x, 11] - 1
                modified_res[x, 10] = round(combine[x, 10] + randint(3, 8) * 0.5, 2)


            if combine[x, 3] > 279:
                modified_res[x, 10] = round(combine[x, 10] + randint(3, 8) * 0.5, 2)
                modified_res[x, 9] = round(combine[x, 9] + randint(4, 11) * 0.2, 2)
                
            # jb_tmp = modified_res
        # modified_res = jb_tmp
        elif combine[x, 2] < 105:
            modified_res[x] = -1
        else :
            modified_res[x] = -1
 
    return modified_res, train_pred



def jingbai_process_v2(data):
    print "------------------------------"
    print "this is a test for jingbai version2.0"
    print "------------------------------"

    model = joblib.load(os.path.join(BASE_DIR, 'ml_models/model_gboost_jingbai_20181023.pkl'))
    density_checking_switch = data.iloc[0]['density_checking_switch_2']
    m = data['f_m'][0]
    del data['density_checking_switch_2']
    del data['brand']
    del data['f_m']

    df_ready = data 

    train = df_ready.values
    train_pred = np.expm1(model.predict(train))
    
    combine = np.column_stack((train_pred, train))
   
    rows = combine.shape[0]
    cols = combine.shape[1]
    modified_res = copy.deepcopy(combine)
    print "before modified"
    print modified_res
    for x in range(0, rows):
        
        if ((combine[x, 0] > 33) and (combine[x, 2] >= 103) and (density_checking_switch > 530) and (density_checking_switch < 630)):
            delta_airouttemp = combine[x, 1] * 0.001
            delta_basepowdertemp = combine[x, 2] * 0.00017
            delta_airintemp1 = 1  #max 279
            delta_slurrytemp = 1  #max 71
            delta_ttnp  = combine[x, 5] * 0.0005
            delta_agingtankflow = 20
            delta_secinputairtemp = combine[x, 7] * 0.00008
            delta_splllp = combine[x, 8] * 0.0005
            delta_outairmotorfreq = 0.2
            delta_secairmotorfreq = 0.5
            delta_highpp = 1 #max 35 min 31
            delta_gasflow = 1.5 #max 700 min 500
            arr1 = copy.deepcopy(combine)
            arr02 = copy.deepcopy(combine)
            arr3 = copy.deepcopy(combine)
            arr4 = copy.deepcopy(combine)
            arr5 = copy.deepcopy(combine)
            arr6 = copy.deepcopy(combine)
            arr7 = copy.deepcopy(combine)
            arr8 = copy.deepcopy(combine)
            arr9 = copy.deepcopy(combine)
            arr10 = copy.deepcopy(combine)
            arr11 = copy.deepcopy(combine)
            arr12 = copy.deepcopy(combine)
            print "origin: "
            print arr1[x, 1], arr02[x, 2],arr3[x, 3] , arr4[x, 4], arr5[x, 5], arr6[x, 6] , arr7[x, 7], arr8[x, 8], arr9[x, 9],arr10[x, 10], arr11[x, 11], arr12[x, 12]
            
            AIR_IN_TEMP_1 = arr3[x, 3]

            arr1[x, 1] = arr1[x, 1] + delta_airouttemp
            arr02[x, 2] = arr02[x, 2] + delta_basepowdertemp
            arr3[x, 3] = arr3[x, 3] + delta_airintemp1
            arr4[x, 4] = arr4[x, 4] + delta_slurrytemp
            arr5[x, 5] = arr5[x, 5] + delta_ttnp
            arr6[x, 6] = arr6[x, 6] + delta_agingtankflow
            arr7[x, 7] = arr7[x, 7] + delta_secinputairtemp
            arr8[x, 8] = arr8[x, 8] + delta_splllp
            arr9[x, 9] = arr9[x, 9] + delta_outairmotorfreq
            arr10[x, 10] = arr10[x, 10] + delta_secairmotorfreq

            arr9[x, 10] = arr10[x, 10]

            arr11[x, 11] = arr11[x, 11] + delta_highpp
            arr12[x, 12] = arr12[x, 12] + delta_gasflow

            GAS_FLOW = arr12[x, 12]


            for item in range(20):
                print "==========================================="
                print "iterating item"
                print item

                print "before: "
                print arr1[x, 1], arr02[x, 2],arr3[x, 3] , arr4[x, 4], arr5[x, 5], arr6[x, 6] , arr7[x, 7], arr8[x, 8], arr9[x, 9],arr10[x, 10], arr11[x, 11], arr12[x, 12]

                if np.expm1(model.predict(np.reshape(arr1[x][1:], (-1, 12)))) >= combine[x, 0] :
                    arr1[x, 1] = arr1[x, 1] - 2 * delta_airouttemp
                    if np.expm1(model.predict(np.reshape(arr1[x][1:], (-1, 12)))) >= combine[x, 0]:
                        arr1[x, 1] = arr1[x, 1] + delta_airouttemp
                    else :
                        arr1[x, 1] = arr1[x, 1] - delta_airouttemp
                else:
                    arr1[x, 1] = arr1[x, 1] + delta_airouttemp

                if np.expm1(model.predict(np.reshape(arr02[x][1:], (-1, 12)))) >= combine[x, 0] :
                    arr02[x, 2] = arr02[x, 2] - 2 * delta_basepowdertemp
                    if np.expm1(model.predict(np.reshape(arr02[x][1:], (-1, 12)))) >= combine[x, 0] :
                        arr02[x, 2] = arr02[x, 2] + delta_basepowdertemp
                    else :
                        arr02[x, 2] = arr02[x, 2] - delta_basepowdertemp
                else :
                    arr02[x, 2] = arr02[x, 2] + delta_basepowdertemp


                if np.expm1(model.predict(np.reshape(arr3[x][1:], (-1, 12)))) >= combine[x, 0] :
                    arr3[x, 3] = arr3[x, 3] - 2 * delta_airintemp1
                    if np.expm1(model.predict(np.reshape(arr3[x][1:], (-1, 12)))) >= combine[x, 0] :
                        arr3[x, 3] = arr3[x, 3] + delta_airintemp1
                    else : 
                        arr3[x, 3] = arr3[x, 3] - delta_airintemp1
                else :
                    arr3[x, 3] = arr3[x, 3] + delta_airintemp1

                if arr3[x, 3] >= 279:
                    arr3[x, 3] = 278.5


                if np.expm1(model.predict(np.reshape(arr4[x][1:], (-1, 12)))) >= combine[x, 0] :
                    arr4[x, 4] = arr4[x, 4] - 2 * delta_slurrytemp
                    if np.expm1(model.predict(np.reshape(arr4[x][1:], (-1, 12)))) >= combine[x, 0] :
                        arr4[x, 4] = arr4[x, 4] + delta_slurrytemp
                    else :
                        arr4[x, 4] = arr4[x, 4] - delta_slurrytemp
                else :
                    arr4[x, 4] = arr4[x, 4] + delta_slurrytemp


                if np.expm1(model.predict(np.reshape(arr5[x][1:], (-1, 12)))) >= combine[x, 0] :
                    arr5[x, 5] = arr5[x, 5] - 2 * delta_ttnp
                    if np.expm1(model.predict(np.reshape(arr5[x][1:], (-1, 12)))) >= combine[x, 0] :
                        arr5[x, 5] = arr5[x, 5] + delta_ttnp
                    else : 
                        arr5[x, 5] = arr5[x, 5] - delta_ttnp
                else :
                    arr5[x, 5] = arr5[x, 5] + delta_ttnp
                
                if np.expm1(model.predict(np.reshape(arr6[x][1:], (-1, 12)))) >= combine[x, 0] :
                    arr6[x, 6] = arr6[x, 6] - 2 * delta_agingtankflow
                    if np.expm1(model.predict(np.reshape(arr6[x][1:], (-1, 12)))) >= combine[x, 0] :
                        arr6[x, 6] = arr6[x, 6] + delta_agingtankflow
                    else :
                        arr6[x, 6] = arr6[x, 6] - delta_agingtankflow
                else :
                    arr6[x, 6] = arr6[x, 6] + delta_agingtankflow


                if np.expm1(model.predict(np.reshape(arr7[x][1:], (-1, 12)))) >= combine[x, 0] :
                    arr7[x, 7] = arr7[x, 7] - 2 * delta_secinputairtemp
                    if np.expm1(model.predict(np.reshape(arr7[x][1:], (-1, 12)))) >= combine[x, 0] :
                        arr7[x, 7] = arr7[x, 7] + delta_secinputairtemp
                    else :
                        arr7[x, 7] = arr7[x, 7] - delta_secinputairtemp
                else :
                    arr7[x, 7] = arr7[x, 7] + delta_secinputairtemp


                if np.expm1(model.predict(np.reshape(arr8[x][1:], (-1, 12)))) >= combine[x, 0] :
                    arr8[x, 8] = arr8[x, 8] - 2 * delta_splllp
                    if np.expm1(model.predict(np.reshape(arr8[x][1:], (-1, 12)))) >= combine[x, 0] :
                        arr8[x, 8] = arr8[x, 8] + delta_splllp
                    else :
                        arr8[x, 8] = arr8[x, 8] - delta_splllp
                else :
                    arr8[x, 8] = arr8[x, 8] + delta_splllp

                if AIR_IN_TEMP_1 < 279 :

                    if np.expm1(model.predict(np.reshape(arr9[x][1:], (-1, 12)))) >= combine[x, 0] :
                        arr9[x, 9] = arr9[x, 9] - 2 * delta_outairmotorfreq
                        arr9[x, 10] = arr9[x, 10] - 2 * delta_secairmotorfreq
                        if np.expm1(model.predict(np.reshape(arr9[x][1:], (-1, 12)))) >= combine[x, 0] :
                          arr9[x, 9] = arr9[x, 9] + delta_outairmotorfreq
                          arr9[x, 10] = arr9[x, 10] + delta_secairmotorfreq
                        else :
                            arr9[x, 9] = arr9[x, 9] - delta_outairmotorfreq
                            arr9[x, 10] = arr9[x, 10] - delta_secairmotorfreq
                    else :
                        arr9[x, 9] = arr9[x, 9] + delta_outairmotorfreq
                        arr9[x, 10] = arr9[x, 10] + delta_secairmotorfreq

                else :
                    arr9[x, 9] = arr9[x, 9] - delta_outairmotorfreq
                    arr9[x, 10] = arr9[x, 10] - delta_secairmotorfreq




                # if np.expm1(model.predict(np.reshape(arr10[x][1:], (-1, 12)))) >= combine[x, 0] :
                #     arr10[x, 10] = arr10[x, 10] - 2 * delta_secairmotorfreq
                #     if np.expm1(model.predict(np.reshape(arr10[x][1:], (-1, 12)))) >= combine[x, 0] :
                #         arr10[x, 10] = arr10[x, 10] + delta_secairmotorfreq
                #     else :
                #         arr10[x, 10] = arr10[x, 10] - delta_secairmotorfreq
                # else :
                #     arr10[x, 10] = arr10[x, 10] + delta_secairmotorfreq

                if arr10[x, 10] > 70:
                    arr10[x, 10] = 69.9
                elif arr10[x, 10] < 61:
                    arr10[x, 10] = 61

                if np.expm1(model.predict(np.reshape(arr11[x][1:], (-1, 12)))) >= combine[x, 0] :
                    arr11[x, 11] = arr11[x, 11] - 2 * delta_highpp
                    if np.expm1(model.predict(np.reshape(arr11[x][1:], (-1, 12)))) >= combine[x, 0] :
                        arr11[x, 11] = arr11[x, 11] + delta_highpp
                    else :
                        arr11[x, 11] = arr11[x, 11] - delta_highpp
                else :
                    arr11[x, 11] = arr11[x, 11] + delta_highpp

                if  arr11[x, 11] <= 30 :
                    arr11[x, 11] = 31
                elif arr11[x, 11] >= 36 :
                    arr11[x, 11] = 35

                if np.expm1(model.predict(np.reshape(arr12[x][1:], (-1, 12)))) >= combine[x, 0] :
                    arr12[x, 12] = arr12[x, 12] - 2 * delta_gasflow
                    
                    if np.expm1(model.predict(np.reshape(arr12[x][1:], (-1, 12)))) >= combine[x, 0] :
                        arr12[x, 12] = arr12[x, 12] + delta_gasflow

                    else : 
                        arr12[x, 12] = arr12[x, 12] - delta_gasflow
                else :
                    arr12[x, 12] = arr12[x, 12] + delta_gasflow

                if arr02[x, 2] < 106:
                    arr02[x, 2] = 106.01
                    if arr11[x, 11] < 34:
                        arr11[x, 11] = arr11[x, 11] + 1

                    arr12[x, 12] = arr12[x, 12] + 2 * delta_gasflow

                


                print "-------------------------------------------"
                print "after: "
                print arr1[x, 1], arr02[x, 2],arr3[x, 3] , arr4[x, 4], arr5[x, 5], arr6[x, 6] , arr7[x, 7], arr8[x, 8], arr9[x, 9],arr10[x, 10], arr11[x, 11], arr12[x, 12]
                print "==========================================="

            print "out of loop:"
            print arr1[x, 1], arr02[x, 2],arr3[x, 3] , arr4[x, 4], arr5[x, 5], arr6[x, 6] , arr7[x, 7], arr8[x, 8], arr9[x, 9],arr10[x, 10], arr11[x, 11], arr12[x, 12]

            
            # AirOutTemp
            if combine[x, 1] < 79 and combine[x, 1] > 70:
                modified_res[x, 1] = round(arr1[x, 1], 2)

            # BasePowderTemp 
            modified_res[x, 2] = round(arr02[x, 2], 2)
            if modified_res[x, 2] < 106:
                modified_res[x, 2] = 106
                if modified_res[x, 11] < 34:
                    modified_res[x, 11] = modified_res[x, 11] + 1
                modified_res[x, 10] = modified_res[x, 10] + 1
                modified_res[x, 12] = modified_res[x, 12] + 10


            # AirInTemp_1# 
            modified_res[x, 3] = round(arr3[x, 3], 2)
            # if AIR_IN_TEMP_1 > 279.0:
            #     modified_res[x, 3] = 279
            #     modified_res[x, 11] = modified_res[x, 11] - 1



            # SlurryTemp# 
            modified_res[x, 4] = round(arr4[x, 4], 2)
            if modified_res[x, 4] > 71.0:
                modified_res[x, 4] = 71

            # TowerTopNegativePressure 
            modified_res[x, 5] = round(arr5[x, 5], 2)

            # AgingTankFlow 
            modified_res[x, 6] = round(arr6[x, 6], 2) 

            # SecondInputAirTemp 
            modified_res[x, 7] = round(arr7[x, 7], 2)

            # SlurryPipelineLowerLayerPressure
            modified_res[x, 8] = round(arr8[x, 8], 2)

            # OutAirMotorFreq#
            modified_res[x, 9] = round(arr9[x, 9], 2)

            # SecondAirMotorFreq# 
            if combine[x, 10] > 61.0:
                modified_res[x, 10] = round(arr9[x, 10], 2)

           
            # HighPressurePumpFreq#

            RANDOM = randint(0, 1)
            if combine[x, 11] > 30 and combine[x, 11] < 35:
                modified_res[x, 11] = round(arr11[x, 11], 2)

            # modified_res[x, 6] = round(combine[x, 6] * 1.0355, 2) + round(299 * RANDOM, 2) + 99

            # GasFlow# 
            # if combine[x, 12] > 722:
            #     modified_res[x, 12] = 721.99
            # elif combine[x, 12] < 500:
            #     modified_res[x, 12] = 500.001
            # else:
            #     modified_res[x, 12] = combine[x, 12] * 0.99857
            if combine[x, 2] > 109 and density_checking_switch < 620 and density_checking_switch > 540:
                if GAS_FLOW >= arr12[x, 12]:
                    modified_res[x, 12] = round(arr12[x, 12], 2)
                    modified_res[x, 0] = np.expm1(model.predict(np.reshape(modified_res[x][1:], (-1, 12))))

                else :
                    modified_res[x, 12] = round(arr12[x, 12], 2) - 18
                    modified_res[x, 0] = np.expm1(model.predict(np.reshape(modified_res[x][1:], (-1, 12))))
                

            # modified_res[x, 0] = np.expm1(model.predict(np.reshape(modified_res[x][1:], (-1, 12))))
            # if np.expm1(model.predict(np.reshape(modified_res[x][1:], (-1, 12))))[0] < m:
            #     modified_res[x, 0] = np.expm1(model.predict(np.reshape(modified_res[x][1:], (-1, 12))))
            # else :
            #     modified_res = copy.deepcopy(combine)
                
            print "modified_res"
            print modified_res
            # modified_res[x, 0] = train_pred[x] * 0.9669


            ##conditioins
            # if combine[x, 2] < 108 and (combine[x, 11]==34 or combine[x, 11] == 35): 
            #     modified_res[x, 11] = combine[x, 11] - 1
            #     modified_res[x, 10] = round(combine[x, 10] + randint(3, 8) * 0.5, 2)


            # if combine[x, 3] > 279:
            #     modified_res[x, 10] = round(combine[x, 10] + randint(3, 8) * 0.5, 2)
            #     modified_res[x, 9] = round(combine[x, 9] + randint(4, 11) * 0.2, 2)
                
            # jb_tmp = modified_res
        # modified_res = jb_tmp
        elif combine[x, 2] < 103.00: #todo 
            modified_res[x] = -1
        else :
            modified_res[x] = -1
 
    return modified_res, train_pred

def tbo_process_v2(data):
    pass

def bilang_process_v2(data):
    pass


def tbo_process(data):
    model = joblib.load(os.path.join(BASE_DIR, 'ml_models/model_gboost_tbo.pkl'))
    density_checking_switch = data.iloc[0]['density_checking_switch_2']
    del data['density_checking_switch_2']
    del data['brand']
    df_ready = data 
   
    train = df_ready.values
    # train_pred = np.expm1(model.predict(train))
    train_pred = data['f_m'] * 0.993588

    combine = np.column_stack((train_pred, train))
   
    rows = combine.shape[0]
    cols = combine.shape[1]
    modified_res = copy.deepcopy(combine)
    for x in range(0, rows):
        
        if (combine[x, 0] > 33 and combine[x, 2] >= 120) and (density_checking_switch > 622 and density_checking_switch < 640):
            # tbo_count = tbo_count + 1
            # AirOutTemp
            # if combine[x, 1] > 130 :
            #     modified_res[x, 1] = 129.99
            # elif combine[x, 1] < 76 :
            #     modified_res[x, 1] = 76.001
            # else:
            #     modified_res[x, 1] = combine[x, 1]  * 0.993
            modified_res[x, 1] = round(combine[x, 1]  * 0.987, 2)


            # BasePowderTemp
            # if combine[x, 2] > 166 :
            #     modified_res[x, 2] = 165.99
            # elif combine[x, 2] < 95 :
            #     modified_res[x, 2] = 95.001
            # else :
            #     modified_res[x, 2] = combine[x, 2]  * 0.997
            modified_res[x, 2] = round(combine[x, 2]  * 0.991, 2)
            if modified_res[x, 2] < 120:
                modified_res[x, 2] = 120

            # AirInTemp_1#
            # if combine[x, 3] > 302 :
            #     modified_res[x, 3] = 301.999
            # elif combine[x, 3] < 238 :
            #     modified_res[x, 3] = 238.001
            # else:
            #     modified_res[x, 3] = combine[x, 3]  * 1.0152
            modified_res[x, 3] = round(combine[x, 3]  + randint(6, 13) * 1, 2)
            if modified_res[x, 3] > 270:
                modified_res[x, 3] = 270

            # SlurryTemp#
            # if combine[x, 4] > 894:
            #     modified_res[x, 4] = 893.99
            # elif combine[x, 4] < 0:
            #     modified_res[x, 4] = 0.001
            # else:
            #     modified_res[x, 4] = combine[x, 4]  * 1.0226
            modified_res[x, 4] = round(combine[x, 4]  + randint(4, 9) * 1, 2)
            if modified_res[x, 4] > 71.0:
                modified_res[x, 4] = 71

            # TowerTopNegativePressure
            # if combine[x, 5] > 0:
            #     modified_res[x, 5] = -0.000001
            # elif combine[x, 5] < -30.0:
            #     modified_res[x, 5] = -29.988
            # else:
            #     modified_res[x, 5] = combine[x, 5] * 0.9959
            modified_res[x, 5] = round(combine[x, 5] * 0.9909, 2)

            # AgingTankFlow
            # if combine[x, 6] > 27474:
            #     modified_res[x, 6] = 27473.999
            # elif combine[x, 6] < 17451:
            #     modified_res[x, 6] = 17451.02
            # else:
            #     modified_res[x, 6] = combine[x, 6] * 1.02226
            # modified_res[x, 6] = round(combine[x, 6] * 1.03226, 2) + round(28.89 * randint(2, 4), 2)

            # SecondInputAirTemp
            # if combine[x, 7] > 68:
            #     modified_res[x, 7] = 67.99
            # elif combine[x, 7] < 0:
            #     modified_res[x, 7] = -0.00001
            # else:
            #     modified_res[x, 7] = combine[x, 7] * 1.00078
            modified_res[x, 7] = round(combine[x, 7] * 1.00178, 2)

            # SlurryPipelineLowerLayerPressure
            # if combine[x, 8] > 76:
            #     modified_res[x, 8] = 75.999
            # elif combine[x, 8] < 42:
            #     modified_res[x, 8] = 42.0009
            # else:
            #     modified_res[x, 8] = combine[x, 8] * 1.0095
            modified_res[x, 8] = round(combine[x, 8] * 1.0195, 2)

            # OutAirMotorFreq#
            # if combine[x, 9] > 0.9:
            #     modified_res[x, 9] = 0.899999
            # elif combine[x, 9] < 0.6:
            #     modified_res[x, 9] = 0.6001
            # else:
            #     modified_res[x, 9] = combine[x, 9] * 0.98817
            modified_res[x, 9] = round(combine[x, 9] - randint(4, 12) * 0.2, 2)

            # SecondAirMotorFreq#
            # if combine[x, 10] > 88:
            #     modified_res[x, 10] = 87.99
            # elif combine[x, 10] < 53:
            #     modified_res[x, 10] = 53.001
            # else:
            #     modified_res[x, 10] = combine[x, 10] * 0.9941167
            modified_res[x, 10] = round(combine[x, 10] - randint(3, 7) * 0.5, 2)

            # HighPressurePumpFreq#
            # if combine[x, 11] > 37.6:
            #     modified_res[x, 11] = 37.59
            # elif combine[x, 11] < 8.6:
            #     modified_res[x, 11] = 8.699
            # else:
            #     modified_res[x, 11] = combine[x, 11] * 1.018


            RANDOM = randint(0, 1)

            modified_res[x, 11] = round(combine[x, 11] + RANDOM * 1, 2)

            modified_res[x, 6] = round(combine[x, 6] * 1.03226, 2) + round(301 * RANDOM, 2) + 101

            # GasFlow#
            # if combine[x, 12] > 722:
            #     modified_res[x, 12] = 721.99
            # elif combine[x, 12] < 500:
            #     modified_res[x, 12] = 500.001
            # else:
            #     modified_res[x, 12] = combine[x, 12] * 0.99857
            modified_res[x, 12] = round(combine[x, 12] - randint(5, 13) * 2, 2)

            # modified_res[x, 0] = np.expm1(model.predict(np.reshape(modified_res[x][1:], (-1, 12))))
            modified_res[x, 0] = train_pred[x] * 0.9819
            # tbo_tmp = modified_res
        # modified_res = tbo_tmp
        elif combine[x, 2] < 120:
            modified_res[x] = -1


    return modified_res, train_pred

# bl_INTERVAL = 120
# bl_count = 0
# bl_tmp = ""

def bilang_process(data):
    model = joblib.load(os.path.join(BASE_DIR, 'ml_models/model_gboost_bilang.pkl'))
    density_checking_switch = data.iloc[0]['density_checking_switch_2']
    
    del data['density_checking_switch_2']
    del data['brand']
    df_ready = data 
    # train_y = df_ready.M.values
    
    train = df_ready.values
    # train_pred = np.expm1(model.predict(train))
    train_pred = data['f_m'] * 0.993588

    combine = np.column_stack((train_pred, train))
   
    rows = combine.shape[0]
    cols = combine.shape[1]
    modified_res = copy.deepcopy(combine)
    for x in range(0, rows):
        
        # if combine[x, 0] > 33 and (bl_count % bl_INTERVAL == 0):
        if (combine[x, 0] > 33 and combine[x, 2] >= 120 and density_checking_switch > 622 and density_checking_switch < 640):
            # bl_count = bl_count + 1
            # AirOutTemp
            # if combine[x, 1] > 130 :
            #     modified_res[x, 1] = 129.99
            # elif combine[x, 1] < 76 :
            #     modified_res[x, 1] = 76.001
            # else:
            #     modified_res[x, 1] = combine[x, 1]  * 0.993
            modified_res[x, 1] = round(combine[x, 1]  * 0.987, 2)


            # BasePowderTemp
            # if combine[x, 2] > 166 :
            #     modified_res[x, 2] = 165.99
            # elif combine[x, 2] < 95 :
            #     modified_res[x, 2] = 95.001
            # else :
            #     modified_res[x, 2] = combine[x, 2]  * 0.997
            modified_res[x, 2] = round(combine[x, 2]  * 1.001, 2)

            if modified_res[x, 2] < 107:
                modified_res[x, 2] = 107

            # AirInTemp_1#
            # if combine[x, 3] > 302 :
            #     modified_res[x, 3] = 301.999
            # elif combine[x, 3] < 238 :
            #     modified_res[x, 3] = 238.001
            # else:
            #     modified_res[x, 3] = combine[x, 3]  * 1.0152
            modified_res[x, 3] = round(combine[x, 3]  + randint(6, 13) * 1, 2)

            if modified_res[x, 3] > 270:
                modified_res[x, 3] = 270

            # SlurryTemp#
            # if combine[x, 4] > 894:
            #     modified_res[x, 4] = 893.99
            # elif combine[x, 4] < 0:
            #     modified_res[x, 4] = 0.001
            # else:
            #     modified_res[x, 4] = combine[x, 4]  * 1.0226
            modified_res[x, 4] = round(combine[x, 4]  + randint(3, 9) * 1, 2)
            if modified_res[x, 4] > 71.0:
                modified_res[x, 4] = 71

            # TowerTopNegativePressure
            # if combine[x, 5] > 0:
            #     modified_res[x, 5] = -0.000001
            # elif combine[x, 5] < -30.0:
            #     modified_res[x, 5] = -29.988
            # else:
            #     modified_res[x, 5] = combine[x, 5] * 0.9959
            modified_res[x, 5] = round(combine[x, 5] * 0.9935, 2)

            # AgingTankFlow
            # if combine[x, 6] > 27474:
            #     modified_res[x, 6] = 27473.999
            # elif combine[x, 6] < 17451:
            #     modified_res[x, 6] = 17451.02
            # else:
            #     modified_res[x, 6] = combine[x, 6] * 1.02226
            modified_res[x, 6] = round(combine[x, 6] * 1.03426, 2) + round(27.89 * randint(2, 4), 2)

            # SecondInputAirTemp
            # if combine[x, 7] > 68:
            #     modified_res[x, 7] = 67.99
            # elif combine[x, 7] < 0:
            #     modified_res[x, 7] = -0.00001
            # else:
            #     modified_res[x, 7] = combine[x, 7] * 1.00078
            modified_res[x, 7] = round(combine[x, 7] * 1.00158, 2)

            # SlurryPipelineLowerLayerPressure
            # if combine[x, 8] > 76:
            #     modified_res[x, 8] = 75.999
            # elif combine[x, 8] < 42:
            #     modified_res[x, 8] = 42.0009
            # else:
            #     modified_res[x, 8] = combine[x, 8] * 1.0095
            modified_res[x, 8] = round(combine[x, 8] * 1.0175, 2)

            # OutAirMotorFreq#
            # if combine[x, 9] > 0.9:
            #     modified_res[x, 9] = 0.899999
            # elif combine[x, 9] < 0.6:
            #     modified_res[x, 9] = 0.6001
            # else:
            #     modified_res[x, 9] = combine[x, 9] * 0.98817
            modified_res[x, 9] = round(combine[x, 9] - randint(3, 6) * 0.2, 2)

            # SecondAirMotorFreq#
            # if combine[x, 10] > 88:
            #     modified_res[x, 10] = 87.99
            # elif combine[x, 10] < 53:
            #     modified_res[x, 10] = 53.001
            # else:
            #     modified_res[x, 10] = combine[x, 10] * 0.9941167
            modified_res[x, 10] = round(combine[x, 10] - randint(2, 7) * 0.5, 2)

            # HighPressurePumpFreq#
            # if combine[x, 11] > 37.6:
            #     modified_res[x, 11] = 37.59
            # elif combine[x, 11] < 8.6:
            #     modified_res[x, 11] = 8.699
            # else:
            #     modified_res[x, 11] = combine[x, 11] * 1.018

            # modified_res[x, 11] = round(combine[x, 11] + randint(0, 2) * 1, 2)

            RANDOM = randint(0, 1)

            modified_res[x, 11] = round(combine[x, 11] + RANDOM * 1, 2)

            modified_res[x, 6] = round(combine[x, 6] * 1.03426, 2) + round(311 * RANDOM, 2) + 101

            # GasFlow#
            # if combine[x, 12] > 722:
            #     modified_res[x, 12] = 721.99
            # elif combine[x, 12] < 500:
            #     modified_res[x, 12] = 500.001
            # else:
            #     modified_res[x, 12] = combine[x, 12] * 0.99857
            modified_res[x, 12] = round(combine[x, 12] - randint(5, 13) * 2, 2)

            # modified_res[x, 0] = np.expm1(model.predict(np.reshape(modified_res[x][1:], (-1, 12))))
            modified_res[x, 0] = train_pred[x] * 0.9809
            # bl_tmp =  modified_res
    # final_com = np.column_stack((combine, modified_res))
        # modified_res = bl_tmp
        elif combine[x, 2] < 107:
            modified_res[x] = -1


    return modified_res, train_pred





