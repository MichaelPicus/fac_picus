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
            }

            data = pd.DataFrame(data=d, columns=['air_out_temp', 'base_powder_temp', 'air_in_temp_1', 'slurry_temp', 'tower_top_negative_pressure',
                    'aging_tank_flow', 'second_input_air_temp', 'slurry_pipeline_lower_layer_pressure', 
                    'out_air_motor_freq', 'second_air_motor_freq', 'high_pressure_pump_freq', 'gas_flow', 'brand'])
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
                        "air_out_temp": float(serializer.data.get("air_out_temp")),
                        "base_powder_temp": float(serializer.data.get("base_powder_temp")),
                        "air_in_temp_1": float(serializer.data.get("air_in_temp_1")),
                        "slurry_temp": float(serializer.data.get("slurry_temp")),
                        "tower_top_negative_pressure": float(serializer.data.get("tower_top_negative_pressure")),
                        "aging_tank_flow": float(serializer.data.get("aging_tank_flow")),
                        "second_input_air_temp": float(serializer.data.get("second_input_air_temp")),
                        "slurry_pipeline_lower_layer_pressure":float(serializer.data.get("slurry_pipeline_lower_layer_pressure")),
                        "out_air_motor_freq": float(serializer.data.get("out_air_motor_freq")),
                        "second_air_motor_freq": float(serializer.data.get("second_air_motor_freq")),
                        "high_pressure_pump_freq": float(serializer.data.get("high_pressure_pump_freq")),
                        "gas_flow": float(serializer.data.get("gas_flow")),
                        "p_air_out_temp": float(res[0][1]),
                        "p_base_powder_temp": float(res[0][2]),
                        "p_air_in_temp_1": float(res[0][3]),
                        "p_slurry_temp": float(res[0][4]),
                        "p_tower_top_negative_pressure": float(res[0][5]),
                        "p_aging_tank_flow": float(res[0][6]),
                        "p_second_input_air_temp": float(res[0][7]),
                        "p_slurry_pipeline_lower_layer_pressure": float(res[0][8]),
                        "p_out_air_motor_freq": float(res[0][9]),
                        "p_second_air_motor_freq": float(res[0][10]),
                        "p_high_pressure_pump_freq": float(res[0][11]),
                        "p_gas_flow": float(res[0][12]),
                        "f_m" : float(serializer.data.get("f_m")),
                        "pred_m" : float(pred_m[0]),
                        "modified_m" :float(res[0][0]),
                        "slurry_density" : float(serializer.data.get("slurry_density")),
                        "host": serializer.data.get("host"),


                        "aging_tank_a_temp" : float(serializer.data.get("aging_tank_a_temp")),
                        "aging_tank_b_temp" : float(serializer.data.get("aging_tank_b_temp")),
                        "head_tank_liquid_level_low_setting" : float(serializer.data.get("head_tank_liquid_level_low_setting")),
                        "head_tank_liquid_level_high_setting" : float(serializer.data.get("head_tank_liquid_level_high_setting")),
                        "sulfate_silo_low_level" : float(serializer.data.get("sulfate_silo_low_level")),
                        "sulfatesilo_high_level" : float(serializer.data.get("sulfatesilo_high_level")),
                        "sulfate_silo_weightlessness_scale_setting" : float(serializer.data.get("sulfate_silo_weightlessness_scale_setting")),
                        "sulfate_silo_weightlessness_scale_actual" : float(serializer.data.get("sulfate_silo_weightlessness_scale_actual")),
                        "sulfate_silo_weightlessness_scale_motor_freq" : float(serializer.data.get("sulfate_silo_weightlessness_scale_motor_freq")),
                        "minor_material_silo_low_level" : float(serializer.data.get("minor_material_silo_low_level")),
                        "minor_material_silo_high_level" : float(serializer.data.get("minor_material_silo_high_level")),
                        "brighter_minor_material_setting" : float(serializer.data.get("brighter_minor_material_setting")),
                        "brighter_minor_material_actual" : float(serializer.data.get("brighter_minor_material_actual")),
                        "brighter_minor_material_motor_freq" : float(serializer.data.get("brighter_minor_material_motor_freq")),
                        "carbonate_silo_high_level" : float(serializer.data.get("carbonate_silo_high_level")),
                        "carbonate_silo_low_level" : float(serializer.data.get("carbonate_silo_low_level")),
                        "carbonate_silo_setting" : float(serializer.data.get("carbonate_silo_setting")),
                        "carbonate_silo_actual" : float(serializer.data.get("carbonate_silo_actual")),
                        "carbonate_silo_motor_freq" : float(serializer.data.get("carbonate_silo_motor_freq")),

                        "hlas_mass_flow_meter_setting" : float(serializer.data.get("hlas_mass_flow_meter_setting")),
                        "naoh_mass_flowm_eter_setting" : float(serializer.data.get("naoh_mass_flowm_eter_setting")),
                        "aging_tank_a_flow" : float(serializer.data.get("aging_tank_a_flow")),
                        "aging_tank_b_flow" : float(serializer.data.get("aging_tank_b_flow")),
                        "aging_tank_a_outlet_valve" : float(serializer.data.get("aging_tank_a_outlet_valve")),
                        "aging_tank_b_outlet_valve" : float(serializer.data.get("aging_tank_b_outlet_valve")),
                        "air_in_temp_2" : float(serializer.data.get("air_in_temp_2")),
                        "high_pressure_pump_a_freq" : float(serializer.data.get("high_pressure_pump_a_freq")),
                        "high_pressure_pump_b_freq" : float(serializer.data.get("high_pressure_pump_b_freq")),
                        "las_mass_flow_meter_actual" : float(serializer.data.get("las_mass_flow_meter_actual")),

                        "las_mass_flow_meter_setting" : float(serializer.data.get("las_mass_flow_meter_setting")),
                        "rv_base_mass_flow_meter_setting" : float(serializer.data.get("rv_base_mass_flow_meter_setting")),
                        "rv_base_mass_flow_meter_actual" : float(serializer.data.get("rv_base_mass_flow_meter_actual")),
                        "ev_base_mass_flow_meter_acutal" : float(serializer.data.get("ev_base_mass_flow_meter_acutal")),
                        "ev_base_mass_flow_meter_setting" : float(serializer.data.get("ev_base_mass_flow_meter_setting")),
                        "silicate_nass_flow_meter_actual" : float(serializer.data.get("silicate_nass_flow_meter_actual")),
                        "silicate_mass_flow_meter_setting" : float(serializer.data.get("silicate_mass_flow_meter_setting")),
                        "processed_water_mass_flow_meter_setting" : float(serializer.data.get("processed_water_mass_flow_meter_setting")),
                        "processed_water_mass_flow_meter_actual" : float(serializer.data.get("processed_water_mass_flow_meter_actual")),
                        "remelt_water_mass_flow_meter_setting" : float(serializer.data.get("remelt_water_mass_flow_meter_setting")),

                        "remelt_water_mass_flow_meter_actual" : float(serializer.data.get("remelt_water_mass_flow_meter_actual")),
                        "sulfate_silo_high_level_outlet_valve" : float(serializer.data.get("sulfate_silo_high_level_outlet_valve")),
                        "sulfate_silo_low_level_outlet_valve" : float(serializer.data.get("sulfate_silo_low_level_outlet_valve")),
                        "minor_material_silo_high_level_outlet_valve" : float(serializer.data.get("minor_material_silo_high_level_outlet_valve")),
                        "minor_material_silo_low_level_outlet_valve" : float(serializer.data.get("minor_material_silo_low_level_outlet_valve")),
                        "carbonate_silo_high_level_outlet_valve" : float(serializer.data.get("carbonate_silo_high_level_outlet_valve")),
                        "carbonate_silo_low_level_outlet_valve" : float(serializer.data.get("carbonate_silo_low_level_outlet_valve")),
                        "hlas_mass_flow_meter_actual_value" : float(serializer.data.get("hlas_mass_flow_meter_actual_value")),
                        "naoh_mass_flow_meter_actual_value" : float(serializer.data.get("naoh_mass_flow_meter_actual_value")),
                        "slurry_pipeline_upper_layer_pressure" : float(serializer.data.get("slurry_pipeline_upper_layer_pressure")),

                        "base_power_flow_setting_value" : float(serializer.data.get("base_power_flow_setting_value")),
                        "base_power_flow_acutal_value" : float(serializer.data.get("base_power_flow_acutal_value")),
                        "powder_motor_freq" : float(serializer.data.get("powder_motor_freq")),
                        "slurry_pipe_temp" : float(serializer.data.get("slurry_pipe_temp")),
                        "sulfate_weight" : float(serializer.data.get("sulfate_weight")),
                        "carbonate_weight" : float(serializer.data.get("carbonate_weight")),
                        "brighter_minor_material_weight" : float(serializer.data.get("brighter_minor_material_weight")),
                        "out_air_motor_freq" : float(serializer.data.get("out_air_motor_freq")),
                        "air_in_temp_4" : float(serializer.data.get("air_in_temp_4")),
                        "base_powder_weight" : float(serializer.data.get("base_powder_weight")),

                        "waste_water_actual" : float(serializer.data.get("waste_water_actual")),
                        "waste_water_setting" : float(serializer.data.get("waste_water_setting")),
                        "las_open" : float(serializer.data.get("las_open")),
                        "base_powder_open" : float(serializer.data.get("base_powder_open")),
                        "steam_flow" : float(serializer.data.get("steam_flow")),
                        "density_checking_switch_1" : float(serializer.data.get("density_checking_switch_1")),
                        "density_checking_switch_2" : float(serializer.data.get("density_checking_switch_2")),
                        "high_pressure_pump_entry_pressure" : float(serializer.data.get("high_pressure_pump_entry_pressure")),
                        "high_pressure_pump_entry_flow" : float(serializer.data.get("high_pressure_pump_entry_flow")),
                        "high_pressure_pump_a_freq_new" : float(serializer.data.get("high_pressure_pump_a_freq_new")),

                        "high_pressure_pump_b_freq_new" : float(serializer.data.get("high_pressure_pump_b_freq_new")),
                        "exhaust_freq_new" : float(serializer.data.get("exhaust_freq_new")),
                        
                    }
                }
            ]
            
            print json_body
            con.write_points(json_body)
            print "post sucessfully!"
            con.close()
            return Response(serializer.data, status=status.HTTP_201_CREATED)


    elif request.method == 'GET':
        pass

    else:
        print "post failure!!!"
        return Response("error! sorry!")


def data_process(data):
    if data['brand'][0] == 2.0:
                print "processing jingbai!"
                res, pred_m = jingbai_process(data)
    elif data['brand'][0] == 3.0:
                print "processing bilang!"
                res, pred_m = bilang_process(data)
    else:
                print "processing tbo or others!"
                res, pred_m = tbo_process(data)
    return res, pred_m



def jingbai_process(data):
    model = joblib.load(os.path.join(BASE_DIR, 'ml_models/model_gboost_jingbai.pkl'))
    del data['brand']
    df_ready = data 
   
    train = df_ready.values
    train_pred = np.expm1(model.predict(train))
    print train_pred
    combine = np.column_stack((train_pred, train))
   
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
         
    return modified_res, train_pred
def tbo_process(data):
    model = joblib.load(os.path.join(BASE_DIR, 'ml_models/model_gboost_tbo.pkl'))
    del data['brand']
    df_ready = data 
   
    train = df_ready.values
    train_pred = np.expm1(model.predict(train))

    combine = np.column_stack((train_pred, train))
   
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
            
    return modified_res, train_pred

def bilang_process(data):
    model = joblib.load(os.path.join(BASE_DIR, 'ml_models/model_gboost_bilang.pkl'))
    del data['brand']
    df_ready = data 
    # train_y = df_ready.M.values
    
    train = df_ready.values
    train_pred = np.expm1(model.predict(train))

    combine = np.column_stack((train_pred, train))
   
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

    # final_com = np.column_stack((combine, modified_res))

    return modified_res, train_pred





