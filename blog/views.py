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

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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
    lasso_model = joblib.load(os.path.join(BASE_DIR, 'ml_models/lasso_model_jingbai.pkl'))
    df_ready = pd.read_csv(os.path.join(BASE_DIR, 'media/documents/jingbai_ready.csv'))

    train_y = df_ready.M.values
    del df_ready["Unnamed: 0"]
    del df_ready["M"]
    train = df_ready.values
    train_pred = lasso_model.predict(train)

    combine = np.column_stack((train_pred, train))
    
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
            modified_res[x, 2] = combine[x, 2]  * 1.035
            # AirInTemp_1
            modified_res[x, 3] = combine[x, 3]  * 0.973
            # SlurryTemp
            modified_res[x, 4] = combine[x, 4]  * 0.96

            # TowerTopNegativePressure
            modified_res[x, 5] = combine[x, 5] * 1.001
            # AgingTankFlow
            modified_res[x, 6] = combine[x, 6] * 1.028
            # SecondInputAirTemp
            modified_res[x, 7] = combine[x, 7] * 1.05
            # SlurryPipelineLowerLayerPressure
            modified_res[x, 8] = combine[x, 8] * 0.99
            # OutAirMotorFreq
            modified_res[x, 9] = combine[x, 9] * 1.09
            # SecondAirMotorFreq
            modified_res[x, 10] = combine[x, 10] * 0.97
            # HighPressurePumpFreq
            modified_res[x, 11] = combine[x, 11] * 1.034
            # GasFlow
            modified_res[x, 12] = combine[x, 12] * 0.99

            modified_res[x, 0] = lasso_model.predict(np.reshape(modified_res[x][1:], (-1, 12)))

    final_com = np.column_stack((combine, modified_res))
 
    return render(request, 'blog/jingbai_ds.html', {'final_com': final_com})



def tbo_ds(request):
    lasso_model = joblib.load(os.path.join(BASE_DIR, 'ml_models/lasso_model_tbo.pkl'))
    df_ready = pd.read_csv(os.path.join(BASE_DIR, 'media/documents/tbo_ready.csv'))
    train_y = df_ready.M.values
    del df_ready["Unnamed: 0"]
    del df_ready["M"]
    train = df_ready.values
    train_pred = lasso_model.predict(train)

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
        if combine[x, 0] > 33 :
            # AirOutTemp
            modified_res[x, 1] = combine[x, 1]  * 0.98
            # BasePowderTemp
            modified_res[x, 2] = combine[x, 2]  * 1.035
            # AirInTemp_1
            modified_res[x, 3] = combine[x, 3]  * 0.973
            # SlurryTemp
            modified_res[x, 4] = combine[x, 4]  * 0.96

            # TowerTopNegativePressure
            modified_res[x, 5] = combine[x, 5] * 1.001
            # AgingTankFlow
            modified_res[x, 6] = combine[x, 6] * 1.028
            # SecondInputAirTemp
            modified_res[x, 7] = combine[x, 7] * 1.05
            # SlurryPipelineLowerLayerPressure
            modified_res[x, 8] = combine[x, 8] * 0.99
            # OutAirMotorFreq
            modified_res[x, 9] = combine[x, 9] * 1.09
            # SecondAirMotorFreq
            modified_res[x, 10] = combine[x, 10] * 0.97
            # HighPressurePumpFreq
            modified_res[x, 11] = combine[x, 11] * 1.034
            # GasFlow
            modified_res[x, 12] = combine[x, 12] * 0.99

            modified_res[x, 0] = lasso_model.predict(np.reshape(modified_res[x][1:], (-1, 12)))

    final_com = np.column_stack((combine, modified_res))
    return render(request, 'blog/tbo_ds.html', {'final_com': final_com})


def bilang_ds(request):
    lasso_model = joblib.load(os.path.join(BASE_DIR, 'ml_models/lasso_model_bilang.pkl'))
    df_ready = pd.read_csv(os.path.join(BASE_DIR, 'media/documents/bilang_ready.csv')) 
    train_y = df_ready.M.values
    del df_ready["Unnamed: 0"]
    del df_ready["M"]
    train = df_ready.values
    train_pred = lasso_model.predict(train)

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
            modified_res[x, 2] = combine[x, 2]  * 1.035
            # AirInTemp_1
            modified_res[x, 3] = combine[x, 3]  * 0.973
            # SlurryTemp
            modified_res[x, 4] = combine[x, 4]  * 0.96

            # TowerTopNegativePressure
            modified_res[x, 5] = combine[x, 5] * 1.001
            # AgingTankFlow
            modified_res[x, 6] = combine[x, 6] * 1.028
            # SecondInputAirTemp
            modified_res[x, 7] = combine[x, 7] * 1.05
            # SlurryPipelineLowerLayerPressure
            modified_res[x, 8] = combine[x, 8] * 0.99
            # OutAirMotorFreq
            modified_res[x, 9] = combine[x, 9] * 1.09
            # SecondAirMotorFreq
            modified_res[x, 10] = combine[x, 10] * 0.97
            # HighPressurePumpFreq
            modified_res[x, 11] = combine[x, 11] * 1.034
            # GasFlow
            modified_res[x, 12] = combine[x, 12] * 0.99

            modified_res[x, 0] = lasso_model.predict(np.reshape(modified_res[x][1:], (-1, 12)))

    final_com = np.column_stack((combine, modified_res))
    return render(request, 'blog/bilang_ds.html', {'final_com': final_com})















