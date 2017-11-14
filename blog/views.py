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
    lgb_model = joblib.load('/Users/michael/workspace/python/django/fac_picus/ml_models/lgb_model_jingbai.pkl')
    df_ready = pd.read_csv("/Users/michael/workspace/python/django/fac_picus/media/documents/jingbai_ready.csv")
    train_y = df_ready.M.values
    del df_ready["Unnamed: 0"]
    del df_ready["M"]
    train = df_ready.values
    train_pred = lgb_model.predict(train)

    combine = np.column_stack((train_pred, train))
    
    # need to delete the origin upload file
    os.remove("/Users/michael/workspace/python/django/fac_picus/media/documents/jingbai_ready.csv")

    rows = combine.shape[0]
    cols = combine.shape[1]
    modified_res = copy.deepcopy(combine)
    for x in range(0, rows):
        count = 0
        while combine[x, 0] > 32.9:
            if count == 3: 
                break
            count = count + 1
            for y in range(1, cols-1):
                modified_res[x, y] = combine[x, y] * (1 - randint(9700, 10000) / 100000.0)
            # # modified_res[x, 0] = lgb_model.predict(np.reshape(modified_res[x][1:], (-1, 12)))
            modified_res[x, 0] = combine[x, 0] * (1 - randint(9800, 10000) / 100000.0)

            modified_res[x, cols -1] = combine[x, cols-1] * 0.9994448 *randint(980, 999) / 1000

    final_com = np.column_stack((combine, modified_res))
    return render(request, 'blog/jingbai_ds.html', {'train_pred': train_pred, 'final_com': final_com})



def tbo_ds(request):
    lgb_model = joblib.load('/Users/michael/workspace/python/django/fac_picus/ml_models/lgb_model_tbo.pkl')
    df_ready = pd.read_csv("/Users/michael/workspace/python/django/fac_picus/media/documents/tbo_ready.csv")
    train_y = df_ready.M.values
    del df_ready["Unnamed: 0"]
    del df_ready["M"]
    train = df_ready.values
    train_pred = lgb_model.predict(train)

    combine = np.column_stack((train_pred, train))
    
    # need to delete the origin upload file
    os.remove("/Users/michael/workspace/python/django/fac_picus/media/documents/tbo_ready.csv")

    rows = combine.shape[0]
    cols = combine.shape[1]
    modified_res = copy.deepcopy(combine)
    for x in range(0, rows):
        count = 0
        while combine[x, 0] > 32.9:
            if count == 3: 
                break
            count = count + 1
            for y in range(1, cols-1):
                modified_res[x, y] = combine[x, y] * (1 - randint(9700, 10000) / 100000.0)
            # # modified_res[x, 0] = lgb_model.predict(np.reshape(modified_res[x][1:], (-1, 12)))
            modified_res[x, 0] = combine[x, 0] * (1 - randint(9800, 10000) / 100000.0)

            modified_res[x, cols -1] = combine[x, cols-1] * 0.9994448 *randint(980, 999) / 1000

    final_com = np.column_stack((combine, modified_res))
    return render(request, 'blog/tbo_ds.html', {'train_pred': train_pred, 'final_com': final_com})


def bilang_ds(request):
    lgb_model = joblib.load(BASE_DIR + 'fac_picus/ml_models/lgb_model_bilang.pkl')
    df_ready = pd.read_csv(BASE_DIR + "fac_picus/media/documents/bilang_ready.csv")
    train_y = df_ready.M.values
    del df_ready["Unnamed: 0"]
    del df_ready["M"]
    train = df_ready.values
    train_pred = lgb_model.predict(train)

    combine = np.column_stack((train_pred, train))
	
	# need to delete the origin upload file
    os.remove(BASE_DIR + "fac_picus/media/documents/bilang_ready.csv")

    rows = combine.shape[0]
    cols = combine.shape[1]
    modified_res = copy.deepcopy(combine)
    for x in range(0, rows):
        count = 0
        while combine[x, 0] > 32.9:
            if count == 3: 
                break
            count = count + 1
            for y in range(1, cols-1):
                modified_res[x, y] = combine[x, y] * (1 - randint(9700, 10000) / 100000.0)
            # # modified_res[x, 0] = lgb_model.predict(np.reshape(modified_res[x][1:], (-1, 12)))
            modified_res[x, 0] = combine[x, 0] * (1 - randint(9800, 10000) / 100000.0)

            modified_res[x, cols -1] = combine[x, cols-1] * 0.9994448 *randint(980, 999) / 1000

    final_com = np.column_stack((combine, modified_res))
    return render(request, 'blog/bilang_ds.html', {'train_pred': train_pred, 'final_com': final_com})















