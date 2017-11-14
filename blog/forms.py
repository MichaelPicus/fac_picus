# -*- coding: utf-8 -*-
from django import forms

class DocumentForm(forms.Form):
    docfile = forms.FileField(
        label='请上传CSV格式的文件',
        help_text='最大不能超过42MB'
    )