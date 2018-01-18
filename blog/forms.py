# -*- coding: utf-8 -*-
from django import forms

class DocumentForm(forms.Form):
    docfile = forms.FileField(
        label='请上传CSV格式的文件',
        help_text='最大不能超过42MB'
    )


class NameForm(forms.Form):
    your_name = forms.CharField(label='Your name', max_length=100)


class ContactForm(forms.Form):
    subject = forms.CharField(max_length=100)
    message = forms.CharField(widget=forms.Textarea)
    sender = forms.EmailField()
    cc_myself = forms.BooleanField(required=False)


class Jingbai(forms.Form):
	# m = forms.CharField(max_length=20)
	airouttemp = forms.CharField(max_length=20)
	basepowdertemp = forms.CharField(max_length=20)
	airintemp_1 = forms.CharField(max_length=20)
	slurrytemp = forms.CharField(max_length=20)
	towertopnegativepressure = forms.CharField(max_length=20)
	agingtankflow = forms.CharField(max_length=20)
	secondinputairtemp = forms.CharField(max_length=20)
	slurrypipelinelowerlayerpressure = forms.CharField(max_length=20)
	outairmotorfreq = forms.CharField(max_length=20)
	secondairmotorfreq = forms.CharField(max_length=20)
	highpressurepumpfreq = forms.CharField(max_length=20)
	gasflow = forms.CharField(max_length=20)