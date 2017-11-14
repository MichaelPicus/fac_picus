from django.db import models
from django.utils import timezone
from django.forms import ModelForm


# Create your models here.
class Post(models.Model):
    author = models.ForeignKey('auth.User')
    title = models.CharField(max_length=200)
    text = models.TextField()
    created_date = models.DateTimeField(
            default=timezone.now)
    published_date = models.DateTimeField(
            blank=True, null=True)

    def publish(self):
        self.published_date = timezone.now()
        self.save()

    def __str__(self):
        return self.title


# class Upload(models.Model):
#     pic = models.FileField(upload_to="images/")    
#     upload_date=models.DateTimeField(auto_now_add =True)

# # FileUpload form class.
# class UploadForm(ModelForm):
#     class Meta:
#         model = Upload
#         fields = ('pic',)


class Document(models.Model):
    docfile = models.FileField(upload_to='documents')

