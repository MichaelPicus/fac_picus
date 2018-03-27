from django.db import models
from django.utils import timezone
from django.forms import ModelForm

from pygments.lexers import get_all_lexers
from pygments.styles import get_all_styles


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


class Person(models.Model):
    id = models.AutoField(primary_key=True)
    first_name = models.CharField(max_length=30)
    last_name = models.CharField(max_length=30)



LEXERS = [item for item in get_all_lexers() if item[1]]
LANGUAGE_CHOICES = sorted([(item[1][0], item[0]) for item in LEXERS])
STYLE_CHOICES = sorted((item, item) for item in get_all_styles())


class Snippet(models.Model):
    created = models.DateTimeField(auto_now_add=True)
    title = models.CharField(max_length=100, blank=True, default='')
    code = models.TextField()
    linenos = models.BooleanField(default=False)
    language = models.CharField(choices=LANGUAGE_CHOICES, default='python', max_length=100)
    style = models.CharField(choices=STYLE_CHOICES, default='friendly', max_length=100)

    class Meta:
        ordering = ('created',)


class Valuedata(models.Model):
    time = models.DateTimeField(auto_now_add=True)
    aging_tank_flow = models.FloatField(default='0')
    air_in_temp_1 = models.FloatField(default='0')
    air_out_temp = models.FloatField(default='0')
    base_powder_temp = models.FloatField(default='0')
    brand = models.CharField(default='', max_length=100)
    f_m = models.FloatField(default='0')
    gas_flow = models.FloatField(default='0')
    high_pressure_pump_freq = models.FloatField(default='0')
    modified_m = models.FloatField(default='0')
    out_air_motor_freq = models.FloatField(default='0')
    p_aging_tank_flow = models.FloatField(default='0')
    p_air_in_temp_1 = models.FloatField(default='0')
    p_air_out_temp = models.FloatField(default='0')
    p_base_powder_temp = models.FloatField(default='0')
    p_gas_flow = models.FloatField(default='0')
    p_high_pressure_pump_freq = models.FloatField(default='0')
    p_out_air_motor_freq = models.FloatField(default='0')
    p_second_air_motor_freq = models.FloatField(default='0')
    p_second_input_air_temp = models.FloatField(default='0')
    p_slurry_pipeline_lower_layer_pressure = models.FloatField(default='0')
    p_slurry_temp = models.FloatField(default='0')
    p_tower_top_negative_pressure = models.FloatField(default='0')
    pred_m = models.FloatField(default='0')
    region = models.CharField(default='chengdu', max_length=100)
    second_air_motor_freq = models.FloatField(default='0')
    second_input_air_temp =models.FloatField(default='0')
    slurry_density = models.FloatField(default='0')
    slurry_pipeline_lower_layer_pressure = models.FloatField(default='0')
    slurry_temp = models.FloatField(default='0')
    tower_top_negative_pressure = models.FloatField(default='0')
    host = models.CharField(default='', max_length=50)

    class Meta:
        ordering = ('time',)


























