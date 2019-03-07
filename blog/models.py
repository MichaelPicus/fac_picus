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
    brand = models.FloatField(default='0')
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

    # flag_aging_tank_flow = models.FloatField(default='0')
    # flag_air_in_temp_1 = models.FloatField(default='0')
    # flag_air_out_temp = models.FloatField(default='0')
    # flag_base_powder_temp = models.FloatField(default='0')
    # flag_gas_flow = models.FloatField(default='0')
    # flag_high_pressure_pump_freq = models.FloatField(default='0')
    # flag_out_air_motor_freq = models.FloatField(default='0')
    # flag_second_air_motor_freq = models.FloatField(default='0')
    # flag_second_input_air_temp = models.FloatField(default='0')
    # flag_slurry_pipeline_lower_layer_pressure = models.FloatField(default='0')
    # flag_slurry_temp = models.FloatField(default='0')
    # flag_tower_top_negative_pressure = models.FloatField(default='0')
    # flag_slurry_density = models.FloatField(default='0')
    # flag_density_checking_switch_1 = models.FloatField(default='0')
    # flag_density_checking_switch_2 = models.FloatField(default='0')

    pred_m = models.FloatField(default='0')
    region = models.CharField(default='chengdu', max_length=100)
    second_air_motor_freq = models.FloatField(default='0')
    second_input_air_temp =models.FloatField(default='0')
    slurry_density = models.FloatField(default='0')
    slurry_pipeline_lower_layer_pressure = models.FloatField(default='0')
    slurry_temp = models.FloatField(default='0')
    tower_top_negative_pressure = models.FloatField(default='0')
    host = models.CharField(default='', max_length=50)

    aging_tank_a_temp = models.FloatField(default='0')
    aging_tank_b_temp = models.FloatField(default='0')
    head_tank_liquid_level_low_setting = models.FloatField(default='0')
    head_tank_liquid_level_high_setting = models.FloatField(default='0')
    sulfate_silo_low_level = models.FloatField(default='0')
    sulfatesilo_high_level = models.FloatField(default='0')
    sulfate_silo_weightlessness_scale_setting = models.FloatField(default='0')
    sulfate_silo_weightlessness_scale_actual = models.FloatField(default='0')
    sulfate_silo_weightlessness_scale_motor_freq = models.FloatField(default='0')
    minor_material_silo_low_level = models.FloatField(default='0')
    minor_material_silo_high_level = models.FloatField(default='0')
    brighter_minor_material_setting = models.FloatField(default='0')
    brighter_minor_material_actual = models.FloatField(default='0')
    brighter_minor_material_motor_freq = models.FloatField(default='0')
    carbonate_silo_high_level = models.FloatField(default='0')
    carbonate_silo_low_level = models.FloatField(default='0')
    carbonate_silo_setting = models.FloatField(default='0')
    carbonate_silo_actual = models.FloatField(default='0')
    carbonate_silo_motor_freq = models.FloatField(default='0')
    hlas_mass_flow_meter_setting = models.FloatField(default='0')
    naoh_mass_flowm_eter_setting = models.FloatField(default='0')
    aging_tank_a_flow = models.FloatField(default='0')
    aging_tank_b_flow = models.FloatField(default='0')
    aging_tank_a_outlet_valve = models.FloatField(default='0')
    aging_tank_b_outlet_valve = models.FloatField(default='0')
    air_in_temp_2 = models.FloatField(default='0')
    high_pressure_pump_a_freq = models.FloatField(default='0')
    high_pressure_pump_b_freq = models.FloatField(default='0')
    las_mass_flow_meter_actual = models.FloatField(default='0')
    las_mass_flow_meter_setting = models.FloatField(default='0')
    rv_base_mass_flow_meter_setting = models.FloatField(default='0')
    rv_base_mass_flow_meter_actual = models.FloatField(default='0')
    ev_base_mass_flow_meter_acutal = models.FloatField(default='0')
    ev_base_mass_flow_meter_setting = models.FloatField(default='0')
    silicate_nass_flow_meter_actual = models.FloatField(default='0')
    silicate_mass_flow_meter_setting = models.FloatField(default='0')
    processed_water_mass_flow_meter_setting = models.FloatField(default='0')
    processed_water_mass_flow_meter_actual = models.FloatField(default='0')
    remelt_water_mass_flow_meter_setting = models.FloatField(default='0')
    remelt_water_mass_flow_meter_actual = models.FloatField(default='0')
    sulfate_silo_high_level_outlet_valve = models.FloatField(default='0')
    sulfate_silo_low_level_outlet_valve = models.FloatField(default='0')
    minor_material_silo_high_level_outlet_valve = models.FloatField(default='0')
    minor_material_silo_low_level_outlet_valve = models.FloatField(default='0')
    carbonate_silo_high_level_outlet_valve = models.FloatField(default='0')
    carbonate_silo_low_level_outlet_valve = models.FloatField(default='0')
    hlas_mass_flow_meter_actual_value = models.FloatField(default='0')
    naoh_mass_flow_meter_actual_value = models.FloatField(default='0')
    slurry_pipeline_upper_layer_pressure = models.FloatField(default='0')
    base_power_flow_setting_value = models.FloatField(default='0')
    base_power_flow_acutal_value = models.FloatField(default='0')
    powder_motor_freq = models.FloatField(default='0')
    slurry_pipe_temp = models.FloatField(default='0')
    sulfate_weight = models.FloatField(default='0')
    carbonate_weight = models.FloatField(default='0')
    brighter_minor_material_weight = models.FloatField(default='0')
    out_air_motor_freq = models.FloatField(default='0')
    air_in_temp_4 = models.FloatField(default='0')
    base_powder_weight = models.FloatField(default='0')
    waste_water_actual = models.FloatField(default='0')
    waste_water_setting = models.FloatField(default='0')
    las_open = models.FloatField(default='0')
    base_powder_open = models.FloatField(default='0')
    steam_flow = models.FloatField(default='0')
    density_checking_switch_1 = models.FloatField(default='0')
    density_checking_switch_2 = models.FloatField(default='0')
    high_pressure_pump_entry_pressure = models.FloatField(default='0')
    high_pressure_pump_entry_flow = models.FloatField(default='0')
    high_pressure_pump_a_freq_new = models.FloatField(default='0')
    high_pressure_pump_b_freq_new = models.FloatField(default='0')
    exhaust_freq_new = models.FloatField(default='0')





    class Meta:
        ordering = ('time',)


class BPT(models.Model):

    base_powder_temp =models.FloatField(default='0')

    class Meta:
        ordering = ('base_powder_temp',)























