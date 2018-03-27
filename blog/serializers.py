from rest_framework import serializers
from blog.models import Snippet, LANGUAGE_CHOICES, STYLE_CHOICES, Valuedata


class SnippetSerializer(serializers.ModelSerializer):
    class Meta:
        model = Snippet
        fields = ('id', 'title', 'code', 'linenos', 'language', 'style')



class ValuedataSerializer(serializers.ModelSerializer):
	class Meta:
		model = Valuedata
		fields = ('id', 'aging_tank_flow', 'air_in_temp_1', 'air_out_temp', 'base_powder_temp', 'brand',
			'f_m', 'gas_flow', 'high_pressure_pump_freq', 'modified_m', 'out_air_motor_freq', 'p_aging_tank_flow',
			'p_air_in_temp_1', 'p_air_out_temp', 'p_base_powder_temp', 'p_gas_flow', 'p_high_pressure_pump_freq',
			'p_out_air_motor_freq', 'p_second_air_motor_freq', 'p_second_input_air_temp', 'p_slurry_pipeline_lower_layer_pressure',
			'p_slurry_temp', 'p_tower_top_negative_pressure', 'pred_m', 'region', 'second_air_motor_freq', 'second_input_air_temp',
			'slurry_density', 'slurry_pipeline_lower_layer_pressure', 'slurry_temp', 'tower_top_negative_pressure', 'host')
		