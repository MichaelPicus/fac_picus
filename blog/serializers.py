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
			'slurry_density', 'slurry_pipeline_lower_layer_pressure', 'slurry_temp', 'tower_top_negative_pressure', 'host',
			'aging_tank_a_temp', 'aging_tank_b_temp', 'head_tank_liquid_level_low_setting', 'head_tank_liquid_level_high_setting',
			'sulfate_silo_low_level', 'sulfatesilo_high_level', 'sulfate_silo_weightlessness_scale_setting', 'sulfate_silo_weightlessness_scale_actual',
			'sulfate_silo_weightlessness_scale_motor_freq', 'minor_material_silo_low_level', 'minor_material_silo_high_level', 'brighter_minor_material_setting',
			'brighter_minor_material_actual', 'brighter_minor_material_motor_freq', 'carbonate_silo_high_level', 'carbonate_silo_low_level',
			'carbonate_silo_setting', 'carbonate_silo_actual', 'carbonate_silo_motor_freq', 'hlas_mass_flow_meter_setting', 'naoh_mass_flowm_eter_setting', 
			'aging_tank_a_flow', 'aging_tank_b_flow', 'aging_tank_a_outlet_valve', 'aging_tank_b_outlet_valve','air_in_temp_2',
			'high_pressure_pump_a_freq', 'high_pressure_pump_b_freq', 'las_mass_flow_meter_actual', 'las_mass_flow_meter_setting',
			'rv_base_mass_flow_meter_setting', 'rv_base_mass_flow_meter_actual', 'ev_base_mass_flow_meter_acutal', 'ev_base_mass_flow_meter_setting',
			'silicate_nass_flow_meter_actual', 'silicate_mass_flow_meter_setting', 'processed_water_mass_flow_meter_setting','processed_water_mass_flow_meter_actual',
			'remelt_water_mass_flow_meter_setting', 'remelt_water_mass_flow_meter_actual', 'sulfate_silo_high_level_outlet_valve', 'sulfate_silo_low_level_outlet_valve',
			'minor_material_silo_high_level_outlet_valve', 'minor_material_silo_low_level_outlet_valve', 'carbonate_silo_high_level_outlet_valve', 'carbonate_silo_low_level_outlet_valve',
			'hlas_mass_flow_meter_actual_value', 'naoh_mass_flow_meter_actual_value', 'slurry_pipeline_upper_layer_pressure', 'base_power_flow_setting_value',
			'base_power_flow_acutal_value', 'powder_motor_freq', 'slurry_pipe_temp', 'sulfate_weight', 'carbonate_weight', 'brighter_minor_material_weight', 
			'out_air_motor_freq', 'air_in_temp_4', 'base_powder_weight', 'waste_water_actual', 'waste_water_setting', 'las_open', 'base_powder_open',
			'steam_flow', 'density_checking_switch_1', 'density_checking_switch_2', 'high_pressure_pump_entry_pressure', 'high_pressure_pump_entry_flow', 'high_pressure_pump_a_freq_new', 'high_pressure_pump_b_freq_new', 'exhaust_freq_new',
			# 'flag_aging_tank_flow', 'flag_air_in_temp_1', 'flag_air_out_temp', 'flag_base_powder_temp', 'flag_gas_flow', 'flag_high_pressure_pump_freq',
			# 'flag_out_air_motor_freq', 'flag_second_air_motor_freq', 'flag_second_input_air_temp', 'flag_slurry_pipeline_lower_layer_pressure', 
			# 'flag_slurry_temp', 'flag_tower_top_negative_pressure', 'flag_slurry_density', 'flag_density_checking_switch_1', 'flag_density_checking_switch_2',
			)
		