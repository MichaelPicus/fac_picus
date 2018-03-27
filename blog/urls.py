from django.conf.urls import url
from . import views

urlpatterns = [
    url(r'^signup/$', views.SignUpView.as_view(), name='signup'),
]




urlpatterns = [
    url(r'^$', views.post_list, name='post_list'),
    url(r'^jingbai$', views.jingbai, name='jingbai'),
    url(r'^tbo$', views.tbo, name='tbo'),
    url(r'^bilang$', views.bilang, name='bilang'),
    url(r'^bilang_two_vars$', views.bilang_two_vars, name='bilang_two_vars'),
    url(r'^bilang_gasflow_highppf$', views.bilang_gasflow_highppf, name='bilang_gasflow_highppf'),
    url(r'^list/$', views.list, name='list'),
    url(r'^jingbai_ds$', views.jingbai_ds, name='jingbai_ds'),
    url(r'^tbo_ds$', views.tbo_ds, name='tbo_ds'),
    url(r'^tbo_two_vars$', views.tbo_two_vars, name='tbo_two_vars'),
    url(r'^tbo_gasflow_highppf$', views.tbo_gasflow_highppf, name='tbo_gasflow_highppf'),
    url(r'^bilang_ds$', views.bilang_ds, name='bilang_ds'),
    url(r'^jingbai_two_vars$', views.jingbai_two_vars, name='jingbai_two_vars'),
    url(r'^jingbai_gasflow_highppf$', views.jingbai_gasflow_highppf, name='jingbai_gasflow_highppf'),
    url(r'^delete_bilang$', views.delete_bilang, name='delete_bilang'),
    url(r'^delete_tbo$', views.delete_tbo, name='delete_tbo'),
    url(r'^delete_jingbai$', views.delete_jingbai, name='delete_jingbai'),


    url(r'^signup/$', views.SignUpView.as_view(), name='signup'),
    url(r'^ajax/validate_username/$', views.validate_username, name='validate_username'),
    url(r'^get_name/$', views.get_name, name='get_name'),
    url(r'^thanks/$', views.thanks, name='thanks'),
    url(r'^process/$', views.process, name='process'),
    url(r'^snippets/$', views.snippet_list),
    url(r'^snippets/(?P<pk>[0-9]+)$', views.snippet_detail),
    url(r'^valuedata/$', views.value_data_process),

    # url(r'^post/(?P<pk>\d+)/$', views.post_detail, name='post_detail'),
]

