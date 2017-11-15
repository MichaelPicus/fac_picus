from django.conf.urls import url
from . import views


urlpatterns = [
    url(r'^$', views.post_list, name='post_list'),
    url(r'^jingbai$', views.jingbai, name='jingbai'),
    url(r'^tbo$', views.tbo, name='tbo'),
    url(r'^bilang$', views.bilang, name='bilang'),
    url(r'^list/$', views.list, name='list'),
    url(r'^jingbai_ds$', views.jingbai_ds, name='jingbai_ds'),
    url(r'^tbo_ds$', views.tbo_ds, name='tbo_ds'),
    url(r'^bilang_ds$', views.bilang_ds, name='bilang_ds'),
    url(r'^delete_bilang$', views.delete_bilang, name='delete_bilang'),
    url(r'^delete_tbo$', views.delete_tbo, name='delete_tbo'),
    url(r'^delete_jingbai$', views.delete_jingbai, name='delete_jingbai'),

    # url(r'^post/(?P<pk>\d+)/$', views.post_detail, name='post_detail'),
]

