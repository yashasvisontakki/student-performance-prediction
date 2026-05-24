from django.urls import path
from django.contrib.auth import views as auth_views
from . import views

urlpatterns = [
    # Auth
    path('register/', views.register_view,  name='register'),
    path('login/',    views.login_view,     name='login'),
    path('logout/',   views.logout_view,    name='logout'),

    # App
    path('predict/',  views.predict_view,   name='predict'),
    path('history/',  views.history_view,   name='history'),
    path('results/',  views.results_view,   name='results'),
]
