from django.urls import path
from . import views
from .biometric_auth import biometric_login

urlpatterns = [
    path("profile/", views.UserProfileView.as_view(), name="user-profile"),
    path("profile/update/", views.UpdateUserProfileView.as_view(), name="update-profile"),
    path("save-biometrics/", views.SaveBiometricDataView.as_view(), name="save-biometrics"),
    path("biometric-data/delete-all/", views.DeleteBiometricDataView.as_view(), name="delete-all-biometric-data"),
    path("biometric-model/train/", views.TrainBiometricModelView.as_view(), name="train-biometric-model"),
    path("biometric-login/", biometric_login, name="biometric-login"),
    path("login-attempts/", views.LoginAttemptsView.as_view(), name="login-attempts"),
    path("biometric-data/count/", views.CountBiometricDataView.as_view(), name="biometric-data-count"),
]
