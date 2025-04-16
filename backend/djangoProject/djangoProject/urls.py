from django.urls import path, include
from django.contrib import admin
from auth_app.views import CreateUserView, BehavioralDataView
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView
# from auth_app.views import RegisterView, LoginView, LogoutView, profile_view

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/user/register/', CreateUserView.as_view(),name="register"),
    path('api/token/', TokenObtainPairView.as_view(), name='get_token'),
    path('api/token/refresh/', TokenRefreshView.as_view(), name='refresh'),
    path('api-auth/', include('rest_framework.urls')),
    path('api/',include('auth_app.urls')),
    path('api/biometric-data/', BehavioralDataView.as_view(), name='biometric-data'),
]