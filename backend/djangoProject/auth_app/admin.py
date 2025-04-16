from django.contrib import admin
from .models import BehavioralData, LoginAttempt

@admin.register(BehavioralData)
class BehavioralDataAdmin(admin.ModelAdmin):
    list_display = ('user', 'timestamp', 'session_id', 'login_attempt_number')
    search_fields = ('user__username', 'session_id')
    list_filter = ('timestamp',)

@admin.register(LoginAttempt)
class LoginAttemptAdmin(admin.ModelAdmin):
    list_display = ('user', 'timestamp', 'success', 'method')
    search_fields = ('user__username', 'method')
    list_filter = ('success', 'method', 'timestamp')
