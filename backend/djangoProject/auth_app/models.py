from django.contrib.auth.models import User
from django.db import models
from django.db.models.signals import post_save
from django.dispatch import receiver

class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    phone = models.CharField(max_length=15, blank=True, null=True)
    behaviour_security = models.BooleanField(default=False)

    def __str__(self):
        return self.user.username

@receiver(post_save, sender=User)
def create_user_profile(sender, instance, created, **kwargs):
    if created:
        if not hasattr(instance, 'userprofile'):
            UserProfile.objects.create(user=instance)

@receiver(post_save, sender=User)
def save_user_profile(sender, instance, **kwargs):
    if hasattr(instance, 'userprofile'):
        instance.userprofile.save()


class BehavioralData(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    timestamp = models.DateTimeField(auto_now_add=True)
    key_events = models.JSONField()
    mouse_events = models.JSONField()
    biometric_metrics = models.JSONField(blank=True, null=True)
    session_id = models.CharField(max_length=100)
    login_attempt_number = models.IntegerField()

    class Meta:
        ordering = ['-timestamp']

class LoginAttempt(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    timestamp = models.DateTimeField(auto_now_add=True)
    success = models.BooleanField()
    method = models.CharField(max_length=50)
    details = models.JSONField(blank=True, null=True)

    def __str__(self):
        return f"{self.user.username if self.user else 'Anon'} - {self.method} - {self.success}"

