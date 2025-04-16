from rest_framework import serializers
from django.contrib.auth.models import User
from .models import UserProfile, BehavioralData

class UserProfileSerializer(serializers.ModelSerializer):
    class Meta:
        model = UserProfile
        fields = ['first_name', 'last_name']

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['id', 'username', 'password', 'email', 'first_name', 'last_name']
        extra_kwargs = {"password": {"write_only": True}}

    def create(self, validated_data):
        user = User.objects.create_user(**validated_data)
        return user

class BehavioralDataSerializer(serializers.ModelSerializer):
    class Meta:
        model = BehavioralData
        fields = '__all__'
        read_only_fields = ('user', 'timestamp')