from django.contrib.auth.models import User
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework import generics, status
from rest_framework.permissions import IsAuthenticated, AllowAny
from .biometric_model_trainer import train_biometric_model
from .serializers import UserSerializer, BehavioralDataSerializer
from .models import BehavioralData, LoginAttempt
from .biometric_auth import biometric_login

class CreateUserView(generics.CreateAPIView):
    queryset = User.objects.all()
    serializer_class = UserSerializer
    permission_classes = [AllowAny]

class UserProfileView(APIView):
    permission_classes = [IsAuthenticated]
    def get(self, request):
        user = request.user
        return Response({
            "username": user.username,
            "behaviour_security": user.userprofile.behaviour_security
        })

class UpdateUserProfileView(APIView):
    permission_classes = [IsAuthenticated]
    def patch(self, request):
        user_profile = request.user.userprofile
        behaviour_security = request.data.get("behaviour_security")
        if behaviour_security is None:
            return Response({"error": "Brak wartości behaviour_security."}, status=status.HTTP_400_BAD_REQUEST)
        user_profile.behaviour_security = behaviour_security
        user_profile.save()
        return Response({"message": "Profil zaktualizowany pomyślnie."}, status=status.HTTP_200_OK)

class SaveBiometricDataView(APIView):
    permission_classes = [IsAuthenticated]
    def post(self, request):
        user = request.user
        key_data = request.data.get("keyData", [])
        mouse_data = request.data.get("mouseClickData", [])
        print(f"User: {user.username}, Key Data: {key_data}, Mouse Data: {mouse_data}")
        return Response({"message": "Data logged successfully."}, status=200)

class BehavioralDataView(APIView):
    permission_classes = [IsAuthenticated]
    def post(self, request):
        serializer = BehavioralDataSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save(user=request.user)
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class DeleteBiometricDataView(APIView):
    permission_classes = [IsAuthenticated]
    def delete(self, request):
        user = request.user
        deleted_count, _ = BehavioralData.objects.filter(user=user).delete()
        return Response({"message": f"Usunięto {deleted_count} wpisów danych biometrycznych."}, status=status.HTTP_200_OK)

class TrainBiometricModelView(APIView):
    permission_classes = [IsAuthenticated]
    def post(self, request):
        user = request.user
        result = train_biometric_model(user)
        if result is None or "error" in result:
            return Response(result or {"error": "Wystąpił błąd podczas treningu modelu."}, status=status.HTTP_400_BAD_REQUEST)
        return Response(result, status=status.HTTP_200_OK)

class LoginAttemptsView(APIView):
    permission_classes = [IsAuthenticated]
    def get(self, request):
        attempts = LoginAttempt.objects.filter(user=request.user).order_by("-timestamp")
        data = []
        for att in attempts:
            data.append({
                "timestamp": att.timestamp,
                "success": att.success,
                "method": att.method,
                "details": att.details,
            })
        return Response(data)

class CountBiometricDataView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        count = BehavioralData.objects.filter(user=request.user).count()
        return Response({"count": count})