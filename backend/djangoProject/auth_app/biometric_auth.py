import os
import torch
import torch.nn as nn
import numpy as np
import pickle
from django.contrib.auth import authenticate
from rest_framework.response import Response
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework_simplejwt.tokens import RefreshToken
from .models import LoginAttempt, BehavioralData

# Model MLP
class BiometricNet(nn.Module):
    def __init__(self, input_dim):
        super(BiometricNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, 2)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x

# Model LSTM
class LSTMNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1):
        super(LSTMNet, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        x = x.unsqueeze(1)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

def prepare_feature_vector(metrics):
    # Łączy listy CZN, CNN, CPN w jeden wektor cech
    vector = np.array(metrics.get("CZN", []) + metrics.get("CNN", []) + metrics.get("CPN", []))
    print("prepare_feature_vector ->", vector)  # Debug
    return vector

@api_view(["POST"])
@permission_classes([AllowAny])
def biometric_login(request):
    username = request.data.get("username")
    password = request.data.get("password")
    user = authenticate(username=username, password=password)

    if user is None:
        print("Błędne dane logowania dla:", username)
        LoginAttempt.objects.create(
            user=None,
            username=username,
            success=False,
            method="biometric",
            details={"error": "Nieprawidłowe dane logowania"}
        )
        return Response({"error": "Nieprawidłowe dane logowania"}, status=401)

    if not user.userprofile.behaviour_security:
        print("Biometria wyłączona - logowanie jako normalne dla użytkownika:", user.username)
        refresh = RefreshToken.for_user(user)
        return Response({
            "access": str(refresh.access_token),
            "refresh": str(refresh),
            "message": "Zalogowano pomyślnie (biometria wyłączona)"
        })

    model_path_mlp = f"models/biometric_model_user_{user.id}.pt"
    model_path_lstm = f"models/biometric_lstm_model_user_{user.id}.pt"
    model_path_knn = f"models/knn_model_user_{user.id}.pkl"
    model_path_svm = f"models/svm_model_user_{user.id}.pkl"
    model_path_rf  = f"models/rf_model_user_{user.id}.pkl"

    if not os.path.exists(model_path_mlp):
        print("Brak modelu biometrycznego dla użytkownika:", user.username)
        refresh = RefreshToken.for_user(user)
        return Response({
            "access": str(refresh.access_token),
            "refresh": str(refresh),
            "message": "Zalogowano pomyślnie (brak modelu biometrycznego)"
        })

    biometric_metrics = request.data.get("biometric_metrics")
    if not biometric_metrics:
        print("Brak danych biometrycznych w żądaniu dla użytkownika:", user.username)
        LoginAttempt.objects.create(
            user=user,
            success=False,
            method="biometric",
            details={"error": "Brak danych biometrycznych"}
        )
        return Response({"error": "Brak danych biometrycznych do weryfikacji"}, status=400)

    login_features = prepare_feature_vector(biometric_metrics)

    checkpoint = torch.load(model_path_mlp, map_location=torch.device("cpu"))
    expected_input_dim = checkpoint["fc1.weight"].shape[1]
    print("Oczekiwany wymiar wejścia:", expected_input_dim)
    print("Długość wektora cech:", login_features.shape[0])

    if login_features.shape[0] < expected_input_dim:
        pad_width = expected_input_dim - login_features.shape[0]
        print(f"Padding: dodaję {pad_width} zer")
        login_features = np.pad(login_features, (0, pad_width), mode="constant", constant_values=0)
    elif login_features.shape[0] > expected_input_dim:
        print("Obcinam wektor do wymiaru:", expected_input_dim)
        login_features = login_features[:expected_input_dim]

    input_tensor = torch.tensor(login_features, dtype=torch.float32).unsqueeze(0)
    predictions = {}

    mlp_model = BiometricNet(expected_input_dim)
    mlp_model.load_state_dict(checkpoint)
    mlp_model.eval()
    mlp_output = mlp_model(input_tensor)
    predictions['MLP'] = torch.argmax(mlp_output, dim=1).item()
    print("MLP predykcja:", predictions['MLP'])

    lstm_model = LSTMNet(expected_input_dim)
    lstm_model.load_state_dict(torch.load(model_path_lstm, map_location=torch.device("cpu")))
    lstm_model.eval()
    lstm_output = lstm_model(input_tensor)
    predictions['LSTM'] = torch.argmax(lstm_output, dim=1).item()
    print("LSTM predykcja:", predictions['LSTM'])

    login_features_reshaped = login_features.reshape(1, -1)
    with open(model_path_knn, "rb") as f:
        knn_model = pickle.load(f)
    predictions['KNN'] = int(knn_model.predict(login_features_reshaped)[0])
    print("KNN predykcja:", predictions['KNN'])

    with open(model_path_svm, "rb") as f:
        svm_model = pickle.load(f)
    predictions['SVM'] = int(svm_model.predict(login_features_reshaped)[0])
    print("SVM predykcja:", predictions['SVM'])

    with open(model_path_rf, "rb") as f:
        rf_model = pickle.load(f)
    predictions['RandomForest'] = int(rf_model.predict(login_features_reshaped)[0])
    print("RandomForest predykcja:", predictions['RandomForest'])

    votes = list(predictions.values())
    count_genuine = sum(votes)
    print("Głosowanie modeli:", votes, "Liczba głosów za dostępem:", count_genuine)

    if count_genuine < 3:
        print("Biometria zablokowała dostęp dla użytkownika:", user.username)
        LoginAttempt.objects.create(
            user=user,
            success=False,
            method="biometric",
            details={"votes": votes, "message": "Biometria zablokowała dostęp"}
        )
        return Response({"error": "Biometria zablokowała dostęp"}, status=403)

    LoginAttempt.objects.create(
        user=user,
        success=True,
        method="biometric",
        details={"votes": votes, "message": "Logowanie udane"}
    )

    if BehavioralData.objects.filter(user=user).count() < 20:
        BehavioralData.objects.create(
            user=user,
            key_events=request.data.get("key_events", []),
            mouse_events=request.data.get("mouseClickData", []),
            biometric_metrics=biometric_metrics,
            session_id=request.data.get("session_id", ""),
            login_attempt_number=request.data.get("login_attempt_number", 1)
        )
    else:
        print("BehavioralData dla użytkownika", user.username, "ma już 20 lub więcej rekordów – nowy rekord nie zostanie dodany.")

    refresh = RefreshToken.for_user(user)
    print("Uwierzytelnienie powiodło się dla użytkownika:", user.username)
    return Response({
        "access": str(refresh.access_token),
        "refresh": str(refresh),
        "message": "Zalogowano pomyślnie"
    })


@api_view(["POST"])
@permission_classes([AllowAny])
def record_login_data(request):
    user = request.user
    key_data = request.data.get("keyData", [])
    mouse_data = request.data.get("mouseClickData", [])
    biometric_metrics = request.data.get("biometric_metrics", {})
    session_id = request.data.get("session_id", "")
    login_attempt_number = request.data.get("login_attempt_number", 1)

    BehavioralData.objects.create(
        user=user,
        key_events=key_data,
        mouse_events=mouse_data,
        biometric_metrics=biometric_metrics,
        session_id=session_id,
        login_attempt_number=login_attempt_number
    )
    LoginAttempt.objects.create(
        user=user,
        success=True,
        method="biometric",
        details={"session_id": session_id, "login_attempt_number": login_attempt_number}
    )
    return Response({"message": "Dane logowania zapisane."}, status=200)

