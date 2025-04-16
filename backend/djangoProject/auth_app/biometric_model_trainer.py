import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")

from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, roc_curve
from sklearn.model_selection import train_test_split
from .models import BehavioralData
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import pickle

def prepare_feature_vector(metrics):
    #wektor cech
    vector = np.array(metrics.get("CZN", []) + metrics.get("CNN", []) + metrics.get("CPN", []))
    print("prepare_feature_vector ->", vector)
    return vector

def evaluate_model_pt(model, X, y, dataset_name=""):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32)
        outputs = model(X_tensor)
        _, y_pred_tensor = torch.max(outputs, dim=1)
        y_pred = y_pred_tensor.numpy()
        y_proba = torch.softmax(outputs, dim=1).numpy()
    f1 = f1_score(y, y_pred)
    precision = precision_score(y, y_pred)
    sensitivity = recall_score(y, y_pred)
    auc_val = roc_auc_score(y, y_proba[:, 1])
    print(f"{dataset_name} - F1: {f1}, Precyzja: {precision}, Czułość: {sensitivity}, AUC: {auc_val}")
    return f1, precision, sensitivity, auc_val

def train_biometric_model(user, n_fake=35, epochs=50, batch_size=8, lr=0.001):
    qs = BehavioralData.objects.filter(user=user)
    print("Ilość rekordów dla użytkownika:", qs.count())
    if qs.count() < 20:
        return {"error": "Niewystarczająca liczba prób (minimum 20 wymagane)."}

    genuine_features = []
    for record in qs:
        metrics = record.biometric_metrics
        if metrics:
            vec = prepare_feature_vector(metrics)
            if vec.size > 0:
                genuine_features.append(vec)

    print("Liczba prób genuine:", len(genuine_features))
    if len(genuine_features) < 20:
        return {"error": "Brak wystarczających danych po ekstrakcji."}


    max_len = max(len(vec) for vec in genuine_features)
    print("Maksymalna długość wektora:", max_len)
    genuine_features_padded = [
        np.pad(vec, (0, max_len - len(vec)), mode='constant', constant_values=0)
        for vec in genuine_features
    ]
    X_genuine = np.array(genuine_features_padded)
    y_genuine = np.ones(X_genuine.shape[0])
    print("Kształt genuine data:", X_genuine.shape)


    means = X_genuine.mean(axis=0)
    stds = X_genuine.std(axis=0)
    stds[stds == 0] = 1e-4


    n_fake_lower = n_fake // 2
    n_fake_higher = n_fake - n_fake_lower
    shift_factor = 1.5
    X_fake_lower = np.random.normal(
        loc=means - shift_factor * stds,
        scale=stds,
        size=(n_fake_lower, X_genuine.shape[1])
    )
    X_fake_higher = np.random.normal(
        loc=means + shift_factor * stds,
        scale=stds,
        size=(n_fake_higher, X_genuine.shape[1])
    )
    X_fake = np.vstack([X_fake_lower, X_fake_higher])
    y_fake = np.zeros(n_fake)
    print("Kształt fake data:", X_fake.shape)


    X_all = np.vstack([X_genuine, X_fake])
    y_all = np.concatenate([y_genuine, y_fake])
    print("Łączny zbiór (X):", X_all.shape, "Łączny zbiór (y):", y_all.shape)


    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val
    )
    print("Treningowy:", X_train.shape, y_train.shape)
    print("Walidacyjny:", X_val.shape, y_val.shape)
    print("Testowy:", X_test.shape, y_test.shape)


    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    input_dim = X_all.shape[1]
    print("Wymiar wejściowy modelu:", input_dim)


    class BiometricNet(nn.Module):
        def __init__(self, input_dim):
            super(BiometricNet, self).__init__()
            self.fc1 = nn.Linear(input_dim, 64)
            self.relu1 = nn.ReLU()
            self.fc2 = nn.Linear(64, 32)
            self.relu2 = nn.ReLU()
            self.fc3 = nn.Linear(32, 2)  # 2 klasy: genuine (1) i fake (0)

        def forward(self, x):
            x = self.relu1(self.fc1(x))
            x = self.relu2(self.fc2(x))
            x = self.fc3(x)
            return x

    mlp_model = BiometricNet(input_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer_mlp = optim.Adam(mlp_model.parameters(), lr=lr)

    print("\nTrening modelu MLP:")
    for epoch in range(epochs):
        mlp_model.train()
        epoch_loss = 0.0
        for inputs, labels in train_loader:
            optimizer_mlp.zero_grad()
            outputs = mlp_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_mlp.step()
            epoch_loss += loss.item() * inputs.size(0)
        epoch_loss /= len(train_dataset)
        print(f"MLP Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

    os.makedirs("models", exist_ok=True)
    model_path_mlp = f"models/biometric_model_user_{user.id}.pt"
    torch.save(mlp_model.state_dict(), model_path_mlp)
    print("MLP zapisany pod:", model_path_mlp)
    metrics_mlp = evaluate_model_pt(mlp_model, X_test, y_test, "MLP - Testowy")


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

    lstm_model = LSTMNet(input_dim)
    optimizer_lstm = optim.Adam(lstm_model.parameters(), lr=lr)

    print("\nTrening modelu LSTM:")
    for epoch in range(epochs):
        lstm_model.train()
        epoch_loss = 0.0
        for inputs, labels in train_loader:
            optimizer_lstm.zero_grad()
            outputs = lstm_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_lstm.step()
            epoch_loss += loss.item() * inputs.size(0)
        epoch_loss /= len(train_dataset)
        print(f"LSTM Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

    model_path_lstm = f"models/biometric_lstm_model_user_{user.id}.pt"
    torch.save(lstm_model.state_dict(), model_path_lstm)
    print("LSTM zapisany pod:", model_path_lstm)
    metrics_lstm = evaluate_model_pt(lstm_model, X_test, y_test, "LSTM - Testowy")


    print("\nTrenowanie modeli klasycznych:")

    # KNN
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train, y_train)
    y_pred_knn = knn_model.predict(X_test)
    y_proba_knn = knn_model.predict_proba(X_test)[:, 1]
    f1_knn = f1_score(y_test, y_pred_knn)
    precision_knn = precision_score(y_test, y_pred_knn)
    recall_knn = recall_score(y_test, y_pred_knn)
    auc_knn = roc_auc_score(y_test, y_proba_knn)
    print(f"KNN - Testowy: F1: {f1_knn}, Precyzja: {precision_knn}, Czułość: {recall_knn}, AUC: {auc_knn}")

    svm_model = SVC(probability=True)
    svm_model.fit(X_train, y_train)
    y_pred_svm = svm_model.predict(X_test)
    y_proba_svm = svm_model.predict_proba(X_test)[:, 1]
    f1_svm = f1_score(y_test, y_pred_svm)
    precision_svm = precision_score(y_test, y_pred_svm)
    recall_svm = recall_score(y_test, y_pred_svm)
    auc_svm = roc_auc_score(y_test, y_proba_svm)
    print(f"SVM - Testowy: F1: {f1_svm}, Precyzja: {precision_svm}, Czułość: {recall_svm}, AUC: {auc_svm}")

    rf_model = RandomForestClassifier(n_estimators=100)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    y_proba_rf = rf_model.predict_proba(X_test)[:, 1]
    f1_rf = f1_score(y_test, y_pred_rf)
    precision_rf = precision_score(y_test, y_pred_rf)
    recall_rf = recall_score(y_test, y_pred_rf)
    auc_rf = roc_auc_score(y_test, y_proba_rf)
    print(f"RandomForest - Testowy: F1: {f1_rf}, Precyzja: {precision_rf}, Czułość: {recall_rf}, AUC: {auc_rf}")

    model_path_knn = f"models/knn_model_user_{user.id}.pkl"
    with open(model_path_knn, "wb") as f:
        pickle.dump(knn_model, f)
    print("KNN zapisany pod:", model_path_knn)

    model_path_svm = f"models/svm_model_user_{user.id}.pkl"
    with open(model_path_svm, "wb") as f:
        pickle.dump(svm_model, f)
    print("SVM zapisany pod:", model_path_svm)

    model_path_rf = f"models/rf_model_user_{user.id}.pkl"
    with open(model_path_rf, "wb") as f:
        pickle.dump(rf_model, f)
    print("RandomForest zapisany pod:", model_path_rf)


    with open("wyniki.txt", "w", encoding="utf-8") as f:
        f.write("Metryki dla modelu MLP (BiometricNet) - Testowy:\n")
        f.write(f"F1: {metrics_mlp[0]}\n")
        f.write(f"Precyzja: {metrics_mlp[1]}\n")
        f.write(f"Czułość: {metrics_mlp[2]}\n")
        f.write(f"AUC: {metrics_mlp[3]}\n\n")

        f.write("Metryki dla modelu LSTM - Testowy:\n")
        f.write(f"F1: {metrics_lstm[0]}\n")
        f.write(f"Precyzja: {metrics_lstm[1]}\n")
        f.write(f"Czułość: {metrics_lstm[2]}\n")
        f.write(f"AUC: {metrics_lstm[3]}\n\n")

        f.write("Metryki dla modelu KNN - Testowy:\n")
        f.write(f"F1: {f1_knn}\n")
        f.write(f"Precyzja: {precision_knn}\n")
        f.write(f"Czułość: {recall_knn}\n")
        f.write(f"AUC: {auc_knn}\n\n")

        f.write("Metryki dla modelu SVM - Testowy:\n")
        f.write(f"F1: {f1_svm}\n")
        f.write(f"Precyzja: {precision_svm}\n")
        f.write(f"Czułość: {recall_svm}\n")
        f.write(f"AUC: {auc_svm}\n\n")

        f.write("Metryki dla modelu RandomForest - Testowy:\n")
        f.write(f"F1: {f1_rf}\n")
        f.write(f"Precyzja: {precision_rf}\n")
        f.write(f"Czułość: {recall_rf}\n")
        f.write(f"AUC: {auc_rf}\n")
    print("Wyniki zapisane w 'wyniki.txt'.")

    return {
        "message": "Modele biometryczne zostały wytrenowane i zapisane.",
        "model_paths": {
            "MLP": model_path_mlp,
            "LSTM": model_path_lstm,
            "KNN": model_path_knn,
            "SVM": model_path_svm,
            "RandomForest": model_path_rf,
        },
        "metrics": {
            "MLP": {"F1": metrics_mlp[0], "Precyzja": metrics_mlp[1], "Czułość": metrics_mlp[2], "AUC": metrics_mlp[3]},
            "LSTM": {"F1": metrics_lstm[0], "Precyzja": metrics_lstm[1], "Czułość": metrics_lstm[2], "AUC": metrics_lstm[3]},
            "KNN": {"F1": f1_knn, "Precyzja": precision_knn, "Czułość": recall_knn, "AUC": auc_knn},
            "SVM": {"F1": f1_svm, "Precyzja": precision_svm, "Czułość": recall_svm, "AUC": auc_svm},
            "RandomForest": {"F1": f1_rf, "Precyzja": precision_rf, "Czułość": recall_rf, "AUC": auc_rf}
        }
    }
