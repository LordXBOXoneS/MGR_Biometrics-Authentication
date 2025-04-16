import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import FormBiometrics from "../components/FormBiometrics";
import api from "../api";

function Profile() {
  const [username, setUsername] = useState("");
  const [isCollecting, setIsCollecting] = useState(false);
  const [recordCount, setRecordCount] = useState(0);
  const [isTraining, setIsTraining] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    const fetchProfile = async () => {
      try {
        const res = await api.get("/api/profile/");
        setUsername(res.data.username);
      } catch (error) {
        alert("Nie udało się pobrać profilu. Przekierowywanie do logowania.");
        navigate("/login");
      }
    };
    fetchProfile();
  }, [navigate]);

  const fetchRecordCount = async () => {
    try {
      const res = await api.get("/api/biometric-data/count/");
      setRecordCount(res.data.count);
    } catch (error) {
      console.error("Błąd pobierania liczby rekordów:", error);
    }
  };

  useEffect(() => {
    if (isCollecting && recordCount >= 20 && !isTraining) {
      trainModel();
    }
  }, [recordCount, isCollecting, isTraining]);

  const trainModel = async () => {
    setIsTraining(true);
    alert("Zebrano dane dla 20 logowań. Rozpoczynam trenowanie modelu biometrycznego...");
    try {
      const res = await api.post("/api/biometric-model/train/");
      alert("Model biometryczny został wytrenowany i zapisany");
    } catch (error) {
      console.error("Błąd treningu modelu:", error.response ? error.response.data : error);
      alert("Wystąpił błąd przy trenowaniu modelu biometrycznego.");
    }
    setIsTraining(false);
    setIsCollecting(false);
  };

  const handleLogout = () => {
    localStorage.clear();
    navigate("/logout");
  };

  const startCollection = async () => {
    const accept = window.confirm(
      "Logowanie biometryczne pozwala zabezpieczyć Twoje konto przy użyciu unikalnych danych biometrycznych. Czy akceptujesz użycie tego zabezpieczenia?"
    );
    if (!accept) {
      return;
    }

    try {
      await api.patch("/api/profile/update/", { behaviour_security: true });
    } catch (error) {
      alert("Nie udało się zaktualizować ustawień zabezpieczenia biometrycznego.");
      return;
    }

    setIsCollecting(true);
    await fetchRecordCount();
    alert("Rozpoczęto zbieranie danych biometrycznych. Wykonaj logowanie biometryczne.");
  };

  const cancelBiometric = async () => {
    const accept = window.confirm(
      "Czy na pewno chcesz zrezygnować z ochrony biometrycznej? Wszystkie Twoje dane biometryczne zostaną usunięte."
    );
    if (!accept) return;
  
    try {
      await api.patch("/api/profile/update/", { behaviour_security: false });
      await api.delete("/api/biometric-data/delete-all/");
      alert("Ochrona biometryczna została wyłączona, a wszystkie dane biometryczne usunięte.");
      const res = await api.get("/api/profile/");
      if (res.data.behaviour_security === false) {
        setIsCollecting(false);
      }
    } catch (error) {
      alert("Wystąpił błąd przy rezygnacji z ochrony biometrycznej.");
      console.error(error);
    }
  };
  

  const handleLoginSimulation = async (formData) => {
    console.log(`Logowanie ${recordCount + 1}`);
    console.log("Key Data:", formData.keyData);
    console.log("Mouse Click Data:", formData.mouse_events || formData.mouseClickData);

    const newCount = recordCount + 1;
    setRecordCount(newCount);


    if (newCount >= 20) {
      trainModel();
    }
  };

  return (
    <>
      {!isCollecting ? (
        <div style={{ textAlign: "center", marginTop: "20px" }}>
          <h1>Witaj {username}!</h1>
          <div
            style={{
              marginTop: "20px",
              display: "flex",
              justifyContent: "center",
              gap: "10px",
            }}
          >
            <button
              style={{
                width: "100px",
                height: "100px",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
              }}
              onClick={startCollection}
            >
              Zezwól na ochronę biometryczną
            </button>
            <button
              style={{
                width: "100px",
                height: "100px",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
              }}
              onClick={handleLogout}
            >
              Wyloguj się
            </button>
            <button
              style={{
                width: "100px",
                height: "100px",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
              }}
              onClick={() => navigate("/")}
            >
              Informacje
            </button>
            <button
              style={{
                width: "100px",
                height: "100px",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
              }}
              onClick={cancelBiometric}
            >
              Rezygnuj z ochrony biometrycznej
            </button>
          </div>
        </div>
      ) : isTraining ? (
        <div style={{ textAlign: "center", marginTop: "50px" }}>
          <h2>Uczenie Modeli, Proszę czekać...</h2>
          <div className="spinner"></div>
        </div>
      ) : (
        <div style={{ textAlign: "center", marginTop: "50px" }}>
          <FormBiometrics onSubmit={handleLoginSimulation} />
          <h2>
            {recordCount} / 20 rekordów, pozostało: {20 - recordCount}
          </h2>
        </div>
      )}
    </>
  );
}

export default Profile;
