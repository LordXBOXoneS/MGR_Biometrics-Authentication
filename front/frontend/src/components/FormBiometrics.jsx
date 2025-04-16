import { useState, useRef } from "react";
import api from "../api";
import "../styles/Form.css";

function FormBiometrics({ onSubmit }) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [keyData, setKeyData] = useState([]);
  const [mouseClickData, setMouseClickData] = useState([]);
  const [startTime, setStartTime] = useState(null);

  const [keyPressDurations, setKeyPressDurations] = useState([]); // CZN.x
  const [timeBetweenKeydowns, setTimeBetweenKeydowns] = useState([]); // CNN.x
  const [timeBetweenKeyupAndNextKeydown, setTimeBetweenKeyupAndNextKeydown] = useState([]); // CPN.x

  const [hasBackspace, setHasBackspace] = useState(false);

  const lastKeyDownRef = useRef(null);
  const lastKeyUpRef = useRef(null);
  const activeKeys = useRef({});

  const resetFields = () => {
    setKeyData([]);
    setMouseClickData([]);
    setKeyPressDurations([]);
    setTimeBetweenKeydowns([]);
    setTimeBetweenKeyupAndNextKeydown([]);
    setStartTime(null);
    lastKeyDownRef.current = null;
    lastKeyUpRef.current = null;
    activeKeys.current = {};
    setUsername("");
    setPassword("");
    setHasBackspace(false);
  };

  const handlePasswordKeyDown = (event) => {
    if (event.key === "Backspace") {
      setHasBackspace(true);
      return; 
    }
    const currentTime = Date.now();
    if (!startTime) {
      setStartTime(currentTime);
    }
    const relativeTime = currentTime - (startTime || currentTime);
    setKeyData((prevData) => [
      ...prevData,
      { key: event.key, type: "keydown", time: relativeTime },
    ]);
    if (lastKeyDownRef.current) {
      const diff = currentTime - lastKeyDownRef.current;
      const cnnSec = parseFloat((diff / 1000).toFixed(4));
      setTimeBetweenKeydowns((prev) => [...prev, cnnSec]);
    }
    lastKeyDownRef.current = currentTime;
    activeKeys.current[event.key] = currentTime;
  };

  const handlePasswordKeyUp = (event) => {
    if (event.key === "Backspace") {
      return; 
    }
    const currentTime = Date.now();
    if (!startTime) return;
    const relativeTime = currentTime - startTime;
    setKeyData((prevData) => [
      ...prevData,
      { key: event.key, type: "keyup", time: relativeTime },
    ]);
    if (activeKeys.current[event.key]) {
      const duration = currentTime - activeKeys.current[event.key];
      const durationSec = parseFloat((duration / 1000).toFixed(4));
      setKeyPressDurations((prev) => [...prev, durationSec]);
      delete activeKeys.current[event.key];
    }
    if (lastKeyUpRef.current) {
      const diff = currentTime - lastKeyUpRef.current;
      const cpnSec = parseFloat((diff / 1000).toFixed(4));
      setTimeBetweenKeyupAndNextKeydown((prev) => [...prev, cpnSec]);
    }
    lastKeyUpRef.current = currentTime;
  };

  const handleMouseClick = (event) => {
    const currentTime = Date.now();
    if (!startTime) {
      setStartTime(currentTime);
    }
    const relativeTime = currentTime - startTime;
    setMouseClickData((prevData) => [
      ...prevData,
      { type: "click", x: event.clientX, y: event.clientY, time: relativeTime },
    ]);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (hasBackspace) {
      alert("W trakcie tej próby użyto backspace. Proszę powtórzyć próbę bez usuwania znaków.");
      resetFields();
      return;
    }

    try {
      await api.post("/api/token/", { username, password });
    } catch (error) {
      alert("Podane dane logowania są niepoprawne. Użyj swoich danych.");
      resetFields();
      return;
    }

    const formData = {
      username,
      password,
      key_events: keyData,
      mouse_events: mouseClickData,
      biometric_metrics: {
        CZN: keyPressDurations,
        CNN: timeBetweenKeydowns,
        CPN: timeBetweenKeyupAndNextKeydown,
      },
      session_id: Date.now().toString(),
      login_attempt_number: 1,
    };

    try {
      const response = await api.post("/api/biometric-data/", formData);
      console.log("Dane zapisane pomyślnie:", response.data);
      if (onSubmit) {
        onSubmit(formData);
      }
    } catch (error) {
      console.error("Błąd podczas zapisywania danych:", error);
    }

    resetFields();
  };

  return (
    <div style={{ textAlign: "center", marginTop: "20px" }}>
      <h1 style={{ marginBottom: "20px", fontSize: "2rem", color: "white" }}>
        Badanie Biometrii Behawioralnej by{" "}
        <span style={{ fontWeight: "bold" }}>Gabriel Łukaniuk</span>
      </h1>
      <form onSubmit={handleSubmit} onClick={handleMouseClick} className="form-container">
        <input
          type="text"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
          placeholder="Username"
          className="form-input"
        />
        <input
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          onKeyDown={handlePasswordKeyDown}
          onKeyUp={handlePasswordKeyUp}
          placeholder="Password"
          className="form-input"
        />
        <button type="submit" className="form-button">
          Zaloguj
        </button>
      </form>
    </div>
  );
}

export default FormBiometrics;
