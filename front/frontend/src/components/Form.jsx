import { useState, useEffect, useRef } from "react";
import api from "../api";
import { useNavigate } from "react-router-dom";
import { ACCESS_TOKEN, REFRESH_TOKEN } from "../constants";
import "../styles/Form.css";

function Form({ route, method }) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [email, setEmail] = useState("");
  const [firstName, setFirstName] = useState("");
  const [lastName, setLastName] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [keyData, setKeyData] = useState([]);
  const [mouseClickData, setMouseClickData] = useState([]);
  const [errors, setErrors] = useState({});
  const navigate = useNavigate();

  const [startTime, setStartTime] = useState(null);
  const [hasBackspace, setHasBackspace] = useState(false);
  const [keyTimings, setKeyTimings] = useState([]);

  const lastKeyDownRef = useRef(null);
  const lastKeyUpRef = useRef(null);
  const activeKeys = useRef({});

  useEffect(() => {
    const token = localStorage.getItem(ACCESS_TOKEN);
    if (token) {
      navigate("/profile");
    }
  }, [navigate]);

  const validateEmail = (email) => {
    const re = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return re.test(String(email).toLowerCase());
  };

  const validatePassword = (password) => {
    const re = /^(?=.*\d)(?=.*[a-z])(?=.*[A-Z]).{8,}$/;
    return re.test(String(password));
  };

  const handleKeyDown = (event) => {
    if (method === "login" && event.target.type === "password") {
      if (event.key === "Backspace") {
        setHasBackspace(true);
        return;
      }
      const now = Date.now();
      if (!startTime) setStartTime(now);
      const newEvent = { key: event.key, downTime: now };
      setKeyTimings((prev) => [...prev, newEvent]);
      lastKeyDownRef.current = now;
      activeKeys.current[event.key] = now;
    } else if (method !== "login") {
      if (!startTime) setStartTime(Date.now());
      const relativeTime = Date.now() - startTime;
      setKeyData((prev) => [...prev, { key: event.key, type: "keydown", time: relativeTime }]);
    }
  };

  const handleKeyUp = (event) => {
    if (method === "login" && event.target.type === "password") {
      const now = Date.now();
      if (!startTime) return;
      setKeyTimings((prev) =>
        prev.map((ev) => {
          if (ev.key === event.key && ev.releaseTime === undefined) {
            return { ...ev, releaseTime: now, holdTime: now - ev.downTime };
          }
          return ev;
        })
      );
      lastKeyUpRef.current = now;
      if (activeKeys.current[event.key]) {
        delete activeKeys.current[event.key];
      }
    } else if (method !== "login") {
      if (!startTime) return;
      const relativeTime = Date.now() - startTime;
      setKeyData((prev) => [...prev, { key: event.key, type: "keyup", time: relativeTime }]);
    }
  };

  const handleMouseClick = (event) => {
    if (!startTime) setStartTime(Date.now());
    const relativeTime = Date.now() - startTime;
    const { clientX, clientY } = event;
    setMouseClickData((prev) => [
      ...prev,
      { type: "click", x: clientX, y: clientY, time: relativeTime },
    ]);
  };

  const resetFields = () => {
    setKeyData([]);
    setMouseClickData([]);
    setKeyTimings([]);
    setStartTime(null);
    lastKeyDownRef.current = null;
    lastKeyUpRef.current = null;
    activeKeys.current = {};
    setUsername("");
    setPassword("");
    setHasBackspace(false);
  };

  const computeBiometricMetrics = () => {
    const fullEvents = keyTimings.filter((ev) => ev.holdTime !== undefined);
    const CZN = fullEvents.map((ev) => ev.holdTime / 1000); 
    const CNN = fullEvents.map((ev, i, arr) =>
      i === 0 ? 0 : (ev.downTime - arr[i - 1].downTime) / 1000
    );
    const CPN = fullEvents.map((ev, i, arr) =>
      i === 0 ? 0 : (ev.downTime - (arr[i - 1].releaseTime || ev.downTime)) / 1000
    );
    return { CZN, CNN, CPN };
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);

    if (method === "login") {
      const MIN_FULL_EVENTS = 5;
      const fullEvents = keyTimings.filter((ev) => ev.holdTime !== undefined);
      if (fullEvents.length < MIN_FULL_EVENTS) {
        alert("Nie przeprowadzono wystarczającej rejestracji danych biometrycznych. Wpisz hasło naturalnie.");
        resetFields();
        setLoading(false);
        return;
      }
    }

    if (hasBackspace) {
      alert("W trakcie tej próby użyto backspace. Proszę powtórzyć próbę bez usuwania znaków.");
      resetFields();
      setLoading(false);
      return;
    }

    try {

      if (route.includes("biometric-login")) {
        const biometricMetrics = computeBiometricMetrics();
        const biometricPayload = {
          username,
          password,
          key_events: keyTimings,       
          mouse_events: mouseClickData,
          biometric_metrics: biometricMetrics,
          session_id: Date.now().toString(),
          login_attempt_number: 1,
        };
        const res = await api.post(route, biometricPayload);
        localStorage.setItem(ACCESS_TOKEN, res.data.access);
        localStorage.setItem(REFRESH_TOKEN, res.data.refresh);
        navigate("/profile");
      } else {
        const data = { username, password, keyData, mouseClickData };
        if (method === "register") {
          data.email = email;
          data.first_name = firstName;
          data.last_name = lastName;
        }
        const res = await api.post(route, data);
        localStorage.setItem(ACCESS_TOKEN, res.data.access);
        localStorage.setItem(REFRESH_TOKEN, res.data.refresh);
        navigate("/profile");
      }
    } catch (error) {
      console.error("Błąd:", error.response ? error.response.data : error);
      alert("Wystąpił błąd. Sprawdź konsolę dla szczegółów.");
      resetFields();
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ textAlign: "center", marginTop: "20px" }}>
      <h1 style={{ marginBottom: "20px", fontSize: "2rem", color: "white" }}>
        Badanie Biometrii Behawioralnej by{" "}
        <span style={{ fontWeight: "bold" }}>Gabriel Łukaniuk</span>
      </h1>
      <form onSubmit={handleSubmit} onClick={handleMouseClick} className="form-container">
        <input
          className={`form-input ${errors.username ? "input-error" : ""}`}
          type="text"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
          onKeyDown={handleKeyDown}
          onKeyUp={handleKeyUp}
          placeholder="Login"
        />
        {errors.username && <span className="error-message">{errors.username}</span>}

        {method === "register" && (
          <>
            <input
              className={`form-input ${errors.firstName ? "input-error" : ""}`}
              type="text"
              value={firstName}
              onChange={(e) => setFirstName(e.target.value)}
              placeholder="Imię"
            />
            {errors.firstName && <span className="error-message">{errors.firstName}</span>}

            <input
              className={`form-input ${errors.lastName ? "input-error" : ""}`}
              type="text"
              value={lastName}
              onChange={(e) => setLastName(e.target.value)}
              placeholder="Nazwisko"
            />
            {errors.lastName && <span className="error-message">{errors.lastName}</span>}

            <input
              className={`form-input ${errors.email ? "input-error" : ""}`}
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="Email"
            />
            {errors.email && <span className="error-message">{errors.email}</span>}
          </>
        )}

        <input
          className={`form-input ${errors.password ? "input-error" : ""}`}
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          onKeyDown={handleKeyDown}
          onKeyUp={handleKeyUp}
          placeholder="Hasło"
        />
        {errors.password && <span className="error-message">{errors.password}</span>}

        {method === "register" && (
          <>
            <input
              className={`form-input ${errors.confirmPassword ? "input-error" : ""}`}
              type="password"
              value={confirmPassword}
              onChange={(e) => setConfirmPassword(e.target.value)}
              placeholder="Potwierdź hasło"
            />
            {errors.confirmPassword && (
              <span className="error-message">{errors.confirmPassword}</span>
            )}
          </>
        )}

        <button className="form-button" type="submit" disabled={loading}>
          {loading ? "Ładowanie..." : method === "login" ? "Zaloguj" : "Zarejestruj"}
        </button>
      </form>
    </div>
  );
}

export default Form;
