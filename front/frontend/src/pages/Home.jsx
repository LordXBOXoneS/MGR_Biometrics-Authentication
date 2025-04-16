import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import api from "../api";
import goodImage from "../assets/good.jpg";
import badImage from "../assets/bad.jpg";

function Info() {
  const [attempts, setAttempts] = useState([]);
  const navigate = useNavigate();

  useEffect(() => {
    const fetchAttempts = async () => {
      try {
        const res = await api.get("/api/login-attempts/");
        setAttempts(res.data);
      } catch (error) {
        console.error("Błąd pobierania historii logowań:", error);
      }
    };
    fetchAttempts();
  }, []);

  return (
    <div style={{ padding: "20px", maxWidth: "900px", margin: "0 auto" }}>
      <h1 style={{ textAlign: "center", marginBottom: "20px" }}>
        Historia prób logowania
      </h1>
      <div
        style={{
          maxHeight: "400px",
          overflowY: "auto",
          border: "1px solid #ccc",
          borderRadius: "4px",
          marginBottom: "20px",
          padding: "10px",
        }}
      >
        {attempts.length > 0 ? (
          <table style={{ width: "100%", borderCollapse: "collapse" }}>
            <thead>
              <tr>
                <th style={{ border: "1px solid #ccc", padding: "8px" }}>Data</th>
                <th style={{ border: "1px solid #ccc", padding: "8px" }}>Metoda</th>
                <th style={{ border: "1px solid #ccc", padding: "8px" }}>Wynik</th>
                <th style={{ border: "1px solid #ccc", padding: "8px" }}>Szczegóły</th>
                <th style={{ border: "1px solid #ccc", padding: "8px" }}>Status</th>
              </tr>
            </thead>
            <tbody>
              {attempts.map((att, index) => (
                <tr key={index}>
                  <td style={{ border: "1px solid #ccc", padding: "8px" }}>
                    {new Date(att.timestamp).toLocaleString()}
                  </td>
                  <td style={{ border: "1px solid #ccc", padding: "8px" }}>
                    {att.method}
                  </td>
                  <td style={{ border: "1px solid #ccc", padding: "8px" }}>
                    {att.success ? "Sukces" : "Niepowodzenie"}
                  </td>
                  <td style={{ border: "1px solid #ccc", padding: "8px" }}>
                    {JSON.stringify(att.details.message)}
                  </td>
                  <td style={{ border: "1px solid #ccc", padding: "8px", textAlign: "center" }}>
                    <img
                      src={att.success ? goodImage : badImage}
                      alt={att.success ? "Sukces" : "Niepowodzenie"}
                      style={{ width: "50px", height: "50px" }}
                    />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <p style={{ textAlign: "center" }}>Brak zapisanych prób logowania.</p>
        )}
      </div>
      <div style={{ textAlign: "center" }}>
        <button
          onClick={() => navigate("/profile")}
          style={{
            backgroundColor: "#007bff",
            color: "#fff",
            border: "none",
            padding: "10px 20px",
            borderRadius: "4px",
            cursor: "pointer",
          }}
        >
          Powrót do profilu
        </button>
      </div>
    </div>
  );
}

export default Info;
