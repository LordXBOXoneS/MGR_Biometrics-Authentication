import React, { useEffect } from "react";
import { Link, useNavigate } from "react-router-dom";
import Form from "../components/Form";
import api from "../api";

function Login() {
  const navigate = useNavigate();

  useEffect(() => {
    const token = localStorage.getItem("ACCESS_TOKEN");
    if (token) {
      navigate("/profile");
    }
  }, [navigate]);

  return (
    <div style={{ textAlign: "center", marginTop: "20px" }}>
      <Form route="api/biometric-login/" method="login" />
      <p style={{ marginTop: "20px", color: "white", fontSize: "1rem" }}>
        Nie masz konta?{" "}
        <Link
          to="/register"
          style={{
            color: "#007bff",
            textDecoration: "none",
            fontWeight: "bold",
          }}
        >
          Zarejestruj się!
        </Link>
      </p>
    </div>
  );
}

export default Login;
