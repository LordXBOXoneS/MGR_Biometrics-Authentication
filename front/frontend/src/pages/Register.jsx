import React from "react";
import { Link } from "react-router-dom";
import Form from "../components/Form";

function Register() {
  return (
    <div style={{ textAlign: "center", marginTop: "20px" }}>
      <Form route="api/user/register/" method="register" />
      <div style={{ marginTop: "20px" }}>
        <Link to="/login">
          <button
            style={{
              backgroundColor: "#007bff",
              color: "white",
              border: "none",
              padding: "10px 20px",
              borderRadius: "5px",
              cursor: "pointer",
              fontSize: "1rem",
              transition: "background-color 0.2s ease-in-out",
            }}
            onMouseEnter={(e) => (e.target.style.backgroundColor = "#0056b3")}
            onMouseLeave={(e) => (e.target.style.backgroundColor = "#007bff")}
          >
            Powrót do logowania
          </button>
        </Link>
      </div>
    </div>
  );
}

export default Register;
