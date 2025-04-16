import axios from 'axios';
import { ACCESS_TOKEN } from './constants';

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL,
  headers: {
    "Content-Type": "application/json",
  },
  withCredentials: true, // jeśli korzystasz z sesji
});

// Dodaj interceptor, który dołącza token tylko do żądań, które nie dotyczą logowania biometrycznego
api.interceptors.request.use(
  (config) => {
    if (!config.url.includes("biometric-login")) {
      const token = localStorage.getItem(ACCESS_TOKEN);
      if (token) {
        config.headers.Authorization = `Bearer ${token}`;
      }
    }
    return config;
  },
  (error) => Promise.reject(error)
);

export default api;
