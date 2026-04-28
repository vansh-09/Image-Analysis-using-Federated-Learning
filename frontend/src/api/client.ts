import axios from 'axios';

const API_BASE = 'http://localhost:6060/api';

const client = axios.create({
  baseURL: API_BASE,
  timeout: 30000,
});

export const api = {
  // Model & Info
  health: () => client.get('/health'),
  modelInfo: () => client.get('/model-info'),

  // Dashboard
  networkDashboard: () => client.get('/network-dashboard'),
  hospitalDetails: (name: string) => client.get(`/hospital/${name}`),

  // Analytics
  analytics: () => client.get('/analytics'),
  trainingHistory: () => client.get('/training-history'),

  // Prediction
  predict: (file: File) => {
    const formData = new FormData();
    formData.append('image', file);
    return client.post('/predict', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
  },

  // Prediction PDF Report
  predictPDF: (file: File) => {
    const formData = new FormData();
    formData.append('file', file);
    return client.post('/predict-pdf', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      responseType: 'blob',
    });
  },
};

export default api;
