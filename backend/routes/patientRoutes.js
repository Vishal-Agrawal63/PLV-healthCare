// PATH: /backend/routes/patientRoutes.js
const express = require('express');
const axios = require('axios');
const Patient = require('../models/Patient');
const router = express.Router();

const ML_API_URL = 'http://127.0.0.1:5000/predict';
const DL_API_URL = 'http://127.0.0.1:5001/predict_survival';
const TABNET_PCA_API_URL = 'http://127.0.0.1:5002/predict_survival_pca'; // New TabNet+PCA API
// ... (other routes remain the same) ...
router.get('/patients', async (req, res) => {
    try {
        const patients = await Patient.find({}).sort({ createdAt: -1 });
        res.json(patients);
    } catch (error) {
        res.status(500).json({ message: 'Server Error' });
    }
});


router.post('/predict', async (req, res) => {
    try {
        // --- KEY CHANGE: Expect a structured payload ---
        const { model_name, patient_data } = req.body;
        if (!model_name || !patient_data) {
            return res.status(400).json({ message: 'model_name and patient_data are required.' });
        }

        // 1. Forward the new structure to the Python ML API
        const { data: prediction } = await axios.post(ML_API_URL, { model_name, patient_data });

        // 2. Combine input data, model used, and the prediction result
        const newPatientRecord = {
            ...patient_data,
            modelUsed: model_name, // Save the model name
            predictedExpenditure: prediction.predicted_expenditure,
        };

        // 3. Save the new record to MongoDB
        const savedPatient = await Patient.create(newPatientRecord);

        // 4. Return the saved record to the frontend
        res.status(201).json(savedPatient);

    } catch (error) {
        // ... (error handling remains the same) ...
        console.error('Error during prediction:', error.message);
        if (error.response) {
            console.error('ML API Response Data:', error.response.data);
            console.error('ML API Response Status:', error.response.status);
        } else if (error.request) {
            console.error('No response from ML API. Is it running?');
        }
        res.status(500).json({ message: 'Failed to get prediction.' });
    }
});


// Replace your entire predict-survival route with this one
router.post('/predict-survival', async (req, res) => {
    try {
        const { model_name, ...clinicalData } = req.body;
        
        let targetApiUrl;
        
        // Choose the correct API based on the user's selection
        if (model_name === 'TabNet_PCA') {
            targetApiUrl = TABNET_PCA_API_URL;
        } else {
            targetApiUrl = DL_API_URL; // Default to the original MLP
        }

        console.log(`Routing to ${model_name} via ${targetApiUrl}`);
        
        const { data: prediction } = await axios.post(targetApiUrl, clinicalData);
        
        res.status(200).json(prediction);

    } catch (error) {
        console.error('Error during survival prediction:', error.message);
        if (error.response) {
            console.error('API Response Data:', error.response.data);
        } else {
            console.error('No response from a DL API. Are they running?');
        }
        res.status(500).json({ message: 'Failed to get survival prediction.' });
    }
});


module.exports = router;