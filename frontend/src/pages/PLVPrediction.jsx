// PATH: /frontend/src/pages/PLVPrediction.jsx
import React, { useState } from 'react';
import PatientForm from '../components/PatientForm';
import PatientTable from '../components/PatientTable';
import PredictionResult from '../components/PredictionResult';

const PLVPrediction = () => {
    const [latestPrediction, setLatestPrediction] = useState(null);

    const handleNewPrediction = (predictionData) => {
        setLatestPrediction(predictionData);
    };

    return (
        <>
            <PatientForm onNewPrediction={handleNewPrediction} />
            <PredictionResult result={latestPrediction ? latestPrediction.predictedExpenditure : null} />
            <PatientTable newPrediction={latestPrediction} />
        </>
    );
};

export default PLVPrediction;