// PATH: /frontend/src/components/PredictionResult.jsx
import React from 'react';

const PredictionResult = ({ result }) => {
  if (!result) {
    return (
      <div className="alert alert-info text-center mt-4">
        <h4 className="alert-heading">Awaiting Prediction</h4>
        <p className="mb-0">Your result will appear here after submitting the form.</p>
      </div>
    );
  }

  return (
    <div className="alert alert-success text-center mt-4">
      <h4 className="alert-heading">Prediction Result</h4>
      <p className="display-5 fw-bold">
         ${result.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
      </p>
      <hr />
      <p className="mb-0 small">
        This value represents the model's estimate for the patient's total healthcare spending for one year.
      </p>
    </div>
  );
};

export default PredictionResult;