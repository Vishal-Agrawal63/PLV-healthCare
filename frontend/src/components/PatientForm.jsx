// PATH: /frontend/src/components/PatientForm.jsx
import React, { useState } from 'react';
import axios from 'axios';

const PatientForm = ({ onNewPrediction }) => {
  const [formData, setFormData] = useState({
    // --- CHANGE: Default model is now LightGBM_Tweedie ---
    model_name: 'LightGBM_Tweedie',
    // Patient data
    Age: 45, Sex: 1, Race: 1, PovertyCategory: 4,
    InsuranceCoverage: 1, HealthStatus: 3, OfficeVisits: 5,
    OutpatientVisits: 1, ERVisits: 0, HospitalDischarges: 0,
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleChange = (e) => {
    const { name, value } = e.target;
    const isNumeric = e.target.type === 'number';
    setFormData((prevData) => ({
      ...prevData,
      [name]: isNumeric ? Number(value) : value,
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    const { model_name, ...patient_data } = formData;
    const payload = { model_name, patient_data };
    
    console.log('Submitting this payload to backend:', payload);

    try {
      const res = await axios.post('http://localhost:3001/api/predict', payload);
      onNewPrediction(res.data);
    } catch (err) {
      setError('Prediction failed. Ensure backend and ML API are running.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="card shadow-sm">
      <div className="card-header">
        <h2 className="h4 mb-0">Enter New Patient Data</h2>
        <p className="text-muted small mb-0">Select a model and fill in the details for a prediction.</p>
      </div>
      <div className="card-body">
        <form onSubmit={handleSubmit}>
          {/* --- CHANGE: Update the dropdown options --- */}
          <div className="mb-4">
            <label htmlFor="model_name" className="form-label fw-bold">Select Prediction Model</label>
            <select className="form-select form-select-lg" id="model_name" name="model_name" value={formData.model_name} onChange={handleChange}>
              <option value="LightGBM_Tweedie">LightGBM Tweedie (Recommended)</option>
              <option value="XGBoost">XGBoost</option>
              <option value="LightGBM">LightGBM</option>
              <option value="RandomForest">Random Forest</option>
            </select>
          </div>

          <div className="row g-3">
            {/* All other input fields remain exactly the same */}
            <div className="col-md-4"><label htmlFor="Age" className="form-label">Age</label><input type="number" className="form-control" id="Age" name="Age" value={formData.Age} onChange={handleChange} /></div>
            <div className="col-md-4"><label htmlFor="Sex" className="form-label">Sex</label><select className="form-select" id="Sex" name="Sex" value={formData.Sex} onChange={handleChange}><option value="1">Male</option><option value="2">Female</option></select></div>
            <div className="col-md-4"><label htmlFor="Race" className="form-label">Race (Code)</label><input type="number" className="form-control" id="Race" name="Race" value={formData.Race} onChange={handleChange} /></div>
            <div className="col-md-4"><label htmlFor="PovertyCategory" className="form-label">Poverty Category (1-5)</label><input type="number" className="form-control" id="PovertyCategory" name="PovertyCategory" value={formData.PovertyCategory} onChange={handleChange} /></div>
            <div className="col-md-4"><label htmlFor="InsuranceCoverage" className="form-label">Insurance Coverage (1-3)</label><input type="number" className="form-control" id="InsuranceCoverage" name="InsuranceCoverage" value={formData.InsuranceCoverage} onChange={handleChange} /></div>
            <div className="col-md-4"><label htmlFor="HealthStatus" className="form-label">Health Status (1-5)</label><input type="number" className="form-control" id="HealthStatus" name="HealthStatus" value={formData.HealthStatus} onChange={handleChange} /></div>
            <div className="col-md-6"><label htmlFor="OfficeVisits" className="form-label">Annual Office Visits</label><input type="number" className="form-control" id="OfficeVisits" name="OfficeVisits" value={formData.OfficeVisits} onChange={handleChange} /></div>
            <div className="col-md-6"><label htmlFor="OutpatientVisits" className="form-label">Annual Outpatient Visits</label><input type="number" className="form-control" id="OutpatientVisits" name="OutpatientVisits" value={formData.OutpatientVisits} onChange={handleChange} /></div>
            <div className="col-md-6"><label htmlFor="ERVisits" className="form-label">Annual ER Visits</label><input type="number" className="form-control" id="ERVisits" name="ERVisits" value={formData.ERVisits} onChange={handleChange} /></div>
            <div className="col-md-6"><label htmlFor="HospitalDischarges" className="form-label">Annual Hospital Discharges</label><input type="number" className="form-control" id="HospitalDischarges" name="HospitalDischarges" value={formData.HospitalDischarges} onChange={handleChange} /></div>
          </div>
          <hr className="my-4" />
          <div className="d-flex justify-content-end">
             <button type="submit" className="btn btn-primary btn-lg" disabled={loading}>
              {loading ? `Analyzing with ${formData.model_name}...` : 'Get PLV Prediction'}
            </button>
          </div>
           {error && <p className="text-danger text-end mt-2">{error}</p>}
        </form>
      </div>
    </div>
  );
};

export default PatientForm;