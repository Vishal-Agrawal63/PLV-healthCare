// SurvivalPrediction.jsx
import React, { useState } from 'react';
import axios from 'axios';

const SurvivalPrediction = () => {
    const [formData, setFormData] = useState({
        model_name: 'coxtime_pca', // Default to a strong PCA model
        // CRITICAL: Ensure this matches ALL columns needed for the scaler
        age: 65.0,
        sex: 0,
        dzgroup: 2,
        'num.co': 2.0,
        scoma: 40.0,
        temp: 37.0,   // Example default value
        hrt: 90,      // Example default value
        meanbp: 85,
        // ... Add ALL other original features your scaler was trained on
    });
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');

    const handleChange = (e) => {
        const { name, value } = e.target;
        const isNumeric = e.target.type === 'number';
        setFormData(prev => ({ 
            ...prev, 
            [name]: isNumeric ? Number(value) : value 
        }));
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);
        setError('');
        setResult(null);

        const { model_name, ...patient_data } = formData;
        const payload = { model_name, patient_data };

        try {
            const res = await axios.post('http://localhost:5001/predict_survival', payload);
            setResult(res.data);
        } catch (err) {
            console.error("Prediction API error:", err);
            setError(err.response?.data?.error || 'Prediction failed. Check backend logs.');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="card shadow-sm mt-4">
            <div className="card-header">
                <h2 className="h4 mb-0">PCA-Based Survival Prediction</h2>
                <p className="text-muted small mb-0">Compare different model architectures on the same dimension-reduced feature set.</p>
            </div>
            <div className="card-body">
                <form onSubmit={handleSubmit}>
                    <div className="mb-4">
                        <label htmlFor="model_name" className="form-label fw-bold">Select PCA-Based Model</label>
                        <select className="form-select form-select-lg" id="model_name" name="model_name" value={formData.model_name} onChange={handleChange}>
                            <option value="coxtime_pca">Cox-Time + PCA</option>
                            <option value="deepsurv_pca">DeepSurv + PCA</option>
                            <option value="deephit_pca">DeepHit + PCA</option>
                            <option value="rsf_pca">Random Survival Forest + PCA</option>
                            <option value="keras_pca">Keras (Classification) + PCA</option>
                        </select>
                    </div>

                    <p className="fw-bold">Enter Patient Data (Full Features):</p>
                    <div className="row g-3">
                        {/* Ensure your input form collects ALL original features */}
                        <div className="col-md-4"><label className="form-label">Age</label><input type="number" step="any" name="age" value={formData.age} onChange={handleChange} className="form-control" /></div>
                        <div className="col-md-4"><label className="form-label">Sex (0=F, 1=M)</label><input type="number" name="sex" value={formData.sex} onChange={handleChange} className="form-control" /></div>
                        <div className="col-md-4"><label className="form-label">Disease Group</label><input type="number" name="dzgroup" value={formData.dzgroup} onChange={handleChange} className="form-control" /></div>
                        <div className="col-md-6"><label className="form-label">Num. of Comorbidities</label><input type="number" step="any" name="num.co" value={formData['num.co']} onChange={handleChange} className="form-control" /></div>
                        <div className="col-md-6"><label className="form-label">Coma Score</label><input type="number" step="any" name="scoma" value={formData.scoma} onChange={handleChange} className="form-control" /></div>
                        {/* ... add any other necessary input fields here ... */}
                        <div className="col-md-4"><label className="form-label">Temperature</label><input type="number" step="any" name="temp" value={formData.temp} onChange={handleChange} className="form-control" /></div>
                        <div className="col-md-4"><label className="form-label">Heart Rate</label><input type="number" step="any" name="hrt" value={formData.hrt} onChange={handleChange} className="form-control" /></div>
                        <div className="col-md-4"><label className="form-label">Mean Blood Pressure</label><input type="number" step="any" name="meanbp" value={formData.meanbp} onChange={handleChange} className="form-control" /></div>
                    </div>
                    <hr className="my-4" />
                    <div className="d-flex justify-content-end">
                        <button type="submit" className="btn btn-info btn-lg" disabled={loading}>{loading ? 'Analyzing...' : 'Get 30-Day Survival Prediction'}</button>
                    </div>
                    {error && <p className="text-danger text-end mt-2">{error}</p>}
                </form>

                {result && (
                    <div className="alert alert-primary text-center mt-4">
                        <h4 className="alert-heading">Prediction Result ({formData.model_name.toUpperCase().replace('_', ' ')})</h4>
                        <p className="display-5 fw-bold">{result.survival_probability}%</p>
                        <hr />
                        <p className="mb-0 small">This is the model's predicted probability of surviving beyond 30 days based on PCA-transformed features.</p>
                    </div>
                )}
            </div>
        </div>
    );
};

export default SurvivalPrediction;