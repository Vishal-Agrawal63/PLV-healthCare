// PATH: /frontend/src/pages/SurvivalPrediction.jsx
import React, { useState } from 'react';
import axios from 'axios';

const SurvivalPrediction = () => {
    const [formData, setFormData] = useState({
        // State now includes model_name
        model_name: 'rsf', // Default to RSF as a good starting point
        // Patient data (5 features)
        age: 65,
        sex: 0,
        dzgroup: 2,
        'num.co': 2,
        scoma: 40,
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

        // Separate model_name from the patient_data for the payload
        const { model_name, ...patient_data } = formData;
        const payload = { model_name, patient_data };

        try {
            const res = await axios.post('http://localhost:3001/api/predict-survival', payload);
            setResult(res.data);
        } catch (err) {
            console.error("Prediction API error:", err);
            setError('Prediction failed. Ensure backend and ML APIs are running.');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="card shadow-sm mt-4">
            <div className="card-header">
                <h2 className="h4 mb-0">Unified Survival Prediction</h2>
                <p className="text-muted small mb-0">Select a model and enter patient data for a prediction.</p>
            </div>
            <div className="card-body">
                <form onSubmit={handleSubmit}>
                    {/* Model Selection Dropdown */}
                    <div className="mb-4">
                        <label htmlFor="model_name" className="form-label fw-bold">Select Prediction Model</label>
                        <select className="form-select form-select-lg" id="model_name" name="model_name" value={formData.model_name} onChange={handleChange}>
                            <option value="coxtime">Cox-Time (Best for Calibration)</option>
                            <option value="deepsurv">DeepSurv (Deep Learning Baseline)</option>
                            <option value="deephit">DeepHit (Best for Ranking)</option>
                            <option value="rsf">Random Survival Forest (Benchmark)</option>
                            <option value="keras_pca">Keras + PCA (Original)</option>
                        </select>
                    </div>

                    <div className="row g-3">
                        <div className="col-md-4"><label className="form-label">Age</label><input type="number" name="age" value={formData.age} onChange={handleChange} className="form-control" /></div>
                        <div className="col-md-4"><label className="form-label">Sex (0=F, 1=M)</label><input type="number" name="sex" value={formData.sex} onChange={handleChange} className="form-control" /></div>
                        <div className="col-md-4"><label className="form-label">Disease Group Code</label><input type="number" name="dzgroup" value={formData.dzgroup} onChange={handleChange} className="form-control" /></div>
                        <div className="col-md-6"><label className="form-label">Number of Comorbidities</label><input type="number" name="num.co" value={formData['num.co']} onChange={handleChange} className="form-control" /></div>
                        <div className="col-md-6"><label className="form-label">SUPPORT Coma Score</label><input type="number" name="scoma" value={formData.scoma} onChange={handleChange} className="form-control" /></div>
                    </div>
                    <hr className="my-4" />
                    <div className="d-flex justify-content-end">
                        <button type="submit" className="btn btn-success btn-lg" disabled={loading}>{loading ? 'Analyzing...' : 'Get Survival Prediction'}</button>
                    </div>
                    {error && <p className="text-danger text-end mt-2">{error}</p>}
                </form>

                {result && (
                    <div className="alert alert-info text-center mt-4">
                        <h4 className="alert-heading">Prediction Result (using {formData.model_name.toUpperCase()})</h4>
                        <p className="display-5 fw-bold">{result.survival_probability}%</p>
                        <hr />
                        <p className="mb-0 small">This is the model's predicted probability that the patient will survive beyond 30 days.</p>
                    </div>
                )}
            </div>
        </div>
    );
};

export default SurvivalPrediction;