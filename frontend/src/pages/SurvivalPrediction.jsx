// PATH: /frontend/src/pages/SurvivalPrediction.jsx (FINAL VERSION)

import React, { useState } from 'react';
import axios from 'axios';

// A component to display the result
const ResultDisplay = ({ result, loading, error }) => {
    if (loading) { return <div className="alert alert-info mt-4">Analyzing...</div>; }
    if (error) { return <div className="alert alert-danger mt-4">{error}</div>; }
    if (result === null) { return <div className="alert alert-light mt-4">Prediction result will appear here.</div>; }
    return (
        <div className="alert alert-success mt-4">
            <h4 className="alert-heading">Survival Probability:</h4>
            <p className="display-4 fw-bold mb-0">{result.survival_probability}%</p>
        </div>
    );
};


const SurvivalPrediction = () => {
    const [formData, setFormData] = useState({
        model_name: 'MLP',
        age: 65,
        sex: 0,
        dzgroup: 2,
        'num.co': 2,
        scoma: 40,
    });
    const [result, setResult] = useState(null);
    const [error, setError] = useState('');
    const [loading, setLoading] = useState(false);

    const handleChange = (e) => {
        setFormData({ ...formData, [e.target.name]: e.target.value });
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);
        setError('');
        setResult(null);

        // Convert string inputs from form to numbers
        const payload = {
            model_name: formData.model_name,
            age: Number(formData.age),
            sex: Number(formData.sex),
            dzgroup: Number(formData.dzgroup),
            'num.co': Number(formData['num.co']),
            scoma: Number(formData.scoma),
        };

        try {
            const res = await axios.post('http://localhost:3001/api/predict-survival', payload);
            setResult(res.data);
        } catch (err) {
            setError('Prediction failed. Ensure backend and all APIs are running.');
            console.error(err);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="row justify-content-center">
            <div className="col-lg-8">
                <div className="card shadow">
                    <div className="card-header"><h1 className="h3">Deep Learning Survival Prediction</h1></div>
                    <div className="card-body">
                        <form onSubmit={handleSubmit}>
                            {/* --- UPDATED MODEL SELECTOR --- */}
                            <div className="mb-4">
                                <label htmlFor="model_name" className="form-label fw-bold">Select Prediction Model</label>
                                <select className="form-select form-select-lg" name="model_name" value={formData.model_name} onChange={handleChange}>
                                    <option value="MLP">MLP (Standard Neural Network)</option>
                                    <option value="TabNet_PCA">TabNet with PCA (Advanced)</option>
                                </select>
                            </div>

                            <div className="row g-3">
                                <div className="col-md-4">
                                    <label className="form-label">Age</label>
                                    <input type="number" className="form-control" name="age" value={formData.age} onChange={handleChange} />
                                </div>
                                <div className="col-md-4">
                                    <label className="form-label">Sex (0=F, 1=M)</label>
                                    <input type="number" className="form-control" name="sex" value={formData.sex} onChange={handleChange} />
                                </div>
                                <div className="col-md-4">
                                    <label className="form-label">Disease Group</label>
                                    <input type="number" className="form-control" name="dzgroup" value={formData.dzgroup} onChange={handleChange} />
                                </div>
                                <div className="col-md-6">
                                    <label className="form-label">Comorbidities</label>
                                    <input type="number" className="form-control" name="num.co" value={formData['num.co']} onChange={handleChange} />
                                </div>
                                <div className="col-md-6">
                                    <label className="form-label">Glasgow Coma Score</label>
                                    <input type="number" className="form-control" name="scoma" value={formData.scoma} onChange={handleChange} />
                                </div>
                            </div>
                            <hr className="my-4" />
                            <div className="d-grid">
                                <button type="submit" className="btn btn-success btn-lg" disabled={loading}>
                                    Get Survival Prediction
                                </button>
                            </div>
                        </form>
                    </div>
                </div>
                <ResultDisplay result={result} loading={loading} error={error} />
            </div>
        </div>
    );
};

export default SurvivalPrediction;