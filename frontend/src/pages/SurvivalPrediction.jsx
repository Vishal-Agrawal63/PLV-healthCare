import React, { useState } from 'react';
import axios from 'axios';

const SurvivalPrediction = () => {
    // --- CHANGE: State now only includes the 5 original fields ---
    const [formData, setFormData] = useState({
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
        setFormData(prev => ({ ...prev, [name]: Number(value) }));
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);
        setError('');
        setResult(null);
        try {
            // The payload sent to the API is now the smaller, 5-feature object
            const res = await axios.post('http://localhost:3001/api/predict-survival', formData);
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
                <h2 className="h4 mb-0">Deep Learning Survival Prediction</h2>
                <p className="text-muted small mb-0">Enter patient clinical data to predict survival probability.</p>
            </div>
            <div className="card-body">
                <form onSubmit={handleSubmit}>
                    <div className="row g-3">
                        {/* --- The form now only contains the 5 original inputs --- */}
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
                        <h4 className="alert-heading">Prediction Result</h4>
                        <p className="display-5 fw-bold">{result.survival_probability}%</p>
                        <hr />
                        <p className="mb-0 small">This is the model's predicted probability that the patient will survive the current hospitalization.</p>
                    </div>
                )}
            </div>
        </div>
    );
};

export default SurvivalPrediction;