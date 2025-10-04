// PATH: /frontend/src/components/PatientTable.jsx
import React, { useState, useEffect } from 'react';
import axios from 'axios';

const PatientTable = ({ newPrediction }) => {
  // ... (state and useEffect hooks are the same) ...
  const [patients, setPatients] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchPatients = async () => {
      try {
        const { data } = await axios.get('http://localhost:3001/api/patients');
        setPatients(data);
      } catch (error) {
        console.error("Could not fetch patients", error);
      } finally {
        setLoading(false);
      }
    };
    fetchPatients();
  }, []);

  useEffect(() => {
    if (newPrediction) {
      setPatients(prev => [newPrediction, ...prev]);
    }
  }, [newPrediction]);

  if (loading) return <p className="text-center mt-4 text-muted">Loading patient history...</p>;


  return (
    <div className="mt-5">
      <h2 className="h4">Patient Prediction History</h2>
      <div className="card shadow-sm mt-3">
        <div className="table-responsive">
          <table className="table table-striped table-hover mb-0">
            <thead className="table-light">
              <tr>
                <th scope="col">Date</th>
                {/* --- NEW COLUMN HEADER --- */}
                <th scope="col">Model Used</th>
                <th scope="col">Age</th>
                <th scope="col">Sex</th>
                <th scope="col">Health Status</th>
                <th scope="col">Hospital Stays</th>
                <th scope="col" className="text-end">Predicted Expenditure</th>
              </tr>
            </thead>
            <tbody>
              {patients.length > 0 ? patients.map((p) => (
                <tr key={p._id}>
                  <td>{new Date(p.createdAt).toLocaleDateString()}</td>
                  {/* --- NEW DATA CELL --- */}
                  <td><span className="badge bg-secondary">{p.modelUsed}</span></td>
                  <td>{p.Age}</td>
                  <td>{p.Sex === 1 ? 'Male' : 'Female'}</td>
                  <td>{p.HealthStatus}</td>
                  <td>{p.HospitalDischarges}</td>
                  <td className="text-end fw-bold text-success">
                    ${p.predictedExpenditure.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                  </td>
                </tr>
              )) : (
                <tr>
                  <td colSpan="7" className="text-center text-muted py-5">No prediction history found.</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default PatientTable;