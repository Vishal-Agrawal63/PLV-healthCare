// PATH: /backend/models/Patient.js
const mongoose = require('mongoose');

const patientSchema = new mongoose.Schema({
    // (All other fields remain the same)
    Age: { type: Number, required: true },
    Sex: { type: Number, required: true },
    Race: { type: Number, required: true },
    PovertyCategory: { type: Number, required: true },
    InsuranceCoverage: { type: Number, required: true },
    HealthStatus: { type: Number, required: true },
    OfficeVisits: { type: Number, required: true },
    OutpatientVisits: { type: Number, required: true },
    ERVisits: { type: Number, required: true },
    HospitalDischarges: { type: Number, required: true },

    predictedExpenditure: { type: Number, required: true },

    // --- NEW FIELD ---
    modelUsed: { type: String, required: true },

    createdAt: { type: Date, default: Date.now },
});

const Patient = mongoose.model('Patient', patientSchema);
module.exports = Patient;