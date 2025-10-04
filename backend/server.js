// PATH: /backend/server.js
const express = require('express');
const cors = require('cors');
const connectDB = require('./config/db');
const patientRoutes = require('./routes/patientRoutes');

// Connect to Database
connectDB();

const app = express();

// Middleware
app.use(cors());
app.use(express.json()); // To accept JSON data in the body

// API Routes
app.use('/api', patientRoutes);

const PORT = process.env.PORT || 3001;

app.listen(PORT, () => console.log(`Backend server running on port ${PORT}`));