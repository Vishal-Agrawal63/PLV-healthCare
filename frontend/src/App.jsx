// PATH: /frontend/src/App.jsx
import React from 'react';
// Correctly import Routes instead of Switch
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import Navbar from './components/Navbar';
import PLVPrediction from './pages/PLVPrediction';
import SurvivalPrediction from './pages/SurvivalPrediction';

function App() {
  return (
    <Router>
      <div style={{ backgroundColor: '#f8f9fa', minHeight: '100vh' }}>
        <Navbar />
        <main className="container py-4">
          {/* Use the <Routes> component instead of <Switch> */}
          <Routes>
            <Route path="/survival" element={<SurvivalPrediction />} />
            <Route path="/" element={<PLVPrediction />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;