// PATH: /frontend/src/components/Navbar.jsx
import React from 'react';
import { NavLink } from 'react-router-dom';

const Navbar = () => {
  return (
    <nav className="navbar navbar-expand-lg navbar-dark bg-dark">
      <div className="container">
        <NavLink className="navbar-brand" to="/">Health Prediction Suite</NavLink>
        <div className="collapse navbar-collapse">
          <ul className="navbar-nav me-auto mb-2 mb-lg-0">
            <li className="nav-item">
              {/* --- FIX #1: Update NavLink for React Router v6 --- */}
              <NavLink 
                className={({ isActive }) => "nav-link" + (isActive ? " active" : "")} 
                to="/"
              >
                PLV Prediction
              </NavLink>
            </li>
            <li className="nav-item">
              {/* --- FIX #2: Update NavLink for React Router v6 --- */}
              <NavLink 
                className={({ isActive }) => "nav-link" + (isActive ? " active" : "")} 
                to="/survival"
              >
                DL Survival Analysis
              </NavLink>
            </li>
          </ul>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;