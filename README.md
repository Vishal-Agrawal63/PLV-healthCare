# Project: Patient Lifetime Value Prediction for Hospitals/Clinics

This is a full-stack application designed to predict the annual healthcare expenditure of a patient based on demographic and health data from the MEPS HC-216 dataset. Total annual expenditure serves as a proxy for Patient Lifetime Value (PLV).

## Technology Stack
- **Database**: MongoDB (local instance)
- **Backend**: Node.js (Express.js + Mongoose)
- **Frontend**: React (with TailwindCSS)
- **ML Layer**: Python (Flask for model API)
- **Deployment**: Local Windows machine, no Docker

## File Structure
```
/project-root
    /backend
        /routes
            patientRoutes.js
        /models
            Patient.js
        server.js
        package.json
    /frontend
        /src
            /components
                PatientForm.jsx
                PatientTable.jsx
                PredictionResult.jsx
            App.jsx
            index.js
            index.css
        package.json
        tailwind.config.js
    /ml
        preprocess.py
        train_model.py
        model.pkl
        model_columns.pkl
        api.py
        requirements.txt
        h216.csv  (NOTE: You must add this file)
    /config
        db.js
    README.md
```

---

## Windows Local Setup Instructions

### Prerequisites
1.  **Node.js**: Install from the official website.
2.  **Python**: Install Python 3.8+ and ensure it's added to your PATH.
3.  **MongoDB**: Install MongoDB Community Server and ensure the service is running.
4.  **MEPS Dataset**: Download the **MEPS HC-216 2019 data file**. Convert it to a CSV and name it `h216.csv`. Place this file inside the `/ml` directory.

### Step 1: Python (ML Layer)

Open a terminal in the project's root directory.

```bash
# Navigate to the ml directory
cd ml

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
.\venv\Scripts\activate

# Install required Python packages
pip install -r requirements.txt

# Run the preprocessing script (requires h216.csv)
python preprocess.py
python preprocess_dl.py 

# Run the model training script
python train_model.py
python train_dl.py 

# Run the Flask API server (leave this terminal running)
python api.py
python api_dl.py  
# It should now be running on http://127.0.0.1:5000
```

### Step 2: Node.js Backend

Open a **new, second terminal** in the project's root directory.

```bash
# Navigate to the backend directory
cd backend

# Install npm packages
npm install

# Start the Node.js server (leave this terminal running)
npm start
# It should now be running on http://localhost:3001
```

### Step 3: React Frontend

Open a **new, third terminal** in the project's root directory.

```bash
# Navigate to the frontend directory
cd frontend

# Install npm packages
npm install

# Start the React development server
npm start
# Your browser should open to http://localhost:3000
```

### Step 4: MongoDB Connection

-   Ensure your MongoDB service is running.
-   The connection string in `/config/db.js` is set to `mongodb://localhost:27017/plv_app`. This is the default and should work without changes if MongoDB is installed locally.

Your application is now fully running! You can interact with the form on the frontend to get predictions.



### How to Run Your Application Now

From now on, you only need to follow these steps:

1.  **First-Time Setup (if you haven't already):**
    *   Make sure all Node.js dependencies are installed in both sub-projects:
        ```bash
        npm install --prefix frontend
        npm install --prefix backend
        ```
    *   Make sure your Python virtual environment and its packages are set up:
        ```bash
        # In /ml directory
        python -m venv venv
        .\venv\Scripts\activate
        pip install -r requirements.txt
        
        * datasets used:
        ML : h216.csv
        DL : support_cleaned.csv 
        
        after that do:
        `cd .\ml\
        python .\preprocess.py
        python .\preprocess_dl.py 
        python .\train_dl.py    
        python .\train_model.py
        ```

2.  **Start the Entire Application:**
    *   Open a **single terminal** in your **project-root** directory.
    *   Run the single command:

        ```bash
        npm run dev
        ```
        
        <!-- This will concurrently start: -->
The React Frontend on http://localhost:3000
The Node.js Backend on http://localhost:3001
The Python ML API on http://localhost:5000

Your terminal will come to life with the logs from the backend server, the React development server, and the Python Flask API, all running together.

To **stop all services**, simply press `Ctrl+C` in that one terminal. `concurrently` will ensure all three processes are terminated cleanly.