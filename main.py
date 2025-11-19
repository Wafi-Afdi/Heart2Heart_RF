import joblib
import numpy as np
import neurokit2 as nk2
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import asyncio
from concurrent.futures import ThreadPoolExecutor
import os

# Initialize FastAPI
app = FastAPI(title="ECG Diagnosis API")

# Define Input Schema
class ECGInput(BaseModel):
    ecgSignal: List[float]

# Define Output Schema
class DiagnosisResponse(BaseModel):
    result: str

# Global variables for model
MODEL_PATH = "./models/cardiacArrest.pkl"
clf = None

# Executor for CPU-bound tasks (Feature extraction + Prediction)
executor = ThreadPoolExecutor()

@app.on_event("startup")
def load_model():
    global clf
    if os.path.exists(MODEL_PATH):
        clf = joblib.load(MODEL_PATH)
        print("Model loaded successfully.")
    else:
        print(f"Error: Model not found at {MODEL_PATH}")

def extract_features(segment_list: List[float], fs=250) -> List[float]:
    """
    Replicates the exact logic from the Jupyter Notebook cell 13.
    """
    # Convert list to numpy array for processing
    segment = np.array(segment_list)
    
    features = []
    
    # --- 1. HR Variability Features (via NeuroKit2) ---
    try:
        # Note: cleaning is skipped here as it wasn't in the notebook's extractFeatures function
        _, rpeaks = nk2.ecg_peaks(segment, sampling_rate=fs)
        
        # Calculate RR intervals in seconds
        rr = np.diff(rpeaks["ECG_R_Peaks"]) / fs
        
        if len(rr) < 2:
            rr = np.array([0])
    except Exception:
        rr = np.array([0])

    mean_rr = np.mean(rr)
    median_rr = np.median(rr)
    sdnn = np.std(rr)
    features += [mean_rr, median_rr, sdnn]

    # --- 2. Statistical Features (Manual calculation per notebook) ---
    mean_val = np.mean(segment)
    std_val = np.std(segment)
    
    # Skewness calculation from notebook
    skew_val = np.mean((segment - mean_val)**3) / (std_val**3 + 1e-8)
    
    # Kurtosis calculation from notebook
    kurt_val = np.mean((segment - mean_val)**4) / (std_val**4 + 1e-8)
    
    min_val = np.min(segment)
    max_val = np.max(segment)
    range_val = max_val - min_val

    features += [mean_val, std_val, skew_val, kurt_val, min_val, max_val, range_val]
    
    return features

def run_inference(signal: List[float]):
    """
    Wrapper function to run extraction and prediction synchronously
    so it can be offloaded to a thread.
    """
    if clf is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # 1. Preprocess
    # The notebook uses fs=250 as default
    features = extract_features(signal, fs=250)
    
    # 2. Reshape for sklearn (1 sample, n features)
    features_reshaped = np.array(features).reshape(1, -1)
    
    # 3. Predict
    prediction = clf.predict(features_reshaped)
    
    # Return the string label (e.g., 'normal', 'afib', 'vt', 'vf')
    return prediction[0]

@app.post("/generate-diagnosis", response_model=DiagnosisResponse)
async def generate_diagnosis(input_data: ECGInput):
    """
    Asynchronous endpoint that accepts raw ECG float array,
    extracts features, and returns the diagnosis.
    """
    try:
        # Validate input length slightly (optional, but good practice)
        if not input_data.ecgSignal:
            raise HTTPException(status_code=400, detail="Input signal cannot be empty")

        # Run CPU-bound inference in a separate thread to verify 'Async' behavior
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(executor, run_inference, input_data.ecgSignal)
        
        return {"result": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))