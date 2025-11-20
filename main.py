import joblib
import numpy as np
import neurokit2 as nk2
from fastapi import FastAPI, HTTPException
from scipy.signal import welch
from pydantic import BaseModel
from typing import List
import asyncio
from concurrent.futures import ThreadPoolExecutor
import os
import warnings
import traceback


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

def extract_features(segment, fs=250):
    segment_list = np.array(segment)
    segment_clean = segment_list[~np.isnan(segment_list)]
    if len(segment_clean) == 0:
        return [0] * 5  # 5 fitur

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            ecg_signals, info = nk2.ecg_peaks(pd.Series(segment_clean), sampling_rate=fs)
            peaks = info["ECG_R_Peaks"]
    except:
        peaks = np.array([])

    mean_hr = 0
    sdnn = 0
    if len(peaks) >= 2:
        rr = np.diff(peaks) / fs
        hr = 60.0 / rr
        mean_hr = np.mean(hr)
        sdnn = np.std(rr)

    qrs_width = 0
    if len(peaks) > 0:
        widths = []
        for p in peaks:
            start = max(0, p - int(0.05 * fs))
            end = min(len(segment_clean), p + int(0.05 * fs))
            diff = np.abs(np.diff(segment_clean[start:end]))
            if len(diff) > 0:
                widths.append(np.std(diff))
        if len(widths) > 0:
            qrs_width = np.mean(widths)

    spectral_entropy = 0
    try:
        freqs, power = welch(segment_clean, fs)
        power_norm = power / np.sum(power)
        spectral_entropy = -np.sum(power_norm * np.log2(power_norm + 1e-12))
    except:
        spectral_entropy = 0

    rms_amp = np.sqrt(np.mean(segment_clean**2))

    return [mean_hr, sdnn, qrs_width, spectral_entropy, rms_amp]

def run_inference(signal: List[float]):
    """
    Wrapper function to run extraction and prediction synchronously
    so it can be offloaded to a thread.
    """
    if clf is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # 1. Preprocess
    # The notebook uses fs=250 as default
    print(f"Feature {len(signal)}")
    features = extract_features(signal, fs=250)
    print("Feature", features)
    
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
        traceback.print_exc()
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))