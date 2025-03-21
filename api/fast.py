import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from utils import prepara_data, load_model
from google.cloud import storage
import os
import tensorflow as tf 

# Receberia a imagem
# Prepararia ela (size, normalization, tensor)
# Load model
# Predict

app = FastAPI()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Download model on startup
MODEL_BUCKET = "eyesense-model1"
MODEL_PATH = "best_model.h5"
LOCAL_MODEL_PATH = "/tmp/model.h5"

# Initialize at startup
@app.on_event("startup")
async def startup_event():
    # Download the model from GCS
    storage_client = storage.Client()
    bucket = storage_client.bucket(MODEL_BUCKET)
    blob = bucket.blob(MODEL_PATH)
    blob.download_to_filename(LOCAL_MODEL_PATH)
    
    # Load the model
    global model
    model = tf.keras.models.load_model(LOCAL_MODEL_PATH)

@app.post("/predict")
async def predict(data: dict):
    # Use model for prediction
    result = dict(classification=model.predict(data))

    return result

@app.get("/")
def root():
    return {'Is it working?': True}