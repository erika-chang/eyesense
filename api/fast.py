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

# Initialize at startup
@app.on_event("startup")
async def startup_event():
    model = load_model()

@app.post("/predict")
async def predict(data: dict):
    # Use model for prediction
    result = dict(classification=model.predict(data))

    return result

@app.get("/")
def root():
    return {'Is it working?': True}