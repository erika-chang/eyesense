import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from utils import prepara_data, load_model

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

# http://127.0.0.1:8000/predict?image
@app.get("/predict")
def predict(image):
    """
    Make a single course prediction.
    """
    X = prepare_data(image)
    
    model = load_model()
    result = dict(classification=model.predict(X))

    return result

@app.get("/")
def root():
    return {'Is it working?': True}