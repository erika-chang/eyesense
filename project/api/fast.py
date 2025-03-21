import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
#from taxifare.ml_logic.preprocessor import preprocess_features
#from taxifare.ml_logic.registry import load_model


app = FastAPI()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# http://127.0.0.1:8000/predict?pickup_datetime=2014-07-06+19:18:00&pickup_longitude=-73.950655&pickup_latitude=40.783282&dropoff_longitude=-73.984365&dropoff_latitude=40.769802&passenger_count=2
@app.get("/predict")
def predict(image):
    """
    Make a single course prediction.
    """
    
    X = load(image)

    model = load_model()
    result = dict(classification=model.predict(X))

    return result

@app.get("/")
def root():
    return {'ok': True}
