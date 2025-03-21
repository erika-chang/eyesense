import pandas as pd
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from utils import prepare_image, load_model
from PIL import Image
import io


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
@app.post("/predict")
async def predict(img: UploadFile = File(...)):
    try:
        # Ler e converter a imagem para um formato utilizável
        contents = await img.read()
        image = Image.open(io.BytesIO(contents))

        X = prepare_image(image, 256, 256)

        model = load_model()
        result = dict(classification=model.predict(X))

        return JSONResponse(content=result, status_code=200)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/")
def root():
    return {'Is it working?': True}
