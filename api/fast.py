from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from . import utils
from PIL import Image
import io
import numpy as np
import tensorflow as tf
import operator

# Receberia a imagem
# Prepararia ela (size, normalization, tensor)
# Load model
# Predict

app = FastAPI()
app.state.model = utils.load_model_from_gcp()
# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.post("/predict")
async def predict(img: UploadFile = File(...)):
    try:
        # Ler e converter a imagem para um formato utilizável
        contents = await img.read()
        image = Image.open(io.BytesIO(contents))

        X = utils.prepare_image(image, 299, 299)
         
        CLASS_NAMES = ['cataract', 'degeneration', 'diabets', 'glaucoma', 'hypertension', 'myopia', 'normal']
        predictions = app.state.model.predict(X)
        
        dict_pred = {}
        j = 0
        for i in predictions[0]:
            dict_pred[CLASS_NAMES[j]] = round(float(i),4)
            j += 1

        pred_list = sorted(dict_pred.items(), key=operator.itemgetter(1), reverse=True)

        return JSONResponse(content={"result":pred_list}, status_code=200)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/")
def root():
    return {'Is it working?': True}
