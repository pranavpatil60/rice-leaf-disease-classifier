from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import shutil, os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# FastAPI app
app = FastAPI()

# Mount static folder
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Classes
classes = ['Leaf smut', 'Brown spot', 'Bacterial leaf blight']

# ML model
model = None

# Load model on startup
@app.on_event("startup")
def load_ml_model():
    global model
    try:
        model = load_model("rice_leaf_disease_model.keras")
        print("✅ Model loaded successfully")
    except Exception as e:
        print("❌ Model load failed:", e)

# Home page
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Prediction endpoint
@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    # Upload folder
    upload_folder = "static/uploads"
    os.makedirs(upload_folder, exist_ok=True)

    # Save file
    file_path = os.path.join(upload_folder, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Preprocess image
    img = image.load_img(file_path, target_size=(224, 224))
    x = image.img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)

    # Predict
    if model:
        pred = model.predict(x)
        result = classes[np.argmax(pred)]
    else:
        result = "Model not loaded"

    return templates.TemplateResponse(
        "result.html",
        {"request": request, "prediction": result, "filename": file.filename}
    )
