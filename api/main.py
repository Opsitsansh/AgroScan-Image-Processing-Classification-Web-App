import os
from pathlib import Path

import numpy as np
import tensorflow as tf
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_CANDIDATES = [
    BASE_DIR / "saved_models" / "1",
    BASE_DIR / "potatoes.h5",
]
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]
IMAGE_SIZE = (256, 256)

app = FastAPI(title="AgroScan API", version="1.0.0")


def get_allowed_origins():
    origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000")
    if origins.strip() == "*":
        return ["*"]
    return [origin.strip() for origin in origins.split(",") if origin.strip()]


app.add_middleware(
    CORSMiddleware,
    allow_origins=get_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None


def resolve_model_path():
    configured_path = os.getenv("MODEL_PATH")
    if configured_path:
        model_path = Path(configured_path).expanduser().resolve()
        if model_path.exists():
            return model_path
        raise FileNotFoundError(f"Configured MODEL_PATH does not exist: {model_path}")

    for candidate in DEFAULT_MODEL_CANDIDATES:
        if candidate.exists():
            return candidate

    raise FileNotFoundError("No compatible model file was found.")


def load_model():
    global model
    if model is None:
        model_path = resolve_model_path()
        model = tf.keras.models.load_model(model_path)
    return model


def read_file_as_image(data) -> np.ndarray:
    image = Image.open(data).convert("RGB").resize(IMAGE_SIZE)
    return np.array(image).astype("float32") / 255.0


@app.on_event("startup")
def startup_event():
    load_model()


@app.get("/")
async def root():
    return {"message": "AgroScan API is running"}


@app.get("/ping")
async def ping():
    return {"message": "alive"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload a valid image file.")

    try:
        image = read_file_as_image(file.file)
        img_batch = np.expand_dims(image, axis=0)
        prediction = load_model().predict(img_batch, verbose=0)[0]
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Please try another image.",
        ) from error

    predicted_class = CLASS_NAMES[int(np.argmax(prediction))]
    confidence = float(np.max(prediction))

    return {"class": predicted_class, "confidence": confidence}


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
