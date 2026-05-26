import os
import traceback
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
predict_fn = None
model_input_dtype = np.float32


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
    global model, predict_fn, model_input_dtype
    if model is not None and predict_fn is not None:
        return model, predict_fn, model_input_dtype

    model_path = resolve_model_path()

    try:
        model = tf.keras.models.load_model(model_path)
        if not hasattr(model, "predict"):
            raise TypeError("Loaded object does not expose a predict method.")
        model_input_dtype = np.float32

        def keras_predict(batch):
            return model.predict(batch, verbose=0)

        predict_fn = keras_predict
        return model, predict_fn, model_input_dtype
    except Exception:
        loaded = tf.saved_model.load(str(model_path))
        signature = (
            loaded.signatures.get("serving_default")
            or loaded.signatures.get("serve")
        )
        if signature is None:
            raise RuntimeError("No compatible serving signature was found in the model.")

        model = loaded
        keyword_inputs = signature.structured_input_signature[1]
        positional_inputs = signature.structured_input_signature[0]

        if keyword_inputs:
            input_name, input_spec = next(iter(keyword_inputs.items()))
            model_input_dtype = input_spec.dtype.as_numpy_dtype

            def saved_model_predict(batch):
                tensor = tf.convert_to_tensor(batch, dtype=input_spec.dtype)
                output = signature(**{input_name: tensor})
                return list(output.values())[0].numpy()

        elif positional_inputs:
            input_spec = positional_inputs[0]
            model_input_dtype = input_spec.dtype.as_numpy_dtype

            def saved_model_predict(batch):
                tensor = tf.convert_to_tensor(batch, dtype=input_spec.dtype)
                output = signature(tensor)
                return list(output.values())[0].numpy()

        else:
            raise RuntimeError("The SavedModel signature does not expose an input tensor.")

        predict_fn = saved_model_predict
        return model, predict_fn, model_input_dtype


def read_file_as_image(data, normalize=True) -> np.ndarray:
    image = Image.open(data).convert("RGB").resize(IMAGE_SIZE)
    array = np.array(image)
    if normalize:
        return array.astype("float32") / 255.0
    return array.astype("float32")


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
        _, predictor, input_dtype = load_model()
        should_normalize = np.issubdtype(input_dtype, np.floating)
        image = read_file_as_image(file.file, normalize=should_normalize)
        img_batch = np.expand_dims(image, axis=0)
        prediction = predictor(img_batch)[0]
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except Exception as error:
        print(f"Prediction error: {error}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {error}",
        ) from error

    predicted_class = CLASS_NAMES[int(np.argmax(prediction))]
    confidence = float(np.max(prediction))

    return {"class": predicted_class, "confidence": confidence}


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
