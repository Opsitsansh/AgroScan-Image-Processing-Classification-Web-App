# AgroScan

AgroScan is a potato leaf disease classification web application built as an end-to-end machine learning product. It allows users to upload a leaf image, sends it to a FastAPI inference service, and returns a prediction for one of three classes:

- Early Blight
- Late Blight
- Healthy

The goal of the project is to move beyond a notebook-only ML workflow and present a deployable product experience that recruiters and reviewers can try directly in the browser.

## Highlights

- Professional React frontend designed for portfolio presentation
- FastAPI backend for live image inference
- TensorFlow/Keras model integration
- Upload preview, confidence score, and clear prediction feedback
- Environment-based configuration for deployment
- Ready to split across Vercel frontend and Render backend

## Tech Stack

- Frontend: React, CSS, Axios
- Backend: Python, FastAPI, Uvicorn
- ML: TensorFlow, Keras, NumPy, Pillow
- Deployment: Vercel, Render, Docker-ready backend structure

## Project Structure

```text
frontend/
  src/
    App.js
    App.css
    index.js
api/
  main.py
  requirements.txt
  runtime.txt
saved_models/
potatoes.h5
```

## Local Development

### Frontend

```bash
cd frontend
npm install
npm start
```

Create a `.env` file in `frontend/`:

```env
REACT_APP_API_URL=http://localhost:8000
```

### Backend

```bash
cd api
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

Optional `.env` values for `api/`:

```env
ALLOWED_ORIGINS=http://localhost:3000
# MODEL_PATH=/absolute/path/to/model
```

## Deployment Plan

### Frontend on Vercel

- Root directory: `frontend`
- Build command: `npm run build`
- Output directory: `build`
- Environment variable: `REACT_APP_API_URL=https://your-render-service.onrender.com`

### Backend on Render

- Root directory: `api`
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- Environment variable: `ALLOWED_ORIGINS=https://your-vercel-app.vercel.app`

## Why This Project Stands Out

- It combines computer vision with a real web product workflow
- It demonstrates frontend, backend, and ML integration in one project
- It is easy for recruiters to test from a public link
- It is structured to be deployable instead of staying limited to local notebooks

## Suggested Resume Description

Built and deployed an end-to-end potato leaf disease classification web application using React, FastAPI, TensorFlow, and Docker-ready backend architecture, enabling real-time image upload, inference, and confidence-based prediction.
