import { useEffect, useMemo, useState } from "react";
import axios from "axios";
import "./App.css";
import AgroAI from "./AgroAI.png";
import bgImage from "./bg.png";
import sampleLeaf from "./leaf.jpg";

const API_BASE_URL =
  (process.env.REACT_APP_API_URL || "http://localhost:8000").replace(/\/$/, "");

const DISEASE_CARDS = [
  {
    title: "Early Blight",
    accent: "Amber",
    description:
      "Detects concentric brown spotting patterns so farmers can respond before field spread accelerates.",
  },
  {
    title: "Late Blight",
    accent: "Crimson",
    description:
      "Flags severe moisture-driven lesions that can damage yield quickly if not isolated early.",
  },
  {
    title: "Healthy",
    accent: "Emerald",
    description:
      "Confirms leaf images that appear free from major visible blight symptoms in the trained classes.",
  },
];

const WORKFLOW_STEPS = [
  "Upload a clear potato leaf image from your phone or laptop.",
  "Send the image to the FastAPI inference service for preprocessing.",
  "Return the predicted class and confidence score in seconds.",
];

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [status, setStatus] = useState("idle");

  useEffect(() => {
    if (!selectedFile) {
      setPreviewUrl("");
      return undefined;
    }

    const objectUrl = URL.createObjectURL(selectedFile);
    setPreviewUrl(objectUrl);

    return () => {
      URL.revokeObjectURL(objectUrl);
    };
  }, [selectedFile]);

  const confidence = useMemo(() => {
    if (!result?.confidence && result?.confidence !== 0) {
      return null;
    }

    const rawValue = Number(result.confidence);
    if (Number.isNaN(rawValue)) {
      return null;
    }

    return rawValue <= 1 ? (rawValue * 100).toFixed(2) : rawValue.toFixed(2);
  }, [result]);

  const handleFileSelection = (event) => {
    const file = event.target.files?.[0];

    setError("");
    setResult(null);

    if (!file) {
      setSelectedFile(null);
      return;
    }

    if (!file.type.startsWith("image/")) {
      setSelectedFile(null);
      setError("Please upload a valid image file.");
      return;
    }

    if (file.size > 5 * 1024 * 1024) {
      setSelectedFile(null);
      setError("Please upload an image smaller than 5 MB.");
      return;
    }

    setSelectedFile(file);
  };

  const handleUseSample = async () => {
    setError("");
    setResult(null);

    const response = await fetch(sampleLeaf);
    const blob = await response.blob();
    const file = new File([blob], "sample-potato-leaf.jpg", { type: blob.type });
    setSelectedFile(file);
  };

  const handleReset = () => {
    setSelectedFile(null);
    setResult(null);
    setError("");
    setStatus("idle");
  };

  const handleAnalyze = async () => {
    if (!selectedFile) {
      setError("Upload a leaf image or try the sample image first.");
      return;
    }

    setStatus("uploading");
    setError("");
    setResult(null);

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await axios.post(`${API_BASE_URL}/predict`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
        timeout: 30000,
      });

      setResult(response.data);
      setStatus("success");
    } catch (requestError) {
      const message =
        requestError?.response?.data?.detail ||
        requestError?.response?.data?.error ||
        "Prediction service is unavailable right now. Please try again in a moment.";

      setError(message);
      setStatus("error");
    }
  };

  return (
    <div className="app-shell">
      <div
        className="hero-backdrop"
        style={{ backgroundImage: `linear-gradient(135deg, rgba(10, 35, 24, 0.85), rgba(36, 19, 10, 0.82)), url(${bgImage})` }}
      />

      <header className="topbar">
        <div className="brand-mark">
          <img src={AgroAI} alt="AgroScan logo" className="brand-logo" />
          <div>
            <p className="eyebrow">Portfolio Project</p>
            <h1>AgroScan</h1>
          </div>
        </div>

        <a className="ghost-link" href="#analyzer">
          Open Live Analyzer
        </a>
      </header>

      <main className="page-content">
        <section className="hero-grid">
          <div className="hero-copy">
            <span className="hero-chip">AI for crop health screening</span>
            <h2>Potato disease detection with a cleaner product experience.</h2>
            <p className="hero-text">
              AgroScan is a computer-vision web application that helps classify
              potato leaf images into Early Blight, Late Blight, or Healthy
              classes using a CNN-backed inference pipeline.
            </p>

            <div className="hero-actions">
              <a className="primary-button" href="#analyzer">
                Try the analyzer
              </a>
              <button className="secondary-button" onClick={handleUseSample} type="button">
                Load sample image
              </button>
            </div>

            <div className="hero-metrics">
              <article>
                <strong>3</strong>
                <span>Prediction classes</span>
              </article>
              <article>
                <strong>FastAPI</strong>
                <span>Inference backend</span>
              </article>
              <article>
                <strong>React</strong>
                <span>Responsive web client</span>
              </article>
            </div>
          </div>

          <div className="spotlight-card">
            <p className="spotlight-label">Why this project works on a resume</p>
            <ul className="spotlight-list">
              <li>Shows an end-to-end ML product, not only a notebook.</li>
              <li>Includes real image upload, preprocessing, and live inference.</li>
              <li>Deployable as a public web demo for recruiters to test.</li>
            </ul>
          </div>
        </section>

        <section className="content-grid" id="analyzer">
          <article className="panel analyzer-panel">
            <div className="panel-heading">
              <p className="section-tag">Leaf Analyzer</p>
              <h3>Upload a potato leaf image for prediction</h3>
              <p>
                Use a clear leaf photo with visible disease patterns. The model
                is trained on three classes and returns the top prediction with
                confidence.
              </p>
            </div>

            <div className="upload-card">
              <label className="upload-dropzone" htmlFor="leaf-upload">
                {previewUrl ? (
                  <img src={previewUrl} alt="Leaf preview" className="preview-image" />
                ) : (
                  <div className="upload-placeholder">
                    <span className="upload-icon">+</span>
                    <strong>Choose an image</strong>
                    <p>PNG or JPG, up to 5 MB</p>
                  </div>
                )}
              </label>
              <input
                id="leaf-upload"
                type="file"
                accept="image/*"
                onChange={handleFileSelection}
                className="hidden-input"
              />

              <div className="upload-actions">
                <button className="primary-button" onClick={handleAnalyze} type="button">
                  {status === "uploading" ? "Analyzing..." : "Analyze leaf"}
                </button>
                <button className="secondary-button" onClick={handleReset} type="button">
                  Reset
                </button>
              </div>

              <div className="helper-row">
                <button className="text-button" onClick={handleUseSample} type="button">
                  Try a sample image
                </button>
                <span>API: {API_BASE_URL}</span>
              </div>
            </div>

            {error ? <div className="message error-message">{error}</div> : null}

            {result ? (
              <div className="result-card">
                <div>
                  <p className="result-label">Predicted class</p>
                  <h4>{result.class}</h4>
                </div>
                <div className="confidence-stack">
                  <div className="confidence-header">
                    <span>Confidence</span>
                    <strong>{confidence}%</strong>
                  </div>
                  <div className="confidence-track">
                    <div className="confidence-bar" style={{ width: `${confidence || 0}%` }} />
                  </div>
                </div>
              </div>
            ) : null}

            {status === "uploading" ? (
              <div className="message info-message">
                Processing the uploaded image and running model inference.
              </div>
            ) : null}
          </article>

          <aside className="panel insight-panel">
            <div className="panel-heading">
              <p className="section-tag">Overview</p>
              <h3>Designed like a deployable product</h3>
            </div>

            <div className="mini-grid">
              {DISEASE_CARDS.map((card) => (
                <article className="mini-card" key={card.title}>
                  <span>{card.accent}</span>
                  <h4>{card.title}</h4>
                  <p>{card.description}</p>
                </article>
              ))}
            </div>
          </aside>
        </section>

        <section className="panel feature-panel">
          <div className="panel-heading">
            <p className="section-tag">System Flow</p>
            <h3>What happens behind the interface</h3>
          </div>

          <div className="workflow-grid">
            {WORKFLOW_STEPS.map((step, index) => (
              <article className="workflow-card" key={step}>
                <span className="step-index">0{index + 1}</span>
                <p>{step}</p>
              </article>
            ))}
          </div>

          <div className="footer-note">
            <strong>Note:</strong> AgroScan is an AI-assisted screening tool for
            demonstration and educational use. Field decisions should still be
            validated with agronomy expertise.
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;
