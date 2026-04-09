# ML Pipeline API

A FastAPI service that wraps a scikit-learn classification model for customer churn prediction. Accepts raw customer data, runs it through a preprocessing pipeline, and returns a prediction with confidence score. Containerized with Docker and tracked with MLflow.

---

## Stack

- **FastAPI** — REST API framework
- **scikit-learn** — model training and preprocessing
- **MLflow** — experiment tracking and model registry
- **Docker** — containerization
- **pandas** — data handling
- **pydantic** — request/response validation

---

## Project Structure

```
ml-pipeline-api/
├── pipeline/
│   ├── __init__.py
│   ├── loader.py          # DataLoader class (from Project 1)
│   ├── transformer.py     # DataTransformer class (from Project 1)
│   └── trainer.py         # ModelTrainer class
├── api/
│   ├── __init__.py
│   ├── main.py            # FastAPI app and routes
│   ├── schemas.py         # Pydantic request/response models
│   └── model.py           # Model load/predict logic
├── data/
│   └── sample.csv
├── models/                # Serialized model artifacts
├── mlruns/                # MLflow tracking data
├── tests/
│   └── test_api.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Development Plan

### Step 1 — Project scaffold
Set up the folder structure above. Create all empty `__init__.py` files. Copy `loader.py` and `transformer.py` in from the CSV pipeline project. Initialize a git repo and make the first commit.

### Step 2 — Train and serialize a model
Create `pipeline/trainer.py` with a `ModelTrainer` class. It should load `sample.csv`, run it through `DataTransformer`, train a `RandomForestClassifier` on the churn target, evaluate it (accuracy, precision, recall), and serialize the trained model to `models/churn_model.pkl` using `joblib`. Run it as a standalone script first — `python -m pipeline.trainer` — before wiring it into the API.

### Step 3 — Add MLflow tracking
Wrap the training run in `mlflow.start_run()`. Log hyperparameters with `mlflow.log_param()`, metrics with `mlflow.log_metric()`, and the model artifact with `mlflow.sklearn.log_model()`. Launch the MLflow UI with `mlflow ui` and confirm your run appears before moving on.

### Step 4 — Build the Pydantic schemas
In `api/schemas.py`, define a `PredictionRequest` model with the input fields matching your CSV columns (age, tenure_months, monthly_spend, etc.) and a `PredictionResponse` model with `prediction` (int) and `confidence` (float). Pydantic handles input validation automatically — if a required field is missing or the wrong type, FastAPI returns a clean 422 error with no extra work from you.

### Step 5 — Build the model loader
In `api/model.py`, write a `load_model()` function that deserializes `models/churn_model.pkl` and a `predict()` function that accepts a `PredictionRequest`, runs the transformer, and returns a `PredictionResponse`. Keep all model logic here — the API routes should stay thin.

### Step 6 — Build the FastAPI routes
In `api/main.py`, create the FastAPI app and define three endpoints:
- `GET /health` — returns `{"status": "ok"}` — used by Docker and cloud platforms to check if the service is running
- `GET /model-info` — returns model name, version, and training metrics
- `POST /predict` — accepts a `PredictionRequest`, calls `predict()`, returns a `PredictionResponse`

### Step 7 — Test it locally
Run the server with `uvicorn api.main:app --reload` and open `http://localhost:8000/docs`. FastAPI auto-generates a Swagger UI where you can send test requests directly from the browser. Confirm all three endpoints work before touching Docker.

### Step 8 — Containerize
Update the `Dockerfile` to run the FastAPI server instead of `main.py`. Build the image and confirm the Swagger UI is reachable at `http://localhost:8000/docs` from inside the container. Add a `docker-compose.yml` for local development convenience.

### Step 9 — Write one test
In `tests/test_api.py`, use FastAPI's built-in `TestClient` to write at least one test for the `/predict` endpoint — a valid request that returns a 200, and a malformed request that returns a 422. One test file with two tests is enough to show you understand the pattern.

### Step 10 — Clean up and document
Update this README with real setup instructions, example curl commands, and a screenshot of the MLflow UI showing your training run. This is what a hiring manager will look at first.

---

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Train the model
python -m pipeline.trainer

# Start the API
uvicorn api.main:app --reload

# Open the docs
open http://localhost:8000/docs
```

## Docker

```bash
# Build
docker build -t ml-pipeline-api .

# Run
docker run -p 8000:8000 ml-pipeline-api

# Open the docs
open http://localhost:8000/docs
```

## Example Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 34,
    "tenure_months": 24,
    "monthly_spend": 89.99,
    "num_support_tickets": 1,
    "region": "West",
    "plan_type": "Premium",
    "payment_method": "credit_card"
  }'
```

## Example Response

```json
{
  "prediction": 0,
  "confidence": 0.83
}
```

---

## MLflow

Start the tracking UI to inspect experiment runs:

```bash
mlflow ui
open http://localhost:5000
```

---

## Status

- [x] Step 1 — Project scaffold
- [x] Step 2 — Train and serialize model
- [x] Step 3 — MLflow tracking
- [x] Step 4 — Pydantic schemas
- [ ] Step 5 — Model loader
- [ ] Step 6 — FastAPI routes
- [ ] Step 7 — Local testing
- [ ] Step 8 — Docker
- [ ] Step 9 — Tests
- [ ] Step 10 — Docs and cleanup