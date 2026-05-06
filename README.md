# ML Pipeline API

A FastAPI service that wraps a scikit-learn classification model for customer churn prediction. Accepts raw customer data, runs it through a preprocessing pipeline, and returns a prediction with confidence score. Containerized with Docker and tracked with MLflow.

---

## Stack

- **FastAPI** — REST API framework
- **scikit-learn** — model training and preprocessing
- **MLflow** — experiment tracking
- **Docker** — containerization
- **pandas** — data handling
- **pydantic** — request/response validation

---

## Project Structure

```
ml-pipeline-api/
├── pipeline/
│   ├── loader.py          # CSV validation and loading
│   ├── transformer.py     # Normalization and encoding
│   └── trainer.py         # Model training with MLflow
├── api/
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
└── requirements.txt
```

---

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Train the model
python -m pipeline.trainer

# Start the API
uvicorn api.main:app --reload
```

Swagger UI available at `http://localhost:8000/docs`.

---

## Docker

```bash
# API only (port 80)
docker compose up --build

# Full dev environment (API + MLflow + Jupyter Lab + shell)
docker compose --profile dev up --build

# Run the API image directly
docker build -t ml-pipeline-api .
docker run -p 8000:8000 ml-pipeline-api
```

### Dev services (`--profile dev`)

| Service | URL / access | Description |
|---------|-------------|-------------|
| `mlflow` | `http://localhost:5001` | MLflow tracking UI |
| `notebook` | `http://localhost:8888` | Jupyter Lab (token printed in logs on first start) |
| `shell` | `docker compose exec shell bash` | Python 3.11 container with project mounted at `/app` |

---

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Service health check |
| GET | `/model-info` | Model name, version, and training metrics |
| POST | `/predict` | Churn prediction for a single customer |

---

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

`prediction` is `0` (no churn) or `1` (churn). `confidence` is the model's probability for that class.

---

## MLflow

When running via `docker compose`, the tracking UI is available at `http://localhost:5001`.

To run it locally:

```bash
mlflow ui
open http://localhost:5000
```

---

## Tests

```bash
pytest tests/
```

Requires the model to be trained (`python -m pipeline.trainer`) before running.