from fastapi import FastAPI
from mlflow.client import MlflowClient

import config
from .schemas import HealthResponse, MLflowMetrics, ModelInfoResponse, PredictionRequest
from .model import load_model, predict

app = FastAPI()
client = MlflowClient()  # reads from ./mlruns by default

@app.on_event('startup')
async def startup():
  load_model()

@app.get('/health', response_model=HealthResponse)
async def get_health():
  return HealthResponse(status='ok')

@app.get('/model-info', response_model=ModelInfoResponse)
async def get_model_info():
  experiment = client.get_experiment_by_name(f'{config.MODEL_NAME}')
  
  last_run = client.search_runs(
    experiment_ids=[experiment.experiment_id], 
    order_by=["start_time DESC"], # explicit order to get the latest run
    max_results=1
  )[0]

  metrics = MLflowMetrics(
    training_accuracy_score  = last_run.data.metrics['training_accuracy_score'],
    training_f1_score        = last_run.data.metrics['training_f1_score'],
    training_precision_score = last_run.data.metrics['training_precision_score'],
    training_recall_score    = last_run.data.metrics['training_recall_score'],
    training_roc_auc_score   = last_run.data.metrics['training_roc_auc'],
    training_log_loss        = last_run.data.metrics['training_log_loss'],
    training_score           = last_run.data.metrics['training_score'],
  )

  return ModelInfoResponse(
    model_name     = config.MODEL_NAME,
    model_version  = config.MODEL_VERSION,
    run_id         = last_run.info.run_id,
    experiment_id  = experiment.experiment_id,
    metrics        = metrics,
    params         = last_run.data.params,
  )

@app.post('/predict')
async def get_prediction(request: PredictionRequest):
  return predict(request)