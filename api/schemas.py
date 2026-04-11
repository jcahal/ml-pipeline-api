from pydantic import BaseModel

class HealthResponse(BaseModel):
  status: str

class MLflowMetrics(BaseModel):
  training_accuracy_score : float
  training_f1_score       : float
  training_precision_score: float
  training_recall_score   : float
  training_roc_auc_score  : float
  training_log_loss       : float
  training_score          : float

class ModelInfoResponse(BaseModel):
  model_name     : str
  model_version  : str
  run_id         : str
  experiment_id  : int
  metrics        : MLflowMetrics
  params         : dict


class PredictionRequest(BaseModel):
  age                : int
  tenure_months      : int
  monthly_spend      : float 
  num_support_tickets: int
  region             : str
  plan_type          : str
  payment_method     : str

class PredictionResponse(BaseModel):
  prediction: int
  confidence: float