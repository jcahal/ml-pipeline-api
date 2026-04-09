from pydantic import BaseModel

class PredictionRequest(BaseModel):
  age                 : int
  tenure_months       : int
  monthly_spend       : float 
  num_support_tickets : int
  region              : str
  plan_type           : str
  payment_method      : str

class PredictionResponse(BaseModel):
  prediction: int
  confidence: float