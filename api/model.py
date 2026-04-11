import joblib
import pandas as pd

import config
from .schemas import PredictionRequest, PredictionResponse
from pipeline.transformer import DataTransformer

_clf = None

def load_model():
  global _clf 
  _clf = joblib.load(config.MODEL_FILE)

def predict(request: PredictionRequest) -> PredictionResponse:
  df_req = pd.DataFrame([request.model_dump()])

  # replicate the training transformations manually — I didn't serialize the fitted scaler,
  # so we reconstruct the feature vector here to match what the model was trained on
  candidate = pd.DataFrame([{
    "age"                   : df_req.iloc[0].age,
    "tenure_months"         : df_req.iloc[0].tenure_months,
    "monthly_spend"         : df_req.iloc[0].monthly_spend,
    "num_support_tickets"   : df_req.iloc[0].num_support_tickets,
    "plan_type"             : config.ORDINAL_ENCODING['plan_type'][df_req.iloc[0].plan_type],
    "region_East"           : df_req.iloc[0].region == 'East',
    "region_Midwest"        : df_req.iloc[0].region == 'Midwest',
    "region_South"          : df_req.iloc[0].region == 'South',
    "region_West"           : df_req.iloc[0].region == 'West',
    "payment_bank_transfer" : df_req.iloc[0].payment_method == 'bank_transfer',
    "payment_credit_card"   : df_req.iloc[0].payment_method == 'credit_card',
    "payment_paypal"        : df_req.iloc[0].payment_method == 'paypal'
  }])
  print(candidate)

  prediction = int(_clf.predict(candidate)[0])
  confidence = round(float(_clf.predict_proba(candidate)[0][prediction]), 2)  # index by predicted class to get its probability

  return PredictionResponse(prediction=prediction, confidence=confidence)