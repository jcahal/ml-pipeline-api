INPUT_FILE = 'data/sample.csv'

COLUMNS = ['customer_id', 'age', 'tenure_months', 'monthly_spend', 'num_support_tickets', 'region', 'plan_type', 'payment_method', 'churned']

NUMERICAL_COLS = ['age', 'tenure_months', 'monthly_spend', 'num_support_tickets']

ONE_HOT_ENCODING = {
  'region': 'region', 
  'payment_method': 'payment'
}

ORDINAL_ENCODING = {
  'plan_type': {'Basic': 0, 'Standard': 1, 'Premium': 2, 'Enterprise': 3}
}

MODEL_VERSION = "1"

MODEL_FILE = f'models/churn_model_v{MODEL_VERSION}.pkl'