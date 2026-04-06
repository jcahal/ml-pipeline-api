INPUT_FILE = 'data/sample.csv'

COLUMNS = ['customer_id', 'age', 'tenure_months', 'monthly_spend', 'num_support_tickets', 'region', 'plan_type', 'payment_method', 'churned']

# Continuous numeric columns
NUMERICAL_COLS = ['age', 'tenure_months', 'monthly_spend', 'num_support_tickets']

# Nominal categoricals
ONE_HOT_ENCODING = {
  'region': 'region', 
  'payment_method': 'payment'
}

# Ordinal categoricals
ORDINAL_ENCODING = {
  'plan_type': {'Basic': 0, 'Standard': 1, 'Premium': 2, 'Enterprise': 3}
}
