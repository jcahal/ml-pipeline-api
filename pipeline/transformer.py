import pandas as pd

import config

class DataTransformer:
  def __init__(self, df, drop_na, normalize):
    self.df = df
    self.drop_na = drop_na
    self.normalize = normalize

  def _drop_nulls(self):
    # Drop any rows with missing values
    try:
      self.df = self.df.dropna().copy() # explicit .copy() to quell warning
    
    except Exception as e:
      raise ValueError(f'Failed to drop nulls: {e}')
    
    return self

  def _normalize_numeric_columns(self):
    # Min-max scale numeric features to [0, 1]
    try:
      for col in config.NUMERICAL_COLS:
        self.df[col] = round((self.df[col] - self.df[col].min()) / (self.df[col].max() - self.df[col].min()), 3)
    
    except KeyError as e:
      raise ValueError(f'Expected numeric column not found: {e}')
    except Exception as e:
      raise ValueError(f'Failed to normalize columns: {e}')
    
    return self

  def _encode_categoricals(self):
    # One-hot encode nominal columns
    try:
      one_hot_encoding = config.ONE_HOT_ENCODING
      try:
        self.df = pd.get_dummies(self.df, columns=one_hot_encoding.keys(), prefix=one_hot_encoding.values())
    
      except Exception as e:
        raise ValueError(f'Failed to one-hot encoding: {e}')

      # Ordinal encode — order defined in config.ORDINAL_ENCODING
      ordinal_encoding = config.ORDINAL_ENCODING

      for col in ordinal_encoding.keys():
        self.df[col] = self.df[col].map(ordinal_encoding[col])
    
    except KeyError as e:
      raise ValueError(f'Expected categorical column not found: {e}')
    
    except Exception as e:
      raise ValueError(f'Failed to ordinal encoding: {e}')

    return self
  
  def transform(self):
    # Drop nulls if flagged
    if self.drop_na:
      self = self._drop_nulls()
    
    # Normalize if flagged
    if self.normalize:
      self = self._normalize_numeric_columns()._encode_categoricals()
    
    return self.df
