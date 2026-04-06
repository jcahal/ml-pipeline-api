import pandas as pd

import config

class DataLoader:
  def __init__(self, path):
    self.path = path

  def load(self):
    path = self.path

    # Value verification
    try:
      # File extension must be CSV
      file_ext = path.split('.')[1]
      if file_ext != 'csv':
        raise ValueError(f'Incorrect file type. File must be CSV format')
      
      # File columns must match given columns
      with open(path) as file:
        file_columns = file.readline().strip().split(',')

        columns = config.COLUMNS
        if file_columns != columns:
          raise ValueError(f'Incorrect CSV header. CSV header must be the following columns:\n"{columns}"')

    except FileNotFoundError as e:
      raise ValueError(f'File not found: {e}')

    except IOError as e:
      raise ValueError(f'Failed to read file: {e}')

    # Return a DataFrame
    return pd.read_csv(path)