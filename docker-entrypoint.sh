#!/bin/bash
python -m pipeline.trainer
exec uvicorn api.main:app --host 0.0.0.0 --port 8000
