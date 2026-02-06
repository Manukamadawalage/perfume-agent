@echo off
REM Suppress TensorFlow warnings for cleaner output
set TF_CPP_MIN_LOG_LEVEL=2
set TF_ENABLE_ONEDNN_OPTS=0

echo Starting Perfume Agent with clean output...
python -m uvicorn app:app --reload --port 8000
