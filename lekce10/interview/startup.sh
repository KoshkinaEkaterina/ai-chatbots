#!/bin/bash
cd interview
gunicorn -w 4 -k uvicorn.workers.UvicornWorker fastapi_app:app --bind=0.0.0.0:8000 --timeout 120
