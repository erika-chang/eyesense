FROM python:3.10.6-buster

# Install the application dependencies
COPY requirements_api.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy in the source code
COPY api ./api

CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT