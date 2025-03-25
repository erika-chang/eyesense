FROM python:3.10.6-buster

# Install the application dependencies
COPY requirements_api.txt ./
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements_api.txt

# Copy in the source code
COPY api ./api

# Use shell form of CMD to allow environment variable expansion
CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT
