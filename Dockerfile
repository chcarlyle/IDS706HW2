# Simple Dockerfile for running tests and scripts outside of VS Code Dev Containers
FROM python:3.12-slim

# Set work directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy all project files
COPY . .

# Default command: run all unit tests
CMD ["python", "-m", "unittest", "discover"]
