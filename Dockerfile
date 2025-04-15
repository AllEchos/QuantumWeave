# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY *.py .
COPY LICENSE .
COPY README.md .

# Create directories for persistence
RUN mkdir -p /app/data
RUN mkdir -p /app/models

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=2

# Make port 8000 available if you need a web interface
# EXPOSE 8000

# Define environment variable for configuration
ENV CONFIG_FILE=config.ini

# Create a volume for persistent data
VOLUME ["/app/data"]

# Run when the container launches
# Modify this to run your specific script (e.g., live_trader.py)
CMD ["python", "live_trader.py"]

