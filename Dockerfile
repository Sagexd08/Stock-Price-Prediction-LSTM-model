FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Create necessary directories
RUN mkdir -p data/raw data/processed data/portfolios models results logs

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLE_CORS=true
ENV APP_FILE=app.py

# Expose the port Streamlit will run on
EXPOSE 8501

# Command to run the application
CMD streamlit run ${APP_FILE} --server.port=8501 --server.address=0.0.0.0 --server.enableCORS=true --server.enableXsrfProtection=true
