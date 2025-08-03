# Use a minimal base image with Python
FROM python:3.9-slim

# Set working directory inside the container
WORKDIR /app

# Install required system dependencies for Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    libffi-dev \
    libssl-dev \
    libpq-dev \
    liblzma-dev \
    zlib1g-dev \
    libfreetype6-dev \
    pkg-config \
    gfortran \
    libopenblas-dev \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/

# Copy required data file
COPY data/processed/index_tier2.parquet ./data/processed/

# Expose the port Streamlit will run on
EXPOSE 8080

# Add a healthcheck endpoint
HEALTHCHECK CMD curl --fail http://localhost:8080/_stcore/health || exit 1

# Launch the Streamlit app
ENTRYPOINT ["streamlit", "run", "app/streamlit_app.py", "--server.port=8080", "--server.address=0.0.0.0"]
