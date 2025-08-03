# Use a minimal base image with Python
FROM python:3.12.3-slim

# Set working directory inside the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    gcc \
    g++ \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Copy application code
COPY app/ ./app/

# Install Python dependencies
RUN pip3 install --no-cache-dir -r app/requirements.txt

# Expose the port Streamlit will run on
EXPOSE 8080

# Add a healthcheck endpoint
HEALTHCHECK CMD curl --fail http://localhost:8080/_stcore/health || exit 1

# Launch the Streamlit app
ENTRYPOINT ["streamlit", "run", "app/streamlit_app.py", "--server.port=8080", "--server.address=0.0.0.0"]
