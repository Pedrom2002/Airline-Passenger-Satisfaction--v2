# Multi-stage build for optimized production image
FROM python:3.9-slim as builder

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --user --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH=/home/app/.local/bin:$PATH

# Create non-root user
RUN useradd -m -u 1000 app && \
    mkdir -p /home/app/models /home/app/logs /home/app/monitoring

# Set working directory
WORKDIR /home/app

# Copy Python dependencies from builder
COPY --from=builder /root/.local /home/app/.local

# Copy application files
COPY --chown=app:app . .

# Create necessary directories
RUN mkdir -p models logs monitoring results data

# Switch to non-root user
USER app

# Expose ports
EXPOSE 8000 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command (can be overridden)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

# Alternative commands:
# For Streamlit: CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
# For Flask: CMD ["python", "-m", "flask", "run", "--host=0.0.0.0", "--port=5000"]
