# Build stage
FROM python:3.12-slim-bookworm AS builder

# Install Poetry via pip (this is simpler and more reliable)
RUN pip install poetry==1.8.4

# Set environment variables for poetry (optional, for customizing poetry behavior)
ENV POETRY_VIRTUALENVS_CREATE=true \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_CACHE_DIR=/root/.cache/pypoetry \
    PATH="/root/.local/bin:$PATH"  

# Set the working directory
WORKDIR /app

# Copy pyproject.toml and poetry.lock to the container
COPY pyproject.toml poetry.lock* /app/

# Check if Poetry is installed and accessible
RUN echo "Checking Poetry version..." && poetry --version  

# Install dependencies (poetry install handles all the dependencies)
RUN poetry install --no-dev --no-interaction --no-root 

# Copy the rest of the application
ADD . /app

# Final stage
FROM python:3.12-slim-bookworm

# Copy the application from the builder stage
COPY --from=builder --chown=app:app /app /app

# Place executables in the environment at the front of the path (to use the virtual environment's Python)
ENV PATH="/app/.venv/bin:$PATH"

# Set the working directory
WORKDIR /app
