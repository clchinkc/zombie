# Start with a Python runtime as a parent image for the build stage
FROM python:3.9.13-slim AS build-env

# Set the working directory in the image
WORKDIR /build

# Copy only the requirements.txt at first to leverage Docker cache
COPY requirements.txt .

# Install any needed packages specified in requirements.txt without storing the cache
# Using --no-cache-dir with pip to avoid storing unnecessary files and keeping the image size small
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code to the build environment
COPY src/zombie_simulation/ ./zombie_simulation

# Start a new, final stage to keep the image size small but with necessary dependencies
FROM python:3.9.13-slim

# Set the working directory for the final image
WORKDIR /usr/src/app

# Install dependencies needed for tkinter in the final image
# Combining update, install, and cleanup in a single RUN to reduce the image size
# Using --no-install-recommends to only install essential packages and minimize image size
RUN apt-get update && apt-get install -y --no-install-recommends python3-tk \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy the Python environment and application code from the build stage
COPY --from=build-env /usr/local /usr/local
COPY --from=build-env /build/zombie_simulation ./zombie_simulation

# Add the directory containing main.py to the PYTHONPATH
ENV PYTHONPATH "${PYTHONPATH}:/usr/src/app/zombie_simulation"

# Define environment variable (optional)
ENV NAME ZombieSimulation

# For security, create a non-root user and switch to it
# This is a best practice to limit the privileges of the container process
RUN useradd -m appuser && chown -R appuser /usr/src/app
USER appuser

# Run main.py when the container launches
CMD ["python", "zombie_simulation/main.py"]

