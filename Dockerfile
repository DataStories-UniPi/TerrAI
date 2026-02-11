# Use an official Python runtime as a parent image
FROM python:3.10-slim-bullseye

RUN apt-get update && apt-get install -y \
    gdal-bin \
    libgdal-dev \
    build-essential \
    python3-dev \
    && apt-get clean

RUN python -m pip install --upgrade pip setuptools wheel

# Set the working directory to /terrai
WORKDIR /terrai

# Install any needed packages specified in requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir --trusted-host pypi.python.org -r requirements.txt
