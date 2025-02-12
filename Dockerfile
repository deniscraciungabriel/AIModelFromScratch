# Use the official Python image from the Docker Hub
FROM python:3.9-slim

# Install gcc and other dependencies
RUN apt-get update && \
    apt-get install -y gcc build-essential && \
    apt-get clean

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Set the default command to run when starting the container
CMD ["python"]