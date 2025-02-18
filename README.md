# AIModelFromScratch

## Overview

This project demonstrates how to build an AI model from scratch using PyTorch. It includes training scripts, a chatbot implementation, and a Docker setup to ensure compatibility across different platforms.

## Features

- Transformer-based language model
- Training and evaluation scripts
- Chatbot implementation
- Docker support for easy deployment

## Prerequisites

- Docker
- Docker Compose

## Setup

1. Clone the repository:

   ```sh
   git clone https://github.com/yourusername/AIModelFromScratch.git
   cd AIModelFromScratch
   ```

2. Build and run the Docker container:
   ```sh
   docker-compose up --build
   ```

## GPTv1.py

This file contains a model around 7M parameters that uses character level tokenisation.
It should be trained on small files

## GPTv2.py

This file contains a model around 50M parameters that uses word level tokenisation
It should be trained on bigger files to prevent overfitting
