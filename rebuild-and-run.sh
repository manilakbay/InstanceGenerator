#!/bin/bash

# Configuration
API_URL="${1:-http://localhost:5000}"
CONTAINER_NAME="evrp-generator"
PORT="5000"

echo "🛑 Stopping and removing old container..."
docker rm -f $CONTAINER_NAME 2>/dev/null

echo "🔨 Building Docker image with REACT_APP_API_URL=$API_URL..."
docker build --build-arg REACT_APP_API_URL=$API_URL -t $CONTAINER_NAME .

if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
    echo "🚀 Starting container on port $PORT..."
    docker run -d \
      -p $PORT:5000 \
      -v $(pwd)/backend/created_datasets:/app/created_datasets \
      -v $(pwd)/backend/cache:/app/cache \
      --name $CONTAINER_NAME \
      $CONTAINER_NAME
    
    if [ $? -eq 0 ]; then
        echo "✅ Container started!"
        echo "📊 Container status:"
        docker ps | grep $CONTAINER_NAME
        echo ""
        echo "🌐 Access at: http://localhost:$PORT"
        echo "📋 View logs: docker logs -f $CONTAINER_NAME"
    else
        echo "❌ Failed to start container"
    fi
else
    echo "❌ Build failed"
fi


# How to run: ./rebuild-and-run.sh http://localhost:5000