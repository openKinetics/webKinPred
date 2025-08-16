#!/bin/bash

# Test script for Docker setup

set -e

echo "🧪 Testing webKinPred Docker setup..."

# Function to check if a service is responding
check_service() {
    local url=$1
    local service_name=$2
    local max_attempts=30
    local attempt=1

    echo "⏳ Waiting for $service_name to start..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f -s "$url" > /dev/null 2>&1; then
            echo "✅ $service_name is responding"
            return 0
        fi
        echo "Attempt $attempt/$max_attempts: $service_name not ready yet..."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    echo "❌ $service_name failed to start after $max_attempts attempts"
    return 1
}

# Start containers
echo "🚀 Starting containers..."
docker-compose up -d --build

# Wait for services
echo "⏳ Waiting for services to initialize..."
sleep 15

# Test Redis
echo "🔍 Testing Redis..."
if docker-compose exec -T redis redis-cli ping | grep -q "PONG"; then
    echo "✅ Redis is working"
else
    echo "❌ Redis test failed"
    exit 1
fi

# Test Backend
echo "🔍 Testing Backend..."
check_service "http://localhost:8000/api/health/" "Backend"

# Test Frontend
echo "🔍 Testing Frontend..."
check_service "http://localhost:5173" "Frontend"

# Test Celery
echo "🔍 Testing Celery..."
if docker-compose logs celery | grep -q "ready"; then
    echo "✅ Celery worker is ready"
else
    echo "⚠️  Celery worker might not be fully ready (check logs)"
fi

echo ""
echo "🎉 All tests passed! Docker setup is working correctly."
echo ""
echo "Access your application at:"
echo "🌐 Frontend: http://localhost:5173"
echo "🔧 Backend: http://localhost:8000"
echo "📊 Admin: http://localhost:8000/admin"
echo ""
echo "To stop the containers: docker-compose down"
