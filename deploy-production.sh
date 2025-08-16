#!/bin/bash
set -e

echo "🚀 WebKinPred Production Deployment Script"
echo "========================================="

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   echo "❌ This script should not be run as root"
   exit 1
fi

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Create necessary directories
echo "📁 Creating necessary directories..."
sudo mkdir -p /var/webkinpred/{media,staticfiles,logs,backups,redis_data}
sudo chown -R $USER:$USER /var/webkinpred

# Create production directories in project
mkdir -p ./media/jobs ./media/experimental ./media/PCA ./media/pseq2sites ./media/sequence_info
mkdir -p ./staticfiles

# Stop existing containers if running
echo "🛑 Stopping existing containers..."
docker-compose down 2>/dev/null || true

# Build and start production services
echo "🏗️  Building and starting production services..."
docker-compose -f docker-compose.prod.yml up -d --build

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 30

# Check if backend is running
echo "🏥 Checking backend health..."
max_attempts=30
attempt=0

while [ $attempt -lt $max_attempts ]; do
    if curl -f -s http://localhost:8000/api/health/ > /dev/null 2>&1; then
        echo "✅ Backend is healthy"
        break
    fi
    
    echo "⏳ Waiting for backend to start... ($((attempt+1))/$max_attempts)"
    sleep 10
    attempt=$((attempt+1))
done

if [ $attempt -eq $max_attempts ]; then
    echo "❌ Backend failed to start properly"
    docker-compose -f docker-compose.prod.yml logs backend
    exit 1
fi

# Run migrations
echo "🗃️  Running database migrations..."
docker-compose -f docker-compose.prod.yml exec -T backend python manage.py migrate
docker-compose -f docker-compose.prod.yml exec -T backend python manage.py migrate --database=seqmap

# Collect static files
echo "📦 Collecting static files..."
docker-compose -f docker-compose.prod.yml exec -T backend python manage.py collectstatic --noinput

echo ""
echo "✅ Deployment completed successfully!"
echo ""
echo "📋 Next Steps:"
echo "1. Configure your domain DNS to point to this server"
echo "2. Setup SSL certificate: sudo certbot --nginx -d yourdomain.com"
echo "3. Create Django superuser: docker-compose -f docker-compose.prod.yml exec backend python manage.py createsuperuser"
echo "4. Setup Nginx reverse proxy following the guide in PRODUCTION_DEPLOYMENT.md"
echo ""
echo "🌐 Current Access:"
echo "   Frontend: http://localhost (port 80)"
echo "   Backend API: http://localhost:8000"
echo "   Backend Admin: http://localhost:8000/admin/"
echo ""
echo "📊 Service Status:"
docker-compose -f docker-compose.prod.yml ps
