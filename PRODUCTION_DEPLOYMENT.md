# Production Deployment Guide for WebKinPred

## 1. Server Prerequisites

### Install Docker and Docker Compose
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Install Nginx (as reverse proxy)
sudo apt install nginx -y

# Install SSL certificate tool (recommended)
sudo apt install certbot python3-certbot-nginx -y

# Logout and login again for Docker permissions to take effect
```

### 2. Clone and Setup Project

```bash
# Clone the repository
git clone https://github.com/saleha1wer/webKinPred.git
cd webKinPred

# Create production environment directories
sudo mkdir -p /var/webkinpred/{media,staticfiles,logs,backups}
sudo chown -R $USER:$USER /var/webkinpred

# Create production data directories
mkdir -p ./media/jobs ./media/experimental ./media/PCA ./media/pseq2sites ./media/sequence_info
mkdir -p ./staticfiles
```

### 3. Configure Environment Variables

Create a production environment file:
```bash
# Create .env.prod file
cat > .env.prod << 'EOF'
# Django Settings
DEBUG=0
SECRET_KEY=your-super-secret-production-key-here
ALLOWED_HOSTS=your-domain.com,www.your-domain.com,localhost,127.0.0.1

# Database (if using PostgreSQL in future)
DB_ENGINE=django.db.backends.sqlite3
DB_NAME=/app/db.sqlite3

# Redis
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_URL=redis://redis:6379/0

# Email (configure for production notifications)
EMAIL_HOST=smtp.gmail.com
EMAIL_PORT=587
EMAIL_USE_TLS=1
EMAIL_HOST_USER=your-email@gmail.com
EMAIL_HOST_PASSWORD=your-app-password

# Security
SECURE_SSL_REDIRECT=1
SECURE_BROWSER_XSS_FILTER=1
SECURE_CONTENT_TYPE_NOSNIFF=1
X_FRAME_OPTIONS=DENY
EOF

# Secure the environment file
chmod 600 .env.prod
```

### 4. Update Frontend Nginx Configuration

Update the frontend nginx configuration for production:
```bash
cat > ./frontend/nginx.conf << 'EOF'
server {
    listen 80;
    server_name localhost;
    root /usr/share/nginx/html;
    index index.html;

    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_types text/plain text/css text/xml text/javascript application/x-javascript application/xml+rss application/javascript application/json;

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header Referrer-Policy "no-referrer-when-downgrade" always;

    # Handle React Router
    location / {
        try_files $uri $uri/ /index.html;
        expires 1h;
        add_header Cache-Control "public, immutable";
    }

    # Static assets caching
    location /static/ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }

    # API proxy to backend
    location /api/ {
        proxy_pass http://backend:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_connect_timeout 300s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
        client_max_body_size 100M;
    }

    # Media files proxy to backend
    location /media/ {
        proxy_pass http://backend:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # Admin panel proxy to backend
    location /admin/ {
        proxy_pass http://backend:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
EOF
```

### 5. Create Production Docker Compose Override

Create an enhanced production configuration:
```bash
cat > docker-compose.override.yml << 'EOF'
version: '3.8'

services:
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./nginx/sites-available:/etc/nginx/sites-available
      - ./nginx/sites-enabled:/etc/nginx/sites-enabled
      - ./nginx/ssl:/etc/nginx/ssl
      - /var/log/nginx:/var/log/nginx
    depends_on:
      - frontend
      - backend
    restart: unless-stopped

  redis:
    volumes:
      - /var/webkinpred/redis_data:/data
    command: redis-server --appendonly yes --maxmemory 512mb --maxmemory-policy allkeys-lru

  backend:
    env_file:
      - .env.prod
    volumes:
      - /var/webkinpred/media:/app/media
      - /var/webkinpred/staticfiles:/app/staticfiles
      - /var/webkinpred/logs:/app/logs
      - ./db.sqlite3:/app/db.sqlite3
    environment:
      - DJANGO_SETTINGS_MODULE=webKinPred.settings_docker
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G

  frontend:
    environment:
      - NODE_ENV=production
      - VITE_API_BASE_URL=/api

  celery:
    env_file:
      - .env.prod
    volumes:
      - /var/webkinpred/media:/app/media
      - /var/webkinpred/logs:/app/logs
      - ./db.sqlite3:/app/db.sqlite3
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G

  celery-beat:
    env_file:
      - .env.prod
    volumes:
      - /var/webkinpred/media:/app/media
      - /var/webkinpred/logs:/app/logs
      - ./db.sqlite3:/app/db.sqlite3

volumes:
  redis_data:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /var/webkinpred/redis_data
EOF
```

### 6. Main Nginx Configuration (Host Level)

Create the main Nginx configuration on your server:
```bash
# Create nginx directories
sudo mkdir -p /etc/nginx/sites-available /etc/nginx/sites-enabled

# Create main site configuration
sudo tee /etc/nginx/sites-available/webkinpred << 'EOF'
upstream backend {
    server 127.0.0.1:8000;
}

upstream frontend {
    server 127.0.0.1:3000;  # This will be the Docker frontend service
}

# Redirect HTTP to HTTPS
server {
    listen 80;
    server_name your-domain.com www.your-domain.com;
    return 301 https://$server_name$request_uri;
}

# Main HTTPS server
server {
    listen 443 ssl http2;
    server_name your-domain.com www.your-domain.com;

    # SSL Configuration (will be configured by Certbot)
    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;
    ssl_session_timeout 1d;
    ssl_session_cache shared:MozTLS:10m;
    ssl_session_tickets off;
    
    # Modern SSL configuration
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;

    # Security headers
    add_header Strict-Transport-Security "max-age=63072000" always;
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;

    # File upload size
    client_max_body_size 100M;

    # Logging
    access_log /var/log/nginx/webkinpred_access.log;
    error_log /var/log/nginx/webkinpred_error.log;

    # Main frontend (React app)
    location / {
        proxy_pass http://127.0.0.1:3000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support for development hot reload
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }

    # API routes
    location /api/ {
        proxy_pass http://backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_connect_timeout 300s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
    }

    # Django admin
    location /admin/ {
        proxy_pass http://backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # Static files
    location /static/ {
        proxy_pass http://backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }

    # Media files
    location /media/ {
        proxy_pass http://backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # Health check endpoint
    location /health/ {
        proxy_pass http://backend;
        access_log off;
    }
}
EOF

# Enable the site
sudo ln -sf /etc/nginx/sites-available/webkinpred /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default

# Test nginx configuration
sudo nginx -t
```

### 7. SSL Certificate Setup

```bash
# Replace 'your-domain.com' with your actual domain
sudo certbot --nginx -d your-domain.com -d www.your-domain.com

# Test automatic renewal
sudo certbot renew --dry-run
```

### 8. Create Deployment Scripts

Create a deployment script for easy updates:
```bash
cat > deploy.sh << 'EOF'
#!/bin/bash
set -e

echo "🚀 Starting deployment..."

# Pull latest changes
echo "📥 Pulling latest changes..."
git pull origin main

# Build and start services
echo "🏗️  Building and starting services..."
docker-compose -f docker-compose.prod.yml -f docker-compose.override.yml down
docker-compose -f docker-compose.prod.yml -f docker-compose.override.yml build --no-cache
docker-compose -f docker-compose.prod.yml -f docker-compose.override.yml up -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 30

# Run migrations
echo "🗃️  Running database migrations..."
docker-compose -f docker-compose.prod.yml exec backend python manage.py migrate
docker-compose -f docker-compose.prod.yml exec backend python manage.py migrate --database=seqmap

# Collect static files
echo "📦 Collecting static files..."
docker-compose -f docker-compose.prod.yml exec backend python manage.py collectstatic --noinput

# Restart nginx
echo "🔄 Restarting Nginx..."
sudo systemctl reload nginx

# Health check
echo "🏥 Performing health check..."
sleep 10
if curl -f -s http://localhost:8000/api/health/ > /dev/null; then
    echo "✅ Backend is healthy"
else
    echo "❌ Backend health check failed"
    exit 1
fi

echo "✅ Deployment completed successfully!"
EOF

chmod +x deploy.sh
```

### 9. Create Backup Script

```bash
cat > backup.sh << 'EOF'
#!/bin/bash
set -e

BACKUP_DIR="/var/webkinpred/backups"
DATE=$(date +%Y%m%d_%H%M%S)

echo "📦 Creating backup..."

# Create backup directory
mkdir -p $BACKUP_DIR

# Backup database
echo "🗃️  Backing up database..."
cp db.sqlite3 "$BACKUP_DIR/db_$DATE.sqlite3"

# Backup media files
echo "📁 Backing up media files..."
tar -czf "$BACKUP_DIR/media_$DATE.tar.gz" -C /var/webkinpred media/

# Backup configuration
echo "⚙️  Backing up configuration..."
tar -czf "$BACKUP_DIR/config_$DATE.tar.gz" .env.prod docker-compose.prod.yml docker-compose.override.yml

# Keep only last 7 days of backups
echo "🧹 Cleaning old backups..."
find $BACKUP_DIR -name "*.sqlite3" -mtime +7 -delete
find $BACKUP_DIR -name "*.tar.gz" -mtime +7 -delete

echo "✅ Backup completed: $BACKUP_DIR"
EOF

chmod +x backup.sh
```

### 10. Monitoring and Logging Setup

```bash
cat > docker-compose.monitoring.yml << 'EOF'
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3001:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
    restart: unless-stopped

  redis-exporter:
    image: oliver006/redis_exporter:latest
    environment:
      - REDIS_ADDR=redis:6379
    restart: unless-stopped

volumes:
  grafana_data:
EOF
```

### 11. Deployment Commands

Execute these commands to deploy:

```bash
# 1. Initial setup
./backup.sh  # Create initial backup if migrating

# 2. Start the application
docker-compose -f docker-compose.prod.yml -f docker-compose.override.yml up -d

# 3. Setup database (first time only)
docker-compose -f docker-compose.prod.yml exec backend python manage.py migrate
docker-compose -f docker-compose.prod.yml exec backend python manage.py migrate --database=seqmap
docker-compose -f docker-compose.prod.yml exec backend python manage.py collectstatic --noinput
docker-compose -f docker-compose.prod.yml exec backend python manage.py createsuperuser

# 4. Start Nginx
sudo systemctl enable nginx
sudo systemctl start nginx

# 5. Check status
docker-compose -f docker-compose.prod.yml ps
sudo systemctl status nginx
```

### 12. Maintenance Commands

```bash
# View logs
docker-compose -f docker-compose.prod.yml logs -f backend
docker-compose -f docker-compose.prod.yml logs -f celery
docker-compose -f docker-compose.prod.yml logs -f frontend

# Update application
./deploy.sh

# Backup data
./backup.sh

# Scale celery workers
docker-compose -f docker-compose.prod.yml up -d --scale celery=3

# Restart specific service
docker-compose -f docker-compose.prod.yml restart backend

# Check health
curl -f http://localhost:8000/api/health/
```

### 13. Firewall Configuration

```bash
# Enable UFW firewall
sudo ufw enable

# Allow SSH
sudo ufw allow ssh

# Allow HTTP/HTTPS
sudo ufw allow 80
sudo ufw allow 443

# Check status
sudo ufw status
```

## Security Checklist

- [ ] SSL certificates configured
- [ ] Strong SECRET_KEY in production
- [ ] Database backups scheduled
- [ ] Firewall configured
- [ ] Nginx security headers enabled
- [ ] File upload size limits set
- [ ] Environment variables secured
- [ ] Log rotation configured
- [ ] Monitoring setup
- [ ] Health checks working

## Monitoring URLs

- Main Application: https://your-domain.com
- Django Admin: https://your-domain.com/admin/
- API Health: https://your-domain.com/api/health/
- Grafana (optional): https://your-domain.com:3001

Remember to replace 'your-domain.com' with your actual domain name throughout all configuration files!
EOF
