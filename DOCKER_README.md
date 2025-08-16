# webKinPred Docker Setup

This document explains how to run webKinPred using Docker containers.

## Prerequisites

- Docker (v20.10 or later)
- Docker Compose (v2.0 or later)
- At least 4GB of free RAM
- At least 2GB of free disk space

## Quick Start

### Development Environment

1. **Clone and navigate to the project:**
   ```bash
   cd /home/saleh/webKinPred
   ```

2. **Start the application:**
   ```bash
   ./start-docker.sh
   ```
   
   Or manually:
   ```bash
   docker-compose up --build
   ```

3. **Access the application:**
   - Frontend: http://localhost:5173
   - Backend API: http://localhost:8000
   - Admin Interface: http://localhost:8000/admin

### Production Environment

For production deployment:

```bash
docker-compose -f docker-compose.prod.yml up --build -d
```

Access the application at:
- Frontend: http://localhost:80
- Backend API: http://localhost:8000

## Services

The Docker setup includes:

- **backend**: Django application server
- **frontend**: React development server (or Nginx in production)
- **redis**: Redis server for caching and Celery
- **celery**: Background task worker
- **celery-beat**: Periodic task scheduler

## Useful Commands

### View logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f backend
docker-compose logs -f frontend
docker-compose logs -f celery
```

### Execute commands in containers
```bash
# Django management commands
docker-compose exec backend python manage.py migrate
docker-compose exec backend python manage.py createsuperuser
docker-compose exec backend python manage.py collectstatic

# Access Django shell
docker-compose exec backend python manage.py shell

# Access container shell
docker-compose exec backend bash
docker-compose exec frontend sh
```

### Database operations
```bash
# Run migrations
docker-compose exec backend python manage.py migrate

# Create superuser
docker-compose exec backend python manage.py createsuperuser

# Access database shell
docker-compose exec backend python manage.py dbshell
```

### Stop and cleanup
```bash
# Stop services
docker-compose down

# Stop and remove volumes
docker-compose down -v

# Remove all containers and images
docker-compose down --rmi all
```

## Development Workflow

1. **Make code changes** in your local files
2. **Files are automatically synced** to containers via volume mounts
3. **Services will auto-reload** when files change
4. **View logs** to debug issues: `docker-compose logs -f`

## Troubleshooting

### Common Issues

1. **Port conflicts**: If ports 8000, 5173, or 6379 are in use, modify the port mappings in docker-compose.yml

2. **Permission issues**: Ensure Docker has permission to access the project directory

3. **Services not starting**: Check logs with `docker-compose logs [service-name]`

4. **Database issues**: Reset database with:
   ```bash
   docker-compose down -v
   docker-compose up --build
   ```

### Health Checks

The services include health checks. View status with:
```bash
docker-compose ps
```

### Performance

For better performance:
- Increase Docker's memory allocation to at least 4GB
- Use Docker's native filesystem on macOS/Windows
- Consider using bind mounts instead of volumes for development

## File Structure

```
webKinPred/
├── Dockerfile                 # Backend container
├── docker-compose.yml        # Development setup
├── docker-compose.prod.yml   # Production setup
├── start-docker.sh          # Quick start script
├── .dockerignore            # Docker ignore file
├── frontend/
│   ├── Dockerfile           # Frontend development container
│   ├── Dockerfile.prod      # Frontend production container
│   ├── nginx.conf          # Nginx configuration
│   └── .dockerignore       # Frontend Docker ignore
└── webKinPred/
    └── settings_docker.py   # Docker-specific Django settings
```

## Environment Variables

Key environment variables:

- `DEBUG`: Enable/disable Django debug mode
- `REDIS_HOST`: Redis server hostname
- `REDIS_PORT`: Redis server port
- `DJANGO_SETTINGS_MODULE`: Django settings module to use

## Security Notes

- The development setup uses volume mounts for code hot-reloading
- Production setup uses optimized images and security headers
- Redis is not password-protected in development
- SQLite database files are mounted as volumes

## Support

For issues related to:
- Docker setup: Check this README and Docker logs
- Application bugs: Refer to the main project documentation
- Performance: Monitor container resource usage with `docker stats`
