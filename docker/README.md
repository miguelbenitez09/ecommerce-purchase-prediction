# 🐳 Docker Configuration - Online Shoppers

> **Configuración de Docker para el despliegue del sistema de predicción de intención de compra online.**

---

## 👨‍💻 Autor

**Miguel Antonio Benítez González**
- 📧 Email: mbenitezg01@gmail.com
- 💻 GitHub: [https://github.com/miguelbenitez09](https://github.com/miguelbenitez09?tab=repositories)

---

This directory contains the Docker configuration for deploying the Online Shoppers Purchasing Intention prediction system.

## 📦 Services

### 1. API Service (`online-shoppers-api`)
- **Port**: 8004
- **Framework**: FastAPI
- **Purpose**: REST API for purchase intention predictions
- **Documentation**: http://localhost:8004/docs

### 2. Web Interface (`online-shoppers-web`)
- **Port**: 8503
- **Framework**: Streamlit
- **Purpose**: Interactive dashboard for predictions
- **URL**: http://localhost:8503

## 🚀 Quick Start

### Build and Run

```bash
# From the F_Docker directory
docker-compose up --build

# Or run in detached mode
docker-compose up -d --build
```

### Access Services

- **API**: http://localhost:8004
- **API Documentation**: http://localhost:8004/docs
- **Web Dashboard**: http://localhost:8503

### Stop Services

```bash
docker-compose down
```

## 📝 Configuration Details

### Port Mapping
- API: Host 8004 → Container 8000
- Web: Host 8503 → Container 8503

### Volumes
- `models/`: Read-only access to trained models
- `data/`: Read-only access to data (web interface only)
- `web/`: Source code for web interface

### Networks
- All services connected via `shoppers_network` bridge

### Health Checks
- API service includes health check endpoint
- Web service waits for API health before starting

## 🔧 Troubleshooting

### Port Already in Use

If you see a "port already allocated" error:

```bash
# Check what's using the port
netstat -ano | findstr :8004
netstat -ano | findstr :8503

# Stop other projects or change ports in docker-compose.yml
```

### Container Logs

```bash
# View logs for specific service
docker-compose logs online-shoppers-api
docker-compose logs online-shoppers-web

# Follow logs in real-time
docker-compose logs -f
```

### Rebuild After Changes

```bash
# Force rebuild without cache
docker-compose build --no-cache
docker-compose up
```

## 🌐 Integration with Portfolio

This project uses unique ports to avoid conflicts:
- Credit Card: 8002 (API), 8502 (Web)
- **Online Shoppers**: 8004 (API), 8503 (Web)
- Walmart: 8006 (API), 8506 (Web)

Each project can run simultaneously without port conflicts.

All projects can run simultaneously without port conflicts.
