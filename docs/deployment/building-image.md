# Building the Flexo Image

This guide walks through the process of building a Docker image for Flexo.

## Prerequisites

- Docker installed (20.10.0 or higher)
- Git clone of Flexo repository
- (Optional) Podman 3.0+ as Docker alternative

## Building the Image

### Basic Build
```bash
# From the repository root
docker build -t flexo:latest .
```

### Platform-Specific Build
For cloud deployments, build for AMD64 architecture:
```bash
docker buildx build --platform linux/amd64 \
    -t flexo:latest \
    --load .
```

### Build Arguments
Customize the build with arguments:
```bash
docker build \
    --build-arg PYTHON_VERSION=3.9 \
    -t flexo:latest .
```

## Build Options

### 1. Build with Configuration
Include agent configuration during build:
```bash
# Copy configuration files
cp agent.yaml build/
cp -r tools/ build/

# Build image
docker build -t flexo:configured .
```

### 2. Build Base Image
Build without configuration for runtime mounting:
```bash
docker build -t flexo:base -f Dockerfile.base .
```

## Testing the Build
Verify your build:
```bash
# Run container
docker run -it --rm \
    -p 8000:8000 \
    --env-file .env \
    flexo:latest

# Test endpoint
curl http://localhost:8000/health
```

## Troubleshooting

Common issues and solutions:

- **Build fails**: Check Docker daemon status and Dockerfile syntax
- **Platform errors**: Verify buildx setup for your target platform
- **Size issues**: Use .dockerignore to exclude unnecessary files

## Next Steps
- [Push to Registry](registries/overview.md)
- [Configure Environment](../agent-configuration.md)
