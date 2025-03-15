# Deploying Flexo to IBM Code Engine

This guide walks through deploying Flexo to IBM Code Engine, a fully managed serverless platform.

## Prerequisites

- IBM Cloud account with Code Engine permissions
- IBM Cloud CLI with Code Engine plugin
- Container image in a registry (ICR or Docker Hub)

## Deployment Steps

### 1. Create Project
```bash
# Login to IBM Cloud
ibmcloud login --sso

# Create new project
ibmcloud ce project create --name flexo-project
# Or select existing
ibmcloud ce project select --name flexo-project
```

### 2. Create Environment Variables
Create two environment files to manage your configuration:

**secrets.env** (for sensitive values):
```env
# For watsonx.ai client
WXAI_APIKEY=your-api-key

# For example weather tool
OWM_API_KEY=owm-api-key

# For example RAG tool (using elastic)
ES_ES_API_KEY=elastic-api-key

# API authentication
FLEXO_API_KEY=flexo-api-key
```

**config.env** (for non-sensitive configuration):
```env
# For watsonx.ai client
WXAI_URL=https://us-south.ml.cloud.ibm.com
WXAI_PROJECT_ID=your-project-id

# For example RAG tool
ES_INDEX_NAME=my_index
ES_ES_ENDPOINT=elastic-endpoint

# API Configurations
ENABLE_API_KEY=true
```

Create the configurations in Code Engine:
```bash
# Create secret from file
ibmcloud ce secret create --name flexo-secrets \
    --from-env-file secrets.env

# Create configmap from file
ibmcloud ce configmap create --name flexo-config \
    --from-env-file config.env
```

### 3. Deploy Application
```bash
# Basic deployment
ibmcloud ce application create \
    --name flexo \
    --image us.icr.io/namespace/flexo:latest \
    --port 8000 \
    --cpu 2 \
    --memory 8G \
    --env-from-secret flexo-secrets \
    --env-from-configmap flexo-config
```

### Optional: Mount Configuration
If you want to mount your agent configuration instead of building it into the image:

```bash
# Create configmap from agent.yaml
ibmcloud ce configmap create --name agent-config \
    --from-file agent.yaml

# Deploy with mount
ibmcloud ce application create \
    --name flexo \
    --image us.icr.io/namespace/flexo:latest \
    --env-from-secret flexo-secrets \
    --env-from-configmap flexo-config \
    --mount-configmap /app/config=agent-config
```

## Configuration Options

### Scaling
```bash
ibmcloud ce application update \
    --name flexo \
    --min-scale 1 \
    --max-scale 5
```

### Resource Allocation
```bash
ibmcloud ce application update \
    --name flexo \
    --cpu 2 \
    --memory 8G
```

## Monitoring
```bash
# View logs
ibmcloud ce application logs --name flexo

# Check status
ibmcloud ce application get --name flexo
```

### Environment Variable Access

- Values from both secrets and configmaps are loaded as environment variables
- Access in your code using `os.getenv()`:
  ```python
  import os
  
  api_key = os.getenv('WXAI_APIKEY')
  log_level = os.getenv('LOG_LEVEL')
  ```
- Mounted files (like agent.yaml) are available as files at the specified path

----