# IBM Container Registry Guide

This guide details how to use IBM Container Registry (ICR) for storing and managing your Flexo container images.

## Prerequisites

- IBM Cloud account
- IBM Cloud CLI installed
- Container Registry plugin installed (`ibmcloud plugin install container-registry`)
- Docker installed locally

## Setup Process

### 1. Install and Configure IBM Cloud CLI
```bash
# Login to IBM Cloud
ibmcloud login --sso

# Target a resource group
ibmcloud target -g <resource-group>

# Login to Container Registry
ibmcloud cr login
```

### 2. Create a Namespace
```bash
# Create a namespace for your images
ibmcloud cr namespace-add <your-namespace>
```

### 3. Configure Access
```bash
# Generate API key for automation
ibmcloud iam api-key-create flexo-key -d "Key for Flexo deployments"

# Optional: Create service ID
ibmcloud iam service-id-create flexo-service --description "Service ID for Flexo"
```

## Pushing Images

### Tag Your Image
```bash
docker tag flexo:latest us.icr.io/<your-namespace>/flexo:latest
```

### Push to Registry
```bash
docker push us.icr.io/<your-namespace>/flexo:latest
```

## Best Practices

- Use meaningful image tags (e.g., version numbers, git hashes)
- Enable vulnerability scanning
- Regularly clean up unused images
- Set up retention policies

## Troubleshooting

- **Login Issues**: Check IBM Cloud CLI version and authentication
- **Push Failures**: Verify namespace permissions
- **Pull Issues**: Check network connectivity and credentials

## Next Steps

- [Deploy to Code Engine](../platforms/code-engine.md)
- [Configure Environment Variables](../../configuration/environment.md)
