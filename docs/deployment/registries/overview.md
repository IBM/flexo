# Container Registry Guide

After building your Flexo image, you'll need to push it to a container registry. This guide covers the main registry options and their setup.

----

# Example Registries:

## IBM Container Registry
Best for IBM Cloud deployments:

- Native integration with IBM Cloud
- Built-in vulnerability scanning
- Regional endpoints for faster pulls

[Detailed IBM Registry Guide](ibm-registry.md)

## Docker Hub
Universal option for all platforms:

- Widely supported
- Free public repositories
- Simple authentication

[Detailed Docker Hub Guide](dockerhub.md)

----

# General Process

1. **Tag your image**:
   ```bash
   # Format: registry/namespace/repository:tag
   docker tag flexo:latest <registry-url>/<namespace>/flexo:latest
   ```

2. **Authenticate**:
   ```bash
   # For IBM Container Registry
   ibmcloud cr login

   # For Docker Hub
   docker login
   ```

3. **Push image**:
   ```bash
   docker push <registry-url>/<namespace>/flexo:latest
   ```

---

## Security Best Practices

- Use unique credentials for automation
- Regularly rotate registry credentials
- Scan images for vulnerabilities
- Use specific tags instead of 'latest'

---

## Next Steps

- [IBM Registry Setup](ibm-registry.md)
- [Docker Hub Setup](dockerhub.md)
- [Platform Deployment](../platforms/overview.md)

----