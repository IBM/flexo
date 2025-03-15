# Deployment Platforms Overview

Flexo can be deployed to various cloud platforms and container orchestration services. This guide helps with a few options but it's up to you on determining the right platform for your needs.

---

## Platforms

### Links to docs

- [Deploy to Code Engine](code-engine.md)
- [Deploy to AWS Fargate](fargate.md)
- [Deploy to OpenShift](openshift.md)
- [Deploy to Kubernetes](kubernetes.md)

## Platform Comparison


| Feature | Code Engine | AWS Fargate | OpenShift | Kubernetes |
|---------|------------|-------------|-----------|------------|
| Setup Complexity | Low | Medium | High | Medium |
| Scaling | Automatic | Automatic | Manual/Auto | Manual/Auto |
| Cost Model | Pay-per-use | Pay-per-use | Cluster-based | Cluster-based |
| Management | Managed | Managed | Self/Managed | Self-managed |
| Config Management | Built-in | Parameter Store | ConfigMaps | ConfigMaps |

---

## Deployment Considerations

### Resource Requirements

- Recommended Minimum Memory: 6GB
- Recommended Minimum CPU: 2 vCPU
- Storage: Based on usage

Note: Number of uvicorn workers can be set using `UVICORN_WORKERS`. The default is calculated dynamically and set as `2 × num_vCPUs + 1`.

### Configuration Management
Each platform provides different methods for:

- Environment variables
- Secret management
- Volume mounts
- Network policies

### Monitoring and Logging
Consider platform-specific solutions for:

- Health monitoring
- Log aggregation
- Performance metrics
- Alert management

----