# Deploying Flexo to OpenShift

This guide covers deploying Flexo on OpenShift, providing enterprise-grade orchestration and scaling.

## Prerequisites

- Access to OpenShift cluster
- OpenShift CLI (oc) installed
- Container image in accessible registry
- Cluster admin permissions (for some operations)

## Deployment Steps

### 1. Create Project
```bash
# Create new project
oc new-project flexo

# Or use existing
oc project flexo
```

### 2. Create Secrets and ConfigMaps
```yaml
# secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: flexo-secrets
type: Opaque
stringData:
  WATSON_API_KEY: <your-key>
  OTHER_SECRET: <value>
```

```bash
# Apply configuration
oc apply -f secrets.yaml
oc create configmap flexo-config \
    --from-literal=LOGGING_LEVEL=INFO
```

### 3. Deploy Application
```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: flexo
spec:
  replicas: 1
  template:
    spec:
      containers:
      - name: flexo
        image: us.icr.io/namespace/flexo:latest
        ports:
        - containerPort: 8000
        envFrom:
        - secretRef:
            name: flexo-secrets
        - configMapRef:
            name: flexo-config
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
```

### 4. Create Service and Route
```yaml
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: flexo
spec:
  ports:
  - port: 8000
  selector:
    app: flexo

---
# route.yaml
apiVersion: route.openshift.io/v1
kind: Route
metadata:
  name: flexo
spec:
  to:
    kind: Service
    name: flexo
```

## Security Considerations

- Set up network policies
- Configure service accounts
- Use OpenShift secrets management
- Enable TLS for routes

## Monitoring
```bash
# Check deployment status
oc get deployment flexo

# View logs
oc logs deployment/flexo

# Monitor resources
oc adm top pods
```

## Scaling
```bash
# Manual scaling
oc scale deployment/flexo --replicas=3

# Automatic scaling
oc autoscale deployment/flexo --min=2 --max=5 --cpu-percent=80
```

----