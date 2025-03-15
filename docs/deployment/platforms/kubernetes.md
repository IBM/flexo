# Deploying Flexo to Kubernetes

This guide covers deploying Flexo to a generic Kubernetes cluster.

## Prerequisites

- Access to Kubernetes cluster
- kubectl CLI installed and configured
- Container image in accessible registry
- Kubernetes 1.19+ recommended

## Deployment Steps

### 1. Create Namespace
```bash
kubectl create namespace flexo
kubectl config set-context --current --namespace=flexo
```

### 2. Create Configuration
```yaml
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: flexo-config
data:
  agent.yaml: |
    # Your agent configuration here
  LOGGING_LEVEL: "INFO"

---
# secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: flexo-secrets
type: Opaque
stringData:
  WATSON_API_KEY: <your-key>
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
  selector:
    matchLabels:
      app: flexo
  template:
    metadata:
      labels:
        app: flexo
    spec:
      containers:
      - name: flexo
        image: your-registry/flexo:latest
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

### 4. Create Service
```yaml
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: flexo
spec:
  type: ClusterIP
  ports:
  - port: 8000
  selector:
    app: flexo

---
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: flexo
spec:
  rules:
  - host: flexo.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: flexo
            port:
              number: 8000
```

## Resource Management

### Horizontal Pod Autoscaling
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: flexo
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: flexo
  minReplicas: 1
  maxReplicas: 5
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 80
```

## Monitoring

### Health Checks
```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 30
  periodSeconds: 10

readinessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 5
  periodSeconds: 5
```

---