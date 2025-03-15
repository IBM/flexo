# Deploying Flexo to AWS Fargate

This guide walks through deploying Flexo to AWS Fargate, a serverless container platform that integrates with Amazon ECS (Elastic Container Service).

## Prerequisites

- AWS account with appropriate permissions
- AWS CLI installed and configured
- Container image in Amazon ECR or other accessible registry
- Task execution role with required permissions
 
## Deployment Steps

### 1. Create ECR Repository and Push Image
```bash
# Create repository
aws ecr create-repository --repository-name flexo

# Login to ECR
aws ecr get-login-password --region us-east-1 | \
    docker login --username AWS --password-stdin \
    ${AWS_ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com

# Build and push
docker buildx build --platform linux/amd64 \
    -t ${AWS_ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com/flexo:latest \
    --push .
```

### 2. Create Environment Variables
Store your configuration in AWS Systems Manager Parameter Store:

**For secrets (using SecureString):**
```bash
# Store sensitive values
aws ssm put-parameter \
    --name "/flexo/prod/WXAI_APIKEY" \
    --value "your-api-key" \
    --type "SecureString"

aws ssm put-parameter \
    --name "/flexo/prod/OWM_API_KEY" \
    --value "owm-api-key" \
    --type "SecureString"

aws ssm put-parameter \
    --name "/flexo/prod/ES_ES_API_KEY" \
    --value "elastic-api-key" \
    --type "SecureString"

aws ssm put-parameter \
    --name "/flexo/prod/FLEXO_API_KEY" \
    --value "flexo-api-key" \
    --type "SecureString"
```

**For configuration (using String):**
```bash
# Store non-sensitive values
aws ssm put-parameter \
    --name "/flexo/prod/WXAI_URL" \
    --value "https://us-south.ml.cloud.ibm.com" \
    --type "String"

aws ssm put-parameter \
    --name "/flexo/prod/WXAI_PROJECT_ID" \
    --value "your-project-id" \
    --type "String"

aws ssm put-parameter \
    --name "/flexo/prod/ES_INDEX_NAME" \
    --value "my_index" \
    --type "String"

aws ssm put-parameter \
    --name "/flexo/prod/ES_ES_ENDPOINT" \
    --value "elastic-endpoint" \
    --type "String"
```

### 3. Create ECS Task Definition
```json
{
  "family": "flexo",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "containerDefinitions": [{
    "name": "flexo",
    "image": "${AWS_ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com/flexo:latest",
    "portMappings": [{
      "containerPort": 8000,
      "protocol": "tcp"
    }],
    "secrets": [
      {
        "name": "WXAI_APIKEY",
        "valueFrom": "arn:aws:ssm:us-east-1:${AWS_ACCOUNT_ID}:parameter/flexo/prod/WXAI_APIKEY"
      },
      {
        "name": "OWM_API_KEY",
        "valueFrom": "arn:aws:ssm:us-east-1:${AWS_ACCOUNT_ID}:parameter/flexo/prod/OWM_API_KEY"
      }
    ],
    "environment": [
      {
        "name": "WXAI_URL",
        "valueFrom": "arn:aws:ssm:us-east-1:${AWS_ACCOUNT_ID}:parameter/flexo/prod/WXAI_URL"
      }
    ],
    "logConfiguration": {
      "logDriver": "awslogs",
      "options": {
        "awslogs-group": "/ecs/flexo",
        "awslogs-region": "us-east-1",
        "awslogs-stream-prefix": "ecs"
      }
    }
  }]
}
```

### 4. Create ECS Service
```bash
# Create cluster if needed
aws ecs create-cluster --cluster-name flexo-cluster

# Create service
aws ecs create-service \
    --cluster flexo-cluster \
    --service-name flexo \
    --task-definition flexo:1 \
    --desired-count 1 \
    --launch-type FARGATE \
    --network-configuration "awsvpcConfiguration={subnets=[subnet-xxxxx],securityGroups=[sg-xxxxx],assignPublicIp=ENABLED}"
```

## Configuration Options

### Scaling
Use Application Auto Scaling to set up automatic scaling:
```bash
# Register scalable target
aws application-autoscaling register-scalable-target \
    --service-namespace ecs \
    --scalable-dimension ecs:service:DesiredCount \
    --resource-id service/flexo-cluster/flexo \
    --min-capacity 1 \
    --max-capacity 5

# Configure scaling policy
aws application-autoscaling put-scaling-policy \
    --service-namespace ecs \
    --scalable-dimension ecs:service:DesiredCount \
    --resource-id service/flexo-cluster/flexo \
    --policy-name cpu-scaling \
    --policy-type TargetTrackingScaling \
    --target-tracking-scaling-policy-configuration file://scaling-policy.json
```

## Monitoring
```bash
# View logs
aws logs get-log-events \
    --log-group-name /ecs/flexo \
    --log-stream-name your-log-stream

# Check service status
aws ecs describe-services \
    --cluster flexo-cluster \
    --services flexo
```

## Possible Next Steps

- Set up Application Load Balancer
- Configure CloudWatch alarms
- Set up CI/CD pipeline with AWS CodePipeline

----