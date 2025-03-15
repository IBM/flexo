# Installing and Configuring Elasticsearch with Custom Models

## Prerequisites

- Python 3.7+ to 3.11
- pip package manager
- Access to an Elasticsearch instance
- Administrative privileges (for installation)

## Installation Steps

### 1. Installing Elasticsearch Client

First, install the official Elasticsearch Python client:

```bash
pip install elasticsearch==8.15.0
```

### 2. Installing Eland and Dependencies

Eland is Elasticsearch's machine learning client that helps manage ML models:

```bash
pip install 'eland[pytorch]==8.15.0'
```

Note: If you encounter any dependency conflicts, you may need to change the version of your local `elasticsearch` and `eland` to match the version installed on your hosted Elastic instance. For example, in my case I needed to install version `8.15.0` locally in order to match what was installed at my Elastic endpoint where the huggingface model is being deployed.

## Uploading a Custom Model

### 1. Prepare Your Environment

Ensure you have the following information ready:
- Elasticsearch endpoint URL
- API key with appropriate permissions
- The Hugging Face model ID you want to import

### 2. Model Import Command

Use the following command structure to import your model:

```bash
eland_import_hub_model --url "ES_ENDPOINT" --es-api-key "ES_API_KEY" --hub-model-id "HUGGINGFACE_MODEL" --task-type text_embedding
```

Note: It's not ideal, but you can add the `--insecure` tag if needed to get past SSL cert issues. 

Replace the placeholder values:
- `ES_ENDPOINT`: Your Elasticsearch instance URL (e.g., "https://localhost:9200")
- `ES_API_KEY`: Your Elasticsearch API key
- `HUGGINGFACE_MODEL`: The Hugging Face model ID (e.g., "sentence-transformers/all-MiniLM-L6-v2")

### 3. Command Options Explained

- `--url`: Specifies the Elasticsearch endpoint
- `--es-api-key`: Authentication key for Elasticsearch
- `--hub-model-id`: The specific model to import from Hugging Face
- `--task-type`: Specifies the model's purpose (text_embedding in this case)
- `--insecure`: Allows connections to Elasticsearch without SSL verification

### 4. Verifying Model Installation and Deployment

After running the import command, you can verify the model installation and deployment status using several commands:

1. List all installed models:
```bash
curl -X GET "ES_ENDPOINT/_ml/trained_models" -H "Authorization: ApiKey ES_API_KEY"
```

2. Check specific model details:
```bash
curl -X GET "ES_ENDPOINT/_ml/trained_models/MODEL_ID" -H "Authorization: ApiKey ES_API_KEY"
```

3. Verify model deployment status:
```bash
curl -X GET "ES_ENDPOINT/_ml/trained_models/MODEL_ID/deployment/_stats" -H "Authorization: ApiKey ES_API_KEY"
```

4. Deploy model if not already deployed:
```bash
curl -X POST "ES_ENDPOINT/_ml/trained_models/MODEL_ID/deployment/_start" -H "Authorization: ApiKey ES_API_KEY"
```

The deployment status response will include:
- `state`: Current deployment state (starting, started, stopping, failed)
- `allocation_status`: Resource allocation status
- `nodes`: List of nodes where the model is deployed
- `inference_count`: Number of inferences performed
- `error`: Any error messages if deployment failed

You can also check the Elasticsearch logs for any deployment issues:
```bash
curl -X GET "ES_ENDPOINT/_cat/indices/.logs-ml*?v" -H "Authorization: ApiKey ES_API_KEY"
```

## Troubleshooting

Common issues and solutions:

1. **Connection Errors**
   - Verify your Elasticsearch instance is running
   - Check if the endpoint URL is correct
   - Ensure your API key has sufficient permissions

2. **Import Failures**
   - Confirm you have enough disk space
   - Verify the model ID is correct
   - Check Elasticsearch logs for detailed error messages

3. **SSL/TLS Issues**
   - If using self-signed certificates, ensure proper configuration
   - Consider using `--insecure` flag only in development environments

## Security Considerations

- Always use API keys instead of passwords
- Avoid using `--insecure` flag in production environments

## Additional Resources

- [Elasticsearch Documentation](https://www.elastic.co/guide/index.html)
- [Eland GitHub Repository](https://github.com/elastic/eland)
- [Hugging Face Model Hub](https://huggingface.co/models)
