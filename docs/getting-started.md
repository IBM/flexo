# Getting Started

This guide will help you set up and run the generative AI agent project.

## Prerequisites

- Python 3.10+: [Install Python](https://www.python.org/downloads/)
- Git: [Install Git](https://git-scm.com/)
- Docker (optional): [Install Docker](https://docs.docker.com/get-docker/)
- Access to [flexo](https://github.com/IBM/flexo)

## Quick Setup

1. Fork and clone the repository:
   ```bash
   # First, fork the repository on GitHub by clicking the 'Fork' button
   # Then clone your fork:
   git clone https://github.com/YOUR_USERNAME/flexo.git
   cd flexo
   ```

2. Create Python environment:
   ```bash
   # Mac/Linux
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt

   # Windows
   python -m venv venv
   .\venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Configure the project:

   - Copy `.env.example` to `.env` and update with your settings
   - Review `src/configs/agent.yaml` for agent configuration

4. Start the API server:
   ```bash
   uvicorn src.main:app --reload --host 127.0.0.1 --port 8000
   ```

5. Test the API using one of these methods:

   A. Using cURL:
   ```bash
   curl -X POST http://127.0.0.1:8000/v1/chat/completions \
       -H "Content-Type: application/json" \
       -d '{
             "messages": [
               {
                 "role": "user",
                 "content": [
                   { "text": "Hello, AI!" }
                 ]
               }
             ],
             "context": {
               "values": [
                 { "key": "session_id", "value": "abc123" }
               ]
             },
             "include_status_updates": false,
             "stream": true
           }'
   ```

   > **Note:** Include an `-H "X-API-Key: your-flexo-api-key"` header when `ENABLE_API_KEY=true`

   B. Using the Streamlit testing interface:
   ```bash
   # Navigate to the tests directory
   cd tests
   
   # Run the Streamlit app
   streamlit run test_streamlit_app.py
   
   # This will open a browser window at http://localhost:8501 where you can interact with the agent
   ```

## Docker Deployment (Optional)
```bash
# Build the container
docker build -t flexo-agent .

# Run the container
docker run -p 8000:8000 --env-file .env flexo-agent
```

## Next Steps

- Explore configuration options in the [Agent Configuration Guide](agent-configuration.md)
- Learn about deployment in the [Deployment Guide](deployment/overview.md)
- Customize the agent's behavior and extend its functionality

## Testing Tips

- The Streamlit interface provides a more user-friendly way to test the agent's capabilities
- The Streamlit app is looking for your local agent instance running on port 8000
- For debugging, check both the API server logs and the Streamlit app logs

---