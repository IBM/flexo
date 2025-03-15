# Welcome to Flexo

The [**Flexo Agent Library**](https://github.com/IBM/flexo) is a powerful and flexible codebase that enables users to configure, customize, and deploy a generative AI agent. Designed for adaptability, the library can be tailored to a wide range of use cases, from conversational AI to specialized automation.

---

## Why Flexo?

- **Simplified Deployment**: Deploy anywhere with comprehensive platform guides
- **Production Ready**: Built for scalability and reliability
- **Extensible**: Add custom tools and capabilities
- **Well Documented**: Clear guides for every step

---

## Key Features

- **Configurable Agent**: YAML-based configuration for custom behaviors
- **Tool Integration**: Execute Python functions and REST API calls
- **Streaming Support**: Real-time streaming with pattern detection
- **Production Ready**: Containerized deployment support with logging
- **FastAPI Backend**: Modern async API with comprehensive docs

---

# Supported LLM Providers

<div class="grid" markdown style="text-align: center">

<div markdown>

<h2>☁️ Cloud Providers</h2>

<div class="provider-grid">
    <div class="provider-card">
        <img src="images/OpenAI-white-monoblossom.png" width="90">
        <p><strong>OpenAI</strong></p>
        <p class="description">GPT-powered models</p>
    </div>
    <div class="provider-card">
        <img src="images/watsonxai.png" width="70">
        <p><strong>watsonx.ai</strong></p>
        <p class="description">Enterprise AI solutions</p>
    </div>
    <div class="provider-card">
        <img src="images/Anthropic.png" width="70">
        <p><strong>Anthropic</strong></p>
        <p class="description">Claude family models</p>
    </div>
    <div class="provider-card">
        <img src="images/grok.png" width="70">
        <p><strong>xAI</strong></p>
        <p class="description">Grok and beyond</p>
    </div>
    <div class="provider-card">
        <img src="images/mistral-color.png" width="70">
        <p><strong>Mistral AI</strong></p>
        <p class="description">Efficient open models</p>
    </div>
</div>

</div>

<div markdown style="text-align: center">

<h2>🖥️ Local & Self-Hosted Options</h2>

<div class="provider-grid">
    <div class="provider-card">
        <img src="images/vllm-logo-text-dark.png" width="140">
        <p class="description">High-throughput serving</p>
    </div>
    <div class="provider-card">
        <img src="images/ollama.png" width="70">
        <p><strong>Ollama</strong></p>
        <p class="description">Easy local LLMs</p>
    </div>
    <div class="provider-card">
        <img src="images/llamacpp.png" width="190">
        <p class="description">Optimized C++ runtime</p>
    </div>
    <div class="provider-card">
        <img src="images/lm-studio.png" width="70">
        <p><strong>LM Studio</strong></p>
        <p class="description">User-friendly interface</p>
    </div>
    <div class="provider-card">
        <img src="images/localai.png" width="70">
        <p><strong>LocalAI</strong></p>
        <p class="description">Self-hosted versatility</p>
    </div>
</div>

</div>

</div>


## ⚙️ Unified Configuration Interface

Switch providers effortlessly with Flexo's adapter layer. Customize your LLM settings in one place:

```yaml
gpt-4o:
  provider: "openai"  # Choose your provider
  model: "gpt-4o"     # Select specific model
  temperature: 0.7
  max_tokens: 4000    # Additional model-specific parameters
```

> **Need more details?** Check our comprehensive [Model Configuration Guide](model-configuration.md) for provider-specific settings and optimization tips.

<style>
.grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 20px;
}

.provider-grid {
    display: grid;
    grid-template-columns: 1fr;
    gap: 15px;
    margin-top: 15px;
}

.provider-card {
    text-align: center;
    padding: 15px;
    border-radius: 4px;
    background-color: rgba(255, 255, 255, 0.05);
}

.provider-card img {
    margin-bottom: 10px;
}

.provider-card p {
    margin: 5px 0;
}

.description {
    font-size: 0.9em;
    opacity: 0.8;
}
</style>

---

## Quick Start Guide

### 1. Local Development
Start developing with Flexo locally:

- [Configure Your Agent](agent-configuration.md)
- [Run the Server](getting-started.md)
- [Build from Source](deployment/building-image.md)

### 2. Production Deployment
Deploy Flexo to your preferred platform:

| Platform | Best For | Guide |
|----------|----------|-------|
| IBM Code Engine | Serverless, pay-per-use | [Deploy →](deployment/platforms/code-engine.md) |
| AWS Fargate | AWS integration | [Deploy →](deployment/platforms/fargate.md) |
| OpenShift | Enterprise, hybrid cloud | [Deploy →](deployment/platforms/openshift.md) |
| Kubernetes | Custom infrastructure | [Deploy →](deployment/platforms/kubernetes.md) |

---

## Documentation

### Deployment Guides

- [Container Registries](deployment/registries/overview.md)
- [Platform Deployment](deployment/platforms/overview.md)

### Code Reference

- [Agent](reference/agent/chat_agent_streaming)
- [API Reference](reference/api.md)
- [Model Configuration](model-configuration.md)
- [Tools System](reference/tools/index.md)
- [Data Models](reference/data_models.md)
- [Database Integration](reference/database.md)

---


## System Architecture

```mermaid
graph TB
    Client[Client] --> API[FastAPI Server]
    
    subgraph API["API Layer"]
        Router[Chat Completions Router]
        SSE[SSE Models]
        Validation[Request Validation]
    end
    
    subgraph Agent["Agent Layer"]
        ChatAgent[Streaming Chat Agent]
        State[State Management]
        Config[Configuration]
    end
    
    subgraph LLM["LLM Layer"]
        Factory[LLM Factory]
        Detection[Tool Detection]
        Builders[Prompt Builders]
        
        subgraph Adapters["LLM Adapters"]
            WatsonX
            OpenAI
            Anthropic
            Mistral
        end
    end
    
    subgraph Tools["Tools Layer"]
        Registry[Tool Registry]
        REST[REST Tools]
        NonREST["Non-REST Tools: Python etc"]
        ExampleTools["Included Example 
        Tools: RAG + Weather"]
    end
    
    subgraph Database["Database Layer"]
        ES[Elasticsearch]
        Milvus[Milvus]
    end
    
    API --> Agent
    Agent --> LLM
    Agent --> Tools
    Tools --> Database
    
    style API stroke:#ff69b4,stroke-width:4px
    style Agent stroke:#4169e1,stroke-width:4px
    style LLM stroke:#228b22,stroke-width:4px
    style Tools stroke:#cd853f,stroke-width:4px
    style Database stroke:#4682b4,stroke-width:4px
    
    style Router stroke:#ff69b4,stroke-width:2px
    style SSE stroke:#ff69b4,stroke-width:2px
    style Validation stroke:#ff69b4,stroke-width:2px
    
    style ChatAgent stroke:#4169e1,stroke-width:2px
    style State stroke:#4169e1,stroke-width:2px
    style Config stroke:#4169e1,stroke-width:2px
    
    style Factory stroke:#228b22,stroke-width:2px
    style Detection stroke:#228b22,stroke-width:2px
    style Builders stroke:#228b22,stroke-width:2px
    style Adapters stroke:#228b22,stroke-width:2px
    style WatsonX stroke:#228b22,stroke-width:2px
    style OpenAI stroke:#228b22,stroke-width:2px
    style Anthropic stroke:#228b22,stroke-width:2px
    style Mistral stroke:#228b22,stroke-width:2px
    
    style Registry stroke:#cd853f,stroke-width:2px
    style ExampleTools stroke:#cd853f,stroke-width:2px
    style REST stroke:#cd853f,stroke-width:2px
    style NonREST stroke:#cd853f,stroke-width:2px
    
    style ES stroke:#4682b4,stroke-width:2px
    style Milvus stroke:#4682b4,stroke-width:2px
    
    style Client stroke:#333,stroke-width:2px
```

---

## Chat Agent State Flow

```mermaid
stateDiagram-v2
    [*] --> STREAMING: Initialize Agent
    
    STREAMING --> TOOL_DETECTION: Tool Call Found
    STREAMING --> COMPLETING: Generation Done
    
    TOOL_DETECTION --> EXECUTING_TOOLS: Process Tool Call
    EXECUTING_TOOLS --> INTERMEDIATE: Tool Execution Complete
    
    INTERMEDIATE --> STREAMING: Continue Generation
    INTERMEDIATE --> COMPLETING: Max Iterations
    
    COMPLETING --> [*]: End Response
    
    note right of STREAMING
        Main generation state
        Handles LLM responses
    end note
    
    note right of EXECUTING_TOOLS
        Concurrent tool execution
        Error handling
    end note
```


---

## Contributing
See our [Contributing Guide](https://github.com/IBM/flexo/blob/main/CONTRIBUTING.md) for details.

---

## Security
For security concerns, please review our [Security Policy](https://github.com/IBM/flexo/blob/main/SECURITY.md).

---