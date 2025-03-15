# Model Configuration Guide

This guide provides detailed information on configuring language models for use with the Flexo Framework. Model configurations are specified in the `models_config` section of `src/configs/agent.yaml`.

## Table of Contents

- [Configuration Structure](#configuration-structure)
- [Supported Model Providers](#supported-model-providers)
  - [Cloud Providers](#cloud-providers)
  - [Self-Hosted Options](#self-hosted-options)
- [Configuration Parameters](#configuration-parameters)
  - [Common Parameters](#common-parameters)
  - [Provider-Specific Parameters](#provider-specific-parameters)
- [Environment Variables](#environment-variables)
- [Model Selection in Agent Configuration](#model-selection-in-agent-configuration)
- [Advanced Configurations](#advanced-configurations)
- [Examples](#examples)

## Configuration Structure

The model configuration in `models.yaml` follows this basic structure:

```yaml
models:
  model_name:
    vendor: provider_name
    model_id: model_identifier
    # Additional parameters...
```

Each model entry consists of:

- A unique `model_name` which will be referenced in your agent configuration
- A `vendor` identifier that determines which adapter to use
- A `model_id` that specifies the actual model to use from the vendor
- Additional parameters specific to the provider or model

## Supported Model Providers

### Cloud Providers

| Provider    | Vendor Key | API Endpoints | Environment Variables |
|-------------|-----------|--------------|----------------------|
| OpenAI      | `openai` | `/chat/completions` | `OPENAI_API_KEY` |
| Anthropic   | `anthropic` | `/messages` | `ANTHROPIC_API_KEY` |
| xAI         | `xai` | `/chat/completions` | `XAI_API_KEY` |
| Mistral AI  | `mistral-ai` | `/chat/completions` | `MISTRAL_API_KEY` |
| IBM WatsonX | `watsonx-llama`, `watsonx-granite`, `watsonx-mistral` | `/text/chat_stream`, `/text/generation_stream` | `WATSONX_API_KEY`, `WATSONX_PROJECT_ID` |

### Self-Hosted Options

The `openai-compat` adapter types support any API that implements OpenAI-compatible endpoints:

| Implementation | Vendor Key | Default Base URL |
|----------------|------------|-----------------|
| vLLM | `openai-compat` | `http://localhost:8000/v1` |
| Ollama | `openai-compat` | `http://localhost:11434/v1` |
| LLaMA.cpp | `openai-compat` | `http://localhost:8080/v1` |
| LM Studio | `openai-compat` | `http://localhost:1234/v1` |
| LocalAI | `openai-compat` | `http://localhost:8080/v1` |
| Text Generation WebUI | `openai-compat` | `http://localhost:5000/v1` |

## Configuration Parameters

### Common Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `model_id` | string | Identifier for the specific model to use | Required |
| `vendor` | string | Provider identifier (see tables above) | Required |
| `temperature` | float | Controls randomness in generation (0.0-1.0) | 0.7 |
| `max_tokens` | integer | Maximum tokens to generate | 1024 |
| `top_p` | float | Alternative to temperature for nucleus sampling (0.0-1.0) | 1.0 |
| `stop` | array | Sequences that will stop generation when produced | `[]` |

### Provider-Specific Parameters Examples

#### OpenAI

```yaml
gpt-4o:
  vendor: openai
  model_id: gpt-4-turbo
  temperature: 0.7
  max_tokens: 4096
  top_p: 1.0
  presence_penalty: 0.0  # OpenAI-specific
  frequency_penalty: 0.0  # OpenAI-specific
```

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `presence_penalty` | float | Penalty for token presence (-2.0 to 2.0) | 0.0 |
| `frequency_penalty` | float | Penalty for token frequency (-2.0 to 2.0) | 0.0 |

#### Anthropic

```yaml
claude-35:
  vendor: anthropic
  model_id: claude-3-opus-20240229
  temperature: 0.5
  max_tokens: 4096
  top_p: 0.9
  top_k: 50  # Anthropic-specific
```

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `top_k` | integer | Limits token selection to top K options | 50 |

#### xAI

```yaml
grok-2:
  vendor: xai
  model_id: grok-2-latest
  temperature: 0.7
  max_tokens: 4096
  top_p: 0.95
  base_url: https://api.x.ai/v1  # Optional override
```

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `base_url` | string | API endpoint URL | `https://api.x.ai/v1` |

#### IBM watsonX

```yaml
granite-8b:
  vendor: watsonx-granite
  model_id: ibm/granite-3-8b-instruct
  temperature: 0.5
  max_tokens: 1024
  time_limit: 60  # watsonX-specific
```

| Parameter | Type | Description                        | Default |
|-----------|------|------------------------------------|---------|
| `time_limit` | integer | Maximum generation time in seconds | 60 |

#### Mistral AI

```yaml
mistral-large:
  vendor: mistral-ai
  model_id: mistral-large-latest
  temperature: 0.7
  max_tokens: 4096
  safe_prompt: true  # Mistral-specific
```

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `safe_prompt` | boolean | Enable content filtering | `true` |

#### OpenAI-Compatible

```yaml
local-model:
  vendor: openai-compat  # or openai-compat-granite, openai-compat-llama, etc.
  model_id: your-model-name
  base_url: http://localhost:8000/v1
  api_key: dummy-key
  temperature: 0.7
  max_tokens: 2048
```

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `base_url` | string | The API endpoint URL | Required |
| `api_key` | string | API key (depending on implementation) | Required |

## Environment Variables

The following environment variables are used for API authentication:

| Provider          | Environment Variable | Required |
|-------------------|---------------------|----------|
| OpenAI            | `OPENAI_API_KEY` | Yes |
| Anthropic         | `ANTHROPIC_API_KEY` | Yes |
| xAI               | `XAI_API_KEY` | Yes |
| Mistral AI        | `MISTRAL_API_KEY` | Yes |
| IBM WatsonX       | `WATSONX_API_KEY`, `WATSONX_PROJECT_ID` | Yes |
| OpenAI-Compatible | Varies (can be set in config) | Depends |

## Model Selection in Agent Configuration

In your `agent.yaml` file, you specify which model to use:

```yaml
models_config:
  main_chat_model:
    vendor: openai
    model_id: gpt-4o
    # Parameters...
```

You can also configure the agent to use different models for different purposes:

```yaml
models_config:
  main_chat_model:
    vendor: anthropic
    model_id: claude-3-5-sonnet-latest
    # Primary model parameters...
  
  watsonx_granite:
    vendor: watsonx-granite
    model_id: ibm/granite-3-8b-instruct
    # Additional model instance parameters
```

## Advanced Configurations

### Vendor-Specific Prompt Builders

Flexo automatically selects the appropriate prompt builder based on the vendor:

| Vendor | Prompt Builder Link                                                                                     |
|-------|---------------------------------------------------------------------------------------------------------|
| `openai` | [`OpenAIPromptBuilder`](reference/prompt_builders/openai_prompt_builder.md)                             |
| `anthropic` | [`AnthropicPromptBuilder`](reference/prompt_builders/anthropic_prompt_builder.md)                       |
| `mistral-ai` | [`MistralAIPromptBuilder` ](reference/prompt_builders/mistral_ai_prompt_builder.md)                     |
| `watsonx-granite` | [`WatsonXGranitePromptBuilder`](reference/prompt_builders/watsonx/granite/granite_prompt_builder.md)    |
| `watsonx-llama` | [`WatsonXLlamaPromptBuilder`](reference/prompt_builders/watsonx/llama/llama_prompt_builder.md)          |
| `watsonx-mistral` | [`WatsonXMistralPromptBuilder`](reference/prompt_builders/watsonx/mistral/mistral_prompt_builder.md)    |
| `openai-compat-granite` | [`OpenAICompatGranitePromptBuilder`](reference/prompt_builders/openai_compat/granite_prompt_builder.md) |
| `openai-compat-llama` | [`OpenAICompatLlamaPromptBuilder`](reference/prompt_builders/openai_compat/llama_prompt_builder.md)     |
| `xai` | [`XAIPromptBuilder`](reference/prompt_builders/xai_prompt_builder.md)                                   |

### Tool Detection Modes

The `detection_mode` in your agent configuration affects how tool calls are detected:

- `vendor`: Uses the provider's native tool-calling capabilities (recommended for cloud providers)
- `manual`: Uses Flexo's pattern-matching for tool calls (useful for local models)

```yaml
detection_mode: vendor  # or 'manual'
use_vendor_chat_completions: true  # or false
```

---

## Examples

### OpenAI GPT-4

```yaml
gpt-4:
  vendor: openai
  model_id: gpt-4o
  temperature: 0.7
  max_tokens: 4096
  presence_penalty: 0.0
  frequency_penalty: 0.0
```

### Anthropic Claude

```yaml
claude:
  vendor: anthropic
  model_id: claude-3-5-sonnet-latest
  temperature: 0.7
  max_tokens: 4096
  top_p: 0.9
```

### xAI

```yaml
grok:
  vendor: xai
  model_id: grok-2-latest
  temperature: 0.7
  max_tokens: 4096
```

### Mistral AI

```yaml
mistral:
  vendor: mistral-ai
  model_id: mistral-large-latest
  temperature: 0.7
  max_tokens: 4096
```

### IBM WatsonX (Llama)

```yaml
watsonx-llama:
  vendor: watsonx-llama
  model_id: meta-llama/llama-3-405b-instruct
  decoding_method: greedy
  max_tokens: 4000
  temperature: 0.7
```

### IBM WatsonX (Granite)

```yaml
watsonx-granite:
  vendor: watsonx-granite
  model_id: ibm/granite-3-8b-instruct
  decoding_method: greedy
  max_tokens: 4000
  temperature: 0.7
```

### vLLM (Local Deployment)

```yaml
vllm-local:
  vendor: openai-compat-llama
  model_id: meta-llama/Llama-3.2-8B-Instruct
  base_url: http://localhost:8000/v1
  api_key: dummy-key
  temperature: 0.7
  max_tokens: 2048
```

### Ollama (Local Deployment)

```yaml
ollama-local:
  vendor: openai-compat-granite
  model_id: granite31  # or any model name in Ollama
  base_url: http://localhost:11434/v1
  api_key: ollama
  temperature: 0.7
  max_tokens: 2048
```

---