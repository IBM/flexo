# Agent Configuration Guide

## Overview

This guide explains how to configure the Flexo agent's behavior through the `agent.yaml` configuration file.

## Configuration Structure

### Basic Settings

```yaml
name: flexo
history_limit: 4
timeouts:
  model_response_timeout: 60
max_streaming_iterations: 2
detection_mode: vendor
use_vendor_chat_completions: true
```

### Key Configurations:

- **`detection_mode`**: Supports `'vendor'` or `'manual'` tool call detection (corresponds to `VendorToolCallDetectionStrategy` or `ManualToolCallDetectionStrategy`).
- **`use_vendor_chat_completions`**: Enables the use of the chat completions API instead of the text generation endpoint.
- **`history_limit`**: Controls the conversation context window by limiting the number of previous messages retained.
- **`max_streaming_iterations`**: Limits the number of times the streaming state can be entered in a single session to prevent looping. Must be set to 2+ when using tools.


### System Prompt

```yaml
system_prompt: |
  I am a helpful AI assistant focused on clear, accurate, and direct communication. 
  I solve problems systematically and explain my reasoning when needed. 
  I maintain a professional yet approachable tone.
  
  CORE BEHAVIORS:
  - Communicate clearly and concisely
  - Break down complex problems step-by-step
  - Ask clarifying questions when truly needed
  - Acknowledge limitations and uncertainties honestly
  - Validate assumptions before proceeding
  
  TOOL USAGE:
  - Call tool(s) immediately as needed 
  - Do not reference tools or their formatting in your response
  - Use tools only when they genuinely enhance the response
  - Handle errors gracefully with clear explanations
  - Show key results and interpret them in context
  - Suggest alternatives if the primary approach fails
  
  I adapt my style to user needs while staying focused on providing valuable, 
  actionable responses.
```

### Models Configuration

```yaml
models_config:
  main_chat_model:
    vendor: watsonx-llama  # Supported: anthropic, openai, watsonx (llama, mistral, granite), mistral-ai
    model_id: meta-llama/llama-3-405b-instruct
    decoding_method: greedy
    max_new_tokens: 4000
```

### Tool Configuration

Tools are configured as a list under `tools_config`. The name specified in each tool configuration is used to search for the corresponding implementation.

```yaml
tools_config:
  # Weather API Integration
  - name: "weather"
    endpoint_url: "https://api.openweathermap.org/data/2.5/weather"
    api_key_env: "OWM_API_KEY"

  # Wikipedia Summary Tool
  - name: "wikipedia"
    endpoint_url: "https://{lang}.wikipedia.org/api/rest_v1/page/summary/{encoded_query}"

  # DuckDuckGo Search Tool
  - name: "duckduckgo_search"
```

#### RAG Tool Example (Elasticsearch)

```yaml
tools_config:
  - name: "medicare_search"
    connector_config:
      connector_type: elasticsearch
      index_name: my_index
      api_key_env: ES_API_KEY
      endpoint_env: ES_ENDPOINT
      top_k: 5
      timeout: 30
      max_retries: 3
      query_body:
        _source: false
        fields: ["text"]
        query:
          bool:
            must:
              - match:
                  text:
                    query: "$USER_INPUT"
                    boost: 3.5
        knn:
          field: vector
          query_vector_builder:
            text_embedding:
              model_id: thenlper__gte-base
              model_text: "$USER_INPUT"
          k: 100
          num_candidates: 150
        rank:
          rrf:
            rank_window_size: 40
```

---

## Implementation Details

### Tool Detection Strategies

Based on the codebase's `tool_detection` module:

1. **Vendor Detection** (`detection_mode: vendor`):

     - Uses vendor's native tool calling capabilities
     - Handled by `VendorToolCallDetectionStrategy`

2. **Manual Detection** (`detection_mode: manual`):

     - Custom pattern-matching for tool calls
     - Handled by `ManualToolCallDetectionStrategy`

### Prompt Builders

The system automatically selects the appropriate prompt builder based on the vendor:

- `AnthropicPromptBuilder`
- `OpenAIPromptBuilder`
- `MistralAIPromptBuilder`
- `GranitePromptBuilder`
- `LlamaPromptBuilder`
- `MistralPromptBuilder`

---

## Best Practices

1. **Tool Configuration**

     - Use environment variables for sensitive credentials
     - Configure appropriate timeouts and retry logic
     - Set meaningful tool descriptions
     - Use the appropriate tool name to link to implementation

2. **System Prompt**

     - Keep instructions clear and specific
     - Include explicit tool usage guidelines
     - Define error handling preferences

---

## Related Documentation
- See `tools/` documentation for detailed tool configuration
- Check `llm/adapters/` for vendor-specific options
- Review `prompt_builders/` for prompt customization

---