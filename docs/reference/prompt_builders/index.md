# Prompt Builders Documentation

## Overview

The **Prompt Builders** module provides utilities for constructing prompts tailored to various LLM (Large Language Model) providers. These builders help structure prompts according to the expected input format of different models, ensuring optimal performance.

## Components

### Base Prompt Builder

- `BasePromptBuilder`: The foundational class that all other prompt builders extend.

### Provider-Specific Prompt Builders

- `AnthropicPromptBuilder`: Generates prompts formatted for **Anthropic's Claude** models.
- `MistralAIPromptBuilder`: Constructs prompts for **Mistral AI** models.
- `OpenAIPromptBuilder`: Handles prompt generation for **OpenAI's GPT models**.
- `GranitePromptBuilder`: Tailored for **WatsonX Granite** models.
- `LlamaPromptBuilder`: Supports **WatsonX Llama** models.
- `MistralPromptBuilder`: Specialized for **WatsonX Mistral** models.
- `XAIPromptBuilder`: Specialized for **xAI** models.

### Prompt Models

- `PromptPayload`: Defines the structure of the prompt content.
- `PromptBuilderOutput`: Represents the formatted output of a prompt builder.

### Using Different Providers
Each builder follows a similar pattern but optimizes prompt formatting based on the model requirements.

---