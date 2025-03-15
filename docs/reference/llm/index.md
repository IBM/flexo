# LLM Documentation

## Overview

The **LLM (Large Language Model) Module** provides interfaces and implementations for interacting with various LLM providers. It includes factory methods for creating LLM instances, adapters for different vendors, and utilities for pattern detection and tool call handling.

## Components

### LLM Factory

- **`LLMFactory`** - A factory class for creating LLM instances based on the selected provider.

### LLM Adapters
Adapters to support multiple LLM providers:

- **`BaseVendorAdapter`** - Abstract base class for all LLM vendor adapters.
- **`AnthropicAdapter`** - Adapter for Anthropic’s Claude models.
- **`MistralAdapter`** - Adapter for Mistral AI models.
- **`OpenAIAdapter`** - Adapter for OpenAI’s GPT models.
- **`WatsonXAdapter`** - Adapter for IBM WatsonX models.

#### WatsonX Components

- **`WatsonXConfig`** - Configuration settings for WatsonX integration.
- **`IBMTokenManager`** - Token manager for IBM WatsonX authentication.

### Pattern Detection

- **`AhoCorasickAutomaton`** - An automaton for efficient pattern matching.
- **`AhoCorasickBufferedProcessor`** - A buffered processor using the Aho-Corasick algorithm.

### Tool Call Detection

- **`BaseToolCallDetectionStrategy`** - Abstract class for tool call detection strategies.
- **`DetectionState`** and **`DetectionResult`** - Data models representing the state and results of detection.
- **`ManualToolCallDetectionStrategy`** - A manual approach to tool call detection.
- **`VendorToolCallDetectionStrategy`** - Vendor-specific tool call detection strategies.

## Additional Resources

- [LLM Adapters](adapters/index.md)
- [Pattern Detection](pattern_detection/index.md)
- [Tool Call Detection](tool_detection/index.md)
