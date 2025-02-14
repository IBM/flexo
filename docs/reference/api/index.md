# API Documentation

## Overview

The API module provides HTTP endpoints for interacting with the LLM agent system. It consists of two main components:

1. Chat Completions API - REST endpoints for chat interactions
2. Server-Sent Events (SSE) Models - Data models for streaming responses

## Components

### Chat Completions API

The Chat Completions API provides REST endpoints that follow patterns similar to OpenAI's Chat API, making it familiar for developers. Key features include:

- Support for multiple message types (user, system, assistant)
- Function/tool calling capabilities
- Streaming and non-streaming responses
- Temperature and other LLM parameter controls

Example endpoint: `/v1/chat/completions`

### SSE Models

The SSE (Server-Sent Events) models handle streaming responses from the system. These models structure the data for:

- Incremental token/text-chunk streaming
- Function/tool call events
- Status updates
- Error messages

## API Reference

For detailed information about specific components, see:

- [API Request Models](request_models.md) - Details about the chat endpoint models
- [SSE Models](sse_models.md) - Information about streaming response models

## See Also

- [Agent Configuration](../agent/index.md) - Configure the underlying agent
- [Tool Registry](../tools/tool_registry.md) - Available tools and functions
- [LLM Factory](../llm/llm_factory.md) - LLM provider configuration