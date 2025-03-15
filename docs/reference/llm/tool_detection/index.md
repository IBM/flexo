# Tool Detection Documentation

## Overview

The **Tool Detection Module** is responsible for identifying and extracting tool calls from model responses. It provides various strategies to detect tool invocations, including manual and vendor-specific methods. This module is crucial for integrating LLMs with external tools and APIs.

## Components

### Detection Strategies

- **`BaseToolCallDetectionStrategy`** - An abstract base class that defines the interface for all tool call detection strategies.
- **`ManualToolCallDetectionStrategy`** - A manual approach where predefined patterns and heuristics are used to detect tool calls.
- **`VendorToolCallDetectionStrategy`** - A strategy that utilizes vendor-specific methods for detecting tool calls.

### Detection Results

- **`DetectionState`** - Represents the current state of tool call detection.
- **`DetectionResult`** - Stores the final detection results, including extracted tool call information.

## Additional Resources

- [LLM Overview](../index.md)
- [Pattern Detection](../pattern_detection/index.md)
