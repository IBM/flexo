# Pattern Detection Documentation

## Overview

The **Pattern Detection Module** provides utilities for detecting predefined patterns within text streams. This module leverages the **Aho-Corasick algorithm** — an efficient string-searching technique that matches multiple patterns simultaneously in linear time.

---

## Core Components

| Automaton Classes | Processor Classes |
|-------------------|-------------------|
| **`AhoCorasickAutomaton`**<br>✓ Trie-based pattern matching engine<br>✓ Linear-time complexity<br>✓ Simultaneous multi-pattern search | **`BaseBufferedProcessor`**<br>✓ Abstract base for text streaming<br>✓ Buffer management<br>✓ Chunk-based processing |
| **`AhoCorasickAutomatonNormalized`**<br>✓ Whitespace-insensitive matching<br>✓ Pattern normalization<br>✓ Original-to-normalized index mapping | **`AhoCorasickBufferedProcessor`**<br>✓ Exact pattern matching<br>✓ YAML-configurable patterns<br>✓ Streaming-ready implementation |
| | **`AhoCorasickBufferedProcessorNormalized`**<br>✓ Whitespace-invariant detection<br>✓ Flexible text matching<br>✓ Preserves original text positions |

### Utility Functions

* **`load_patterns(yaml_path)`** — Loads patterns from YAML configuration
* **`normalize_and_map(text)`** — Removes whitespace while tracking original character positions

---

## How It Works

The module processes text in two primary stages:

1. **Pattern Preprocessing**

     - Patterns are loaded from YAML configuration
     - The Aho-Corasick automaton is constructed from patterns
     - Failure links connect states for efficient pattern transitions

2. **Buffered Text Processing**

     - Text is processed in manageable chunks
     - Partial matches at chunk boundaries are preserved
     - Match information includes pattern name and position

### Text Normalization Pipeline

```
┌─────────────┐     ┌─────────────┐     ┌───────────────┐     ┌─────────────┐
│ Input Text  │ ──► │ Normalize   │ ──► │ Pattern Match │ ──► │ Map to      │
│             │     │ (Remove     │     │ (Using        │     │ Original    │
│             │     │  Whitespace)│     │  Automaton)   │     │ Positions   │
└─────────────┘     └─────────────┘     └───────────────┘     └─────────────┘
```

---

## Performance Considerations

**Time Complexity** O(n + m + k) where:

  * n = length of input text
  * m = total length of all patterns
  * k = number of pattern occurrences

**Space Efficiency**:

  * Buffered processing minimizes memory usage
  * Suitable for streaming applications with unbounded input

**Flexibility vs. Performance**:

  * Standard processors offer exact matching with minimal overhead
  * Normalized processors provide flexibility with slight computational cost

---

## Usage Examples

```python
# Standard pattern matching (exact)
processor = AhoCorasickBufferedProcessor('patterns.yaml')
result = await processor.process_chunk("input text")
print(f"Pattern found: {result.pattern_name}" if result.matched else "No match")

# Whitespace-insensitive pattern matching
norm_processor = AhoCorasickBufferedProcessorNormalized('patterns.yaml')
result = await norm_processor.process_chunk("input   text  with   spacing")
print(f"Pattern found: {result.pattern_name}" if result.matched else "No match")
```

---

## Additional Resources

* [LLM Module Overview](../index.md)
* [Tool Call Detection](../tool_detection/index.md)