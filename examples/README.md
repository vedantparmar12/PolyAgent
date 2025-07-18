# Examples Directory

This directory contains example code patterns for the Make It Heavy agent system. These examples help agents understand and follow project conventions.

## Available Examples

### agent_patterns.py
Demonstrates proper agent creation and usage patterns:
- Creating context-aware agents
- Handling tool responses
- Error handling patterns
- Streaming responses

### tool_patterns.py
Shows how to implement new tools:
- Inheriting from BaseTool
- Implementing required properties
- Error handling in execute method
- Returning structured data

### test_patterns.py
Testing patterns for the project:
- Unit test structure
- Mocking external APIs
- Testing async operations
- Fixture usage

### orchestrator_patterns.py
Multi-agent patterns:
- Parallel agent execution
- Task decomposition
- Response synthesis
- Progress tracking

## Using Examples in PRPs

When creating a PRP, reference these examples:

```yaml
- file: examples/tool_patterns.py
  why: Shows proper tool implementation structure
  
- file: examples/test_patterns.py
  why: Testing conventions to follow
```

## Adding New Examples

When adding new patterns:
1. Create a descriptive filename
2. Add comprehensive comments
3. Show both correct and incorrect usage
4. Update this README

## Best Practices Demonstrated

1. **Error Handling**: Every example shows proper try/except usage
2. **Type Hints**: All functions have complete type annotations
3. **Documentation**: Google-style docstrings throughout
4. **Testing**: Each pattern includes test examples
5. **Async Patterns**: Proper async/await usage where applicable