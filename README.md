#  AI Model Router

Smart routing for AI models with automatic cost optimization, fallbacks, and performance tracking.

Stop manually managing multiple AI providers - let the router choose the best model for each task based on your constraints.

##  Features

- **Smart Task Classification** - Automatically detects coding, creative, analysis, and QA tasks
- **Cost Optimization** - Routes to the most cost-effective model that meets your quality requirements  
- **Automatic Fallbacks** - Seamless failover when models are down or overloaded
- **Circuit Breaker** - Temporarily disables failing models to prevent cascading failures
- **Usage Analytics** - Track costs, latency, and success rates across all your AI calls
- **Multi-Provider Support** - Works with OpenAI, Anthropic, Groq, and extensible to other providers

## Quick Start

### Installation

```bash
pip install ai-model-router
```

### Set up API Keys

```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export GROQ_API_KEY="your-groq-key"
```

### Basic Usage

```python
import asyncio
from ai_router import AIModelRouter

async def main():
    router = AIModelRouter()
    
    # Simple routing - automatically selects best model
    result = await router.route("Write a Python function to reverse a string")
    
    print(f"Model used: {result['model_used']}")
    print(f"Cost: ${result['cost']:.4f}")
    print(f"Response: {result['response']}")

asyncio.run(main())
```

### Custom Policies

```python
from ai_router import AIModelRouter, TaskType, RoutingPolicy

router = AIModelRouter()

# Custom policy for coding tasks
coding_policy = RoutingPolicy(
    task_type=TaskType.CODING,
    preferred_models=["claude-3.5-sonnet", "gpt-4o"],
    max_cost_per_request=0.50,
    max_latency=5.0,
    fallback_models=["gpt-4o-mini"]
)

router.add_policy(TaskType.CODING, coding_policy)
```

### Usage Analytics

```python
# Get detailed stats
stats = router.get_stats()
print(f"Total cost: ${stats['total_cost']}")
print(f"Average latency: {stats['avg_latency']}s")
print(f"Success rate: {stats['success_rate']}%")
print(f"Model distribution: {stats['model_usage']}")
```

## Supported Models

| Model | Provider | Cost/1K tokens | Best For |
|-------|----------|----------------|----------|
| GPT-4o | OpenAI | $0.005 | Complex reasoning, coding |
| GPT-4o-mini | OpenAI | $0.00015 | Simple QA, fast responses |
| Claude 3.5 Sonnet | Anthropic | $0.003 | Coding, creative writing, analysis |  
| Claude 3 Haiku | Anthropic | $0.00025 | Quick tasks, cost-sensitive |
| Llama 3.3 70B | Groq | $0.00059 | Fast coding, analysis, complex reasoning |
| Llama 3.1 8B | Groq | $0.00005 | Ultra-fast simple tasks, coding |
| Mixtral 8x7B | Groq | $0.00024 | Balanced performance/cost, creative tasks |

## Task Classification

The router automatically classifies prompts into:

- **CODING** - Programming, debugging, technical implementation
- **CREATIVE** - Writing, storytelling, creative content
- **ANALYSIS** - Data analysis, research, comparison tasks  
- **COMPLEX_REASONING** - Multi-step problems, detailed explanations
- **SIMPLE_QA** - Quick questions, factual lookups

## ⚙️ Configuration

### Environment Variables

```bash
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
GROQ_API_KEY=your-groq-key
AI_ROUTER_LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR
AI_ROUTER_MAX_RETRIES=3
AI_ROUTER_TIMEOUT=30
```

### Custom Model Registry

```python
from ai_router import ModelConfig, TaskType

# Add a custom model
custom_model = ModelConfig(
    name="my-local-model",
    provider="ollama", 
    cost_per_1k_tokens=0.0,  # Free local model
    max_tokens=4096,
    strengths=[TaskType.CODING],
    avg_latency=1.0
)

router.registry.models["my-local-model"] = custom_model
```

## Advanced Usage

### Batch Processing

```python
prompts = [
    "Explain quantum computing",
    "Write a sorting algorithm", 
    "Create a marketing email"
]

results = await asyncio.gather(*[
    router.route(prompt) for prompt in prompts
])

total_cost = sum(r['cost'] for r in results)
print(f"Batch cost: ${total_cost:.4f}")
```

### Custom Task Classification

```python
class CustomClassifier:
    def classify(self, prompt: str) -> TaskType:
        # Your custom logic here
        if "sql" in prompt.lower():
            return TaskType.CODING
        return TaskType.SIMPLE_QA

router.classifier = CustomClassifier()
```

### Monitoring and Alerts

```python
# Set up cost alerts
async def monitor_costs():
    stats = router.get_stats()
    if stats['total_cost'] > 10.0:  # $10 threshold
        send_alert(f"AI costs exceeded $10: ${stats['total_cost']}")

# Run monitoring in background
asyncio.create_task(monitor_costs())
```

##  Development

### Local Development

```bash
git clone https://github.com/ashishjsharda/ai-model-router.git
cd ai-model-router
pip install -r requirements.txt

# Run tests (when available)
pytest


```

### Adding New Providers

```python
class CustomProvider:
    async def call_model(self, model_name: str, prompt: str):
        # Implement your provider logic
        pass

# Register with router
router.providers["custom"] = CustomProvider()
```

## 📋 Roadmap

- [ ] **Local Model Support** - Ollama, HuggingFace integration
- [ ] **Streaming Responses** - Real-time token streaming  
- [ ] **Caching Layer** - Redis-based response caching
- [ ] **Load Balancing** - Distribute across multiple API keys
- [ ] **Web Dashboard** - Real-time monitoring UI
- [ ] **Prompt Templates** - Reusable prompt management
- [ ] **A/B Testing** - Compare model performance
- [ ] **Rate Limiting** - Per-model request limiting
- [ ] **Webhook Integration** - Event-driven monitoring

## Contributing

Contributions welcome! 

### Priority Areas
- Local model providers (Ollama, vLLM)
- Additional cloud providers (Cohere, AI21, etc.)
- Better task classification algorithms
- Performance optimizations
- Documentation improvements

## License

Apache License - 

##  Acknowledgments

Built with inspiration from the amazing AI community. Special thanks to:
- OpenAI for GPT models
- Anthropic for Claude  
- The open source AI ecosystem

---

**Star ⭐ this repo if it saves you time and money on AI costs!**

Made with ❤️ by Ashish Sharda
