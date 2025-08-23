#!/usr/bin/env python3
"""
AI Model Router - Smart routing for AI models
Usage: pip install openai anthropic groq
"""

import asyncio
import time
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TaskType(Enum):
    CODING = "coding"
    CREATIVE = "creative"
    ANALYSIS = "analysis"
    SIMPLE_QA = "simple_qa"
    COMPLEX_REASONING = "complex_reasoning"

@dataclass
class ModelConfig:
    name: str
    provider: str
    cost_per_1k_tokens: float
    max_tokens: int
    strengths: List[TaskType]
    avg_latency: float = 2.0

@dataclass
class RoutingPolicy:
    task_type: TaskType
    preferred_models: List[str]
    max_cost_per_request: float = 1.0
    max_latency: float = 10.0
    fallback_models: List[str] = None

@dataclass
class RequestMetrics:
    model_used: str
    latency: float
    tokens_used: int
    cost: float
    timestamp: datetime
    success: bool
    task_type: TaskType

class ModelRegistry:
    def __init__(self):
        self.models = {
            "gpt-4o": ModelConfig(
                name="gpt-4o",
                provider="openai",
                cost_per_1k_tokens=0.005,
                max_tokens=4096,
                strengths=[TaskType.CODING, TaskType.ANALYSIS, TaskType.COMPLEX_REASONING],
                avg_latency=3.0
            ),
            "gpt-4o-mini": ModelConfig(
                name="gpt-4o-mini", 
                provider="openai",
                cost_per_1k_tokens=0.00015,
                max_tokens=4096,
                strengths=[TaskType.SIMPLE_QA, TaskType.CREATIVE],
                avg_latency=1.5
            ),
            "claude-3.5-sonnet": ModelConfig(
                name="claude-3-5-sonnet-20241022",
                provider="anthropic",
                cost_per_1k_tokens=0.003,
                max_tokens=8192,
                strengths=[TaskType.CODING, TaskType.CREATIVE, TaskType.ANALYSIS],
                avg_latency=2.5
            ),
            "claude-3-haiku": ModelConfig(
                name="claude-3-haiku-20240307",
                provider="anthropic", 
                cost_per_1k_tokens=0.00025,
                max_tokens=4096,
                strengths=[TaskType.SIMPLE_QA, TaskType.CREATIVE],
                avg_latency=1.0
            ),
            "llama-3.3-70b": ModelConfig(
                name="llama-3.3-70b-versatile",
                provider="groq",
                cost_per_1k_tokens=0.00059,
                max_tokens=8192,
                strengths=[TaskType.CODING, TaskType.ANALYSIS, TaskType.COMPLEX_REASONING],
                avg_latency=0.5
            ),
            "llama-3.1-8b": ModelConfig(
                name="llama-3.1-8b-instant",
                provider="groq",
                cost_per_1k_tokens=0.00005,
                max_tokens=8192,
                strengths=[TaskType.SIMPLE_QA, TaskType.CODING],
                avg_latency=0.3
            ),
            # Add mixtral back for better fallback options
            "mixtral-8x7b": ModelConfig(
                name="mixtral-8x7b-32768",
                provider="groq", 
                cost_per_1k_tokens=0.00024,
                max_tokens=32768,
                strengths=[TaskType.CODING, TaskType.ANALYSIS, TaskType.CREATIVE],
                avg_latency=0.4
            )
        }
        
    def get_model(self, name: str) -> Optional[ModelConfig]:
        return self.models.get(name)
    
    def get_models_by_strength(self, task_type: TaskType) -> List[ModelConfig]:
        return [model for model in self.models.values() if task_type in model.strengths]

class TaskClassifier:
    """Simple rule-based task classifier"""
    
    def classify(self, prompt: str) -> TaskType:
        prompt_lower = prompt.lower()
        
        # Coding keywords
        coding_keywords = ['code', 'python', 'javascript', 'function', 'debug', 'programming', 'algorithm', 'script', 'class', 'method']
        if any(keyword in prompt_lower for keyword in coding_keywords):
            return TaskType.CODING
            
        # Creative keywords
        creative_keywords = ['story', 'poem', 'creative', 'write', 'imagine', 'character', 'haiku', 'novel', 'creative']
        if any(keyword in prompt_lower for keyword in creative_keywords):
            return TaskType.CREATIVE
            
        # Analysis keywords
        analysis_keywords = ['analyze', 'compare', 'evaluate', 'assessment', 'research', 'pros and cons', 'advantages', 'disadvantages']
        if any(keyword in prompt_lower for keyword in analysis_keywords):
            return TaskType.ANALYSIS
            
        # Complex reasoning
        reasoning_keywords = ['explain', 'why', 'how', 'complex', 'detailed', 'elaborate', 'discuss']
        if any(keyword in prompt_lower for keyword in reasoning_keywords) and len(prompt) > 100:
            return TaskType.COMPLEX_REASONING
            
        # Default to simple QA
        return TaskType.SIMPLE_QA

class AIModelRouter:
    def __init__(self):
        self.registry = ModelRegistry()
        self.classifier = TaskClassifier()
        self.policies: Dict[TaskType, RoutingPolicy] = {}
        self.metrics: List[RequestMetrics] = []
        self.failed_models = set()  # Circuit breaker
        
        # Initialize API clients only if keys are available
        self.openai_client = None
        self.anthropic_client = None
        self.groq_client = None
        
        # OpenAI client
        if os.getenv('OPENAI_API_KEY'):
            try:
                import openai
                self.openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
                logger.info("OpenAI client initialized")
            except ImportError:
                logger.warning("OpenAI not installed. Install with: pip install openai")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI client: {e}")
        
        # Anthropic client
        if os.getenv('ANTHROPIC_API_KEY'):
            try:
                import anthropic
                self.anthropic_client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
                logger.info("Anthropic client initialized")
            except ImportError:
                logger.warning("Anthropic not installed. Install with: pip install anthropic")
            except Exception as e:
                logger.warning(f"Failed to initialize Anthropic client: {e}")
        
        # Groq client
        if os.getenv('GROQ_API_KEY'):
            try:
                import groq
                self.groq_client = groq.Groq(api_key=os.getenv('GROQ_API_KEY'))
                logger.info("Groq client initialized")
            except ImportError:
                logger.warning("Groq not installed. Install with: pip install groq")
            except Exception as e:
                logger.warning(f"Failed to initialize Groq client: {e}")
        
        # Check that at least one client is available
        if not any([self.openai_client, self.anthropic_client, self.groq_client]):
            available_providers = []
            if os.getenv('OPENAI_API_KEY'): available_providers.append("OpenAI")
            if os.getenv('ANTHROPIC_API_KEY'): available_providers.append("Anthropic") 
            if os.getenv('GROQ_API_KEY'): available_providers.append("Groq")
            
            if not available_providers:
                raise Exception(
                    "No API keys found. Please set at least one:\n"
                    "  - OPENAI_API_KEY\n"
                    "  - ANTHROPIC_API_KEY\n" 
                    "  - GROQ_API_KEY"
                )
            else:
                raise Exception(f"Failed to initialize any API clients despite having keys for: {', '.join(available_providers)}")
        
        # Set default policies
        self._set_default_policies()
    
    def _set_default_policies(self):
        """Set sensible default routing policies"""
        self.policies[TaskType.CODING] = RoutingPolicy(
            TaskType.CODING,
            preferred_models=["claude-3.5-sonnet", "llama-3.3-70b", "gpt-4o"],
            max_cost_per_request=0.50,
            fallback_models=["mixtral-8x7b", "llama-3.1-8b", "gpt-4o-mini"]
        )
        
        self.policies[TaskType.SIMPLE_QA] = RoutingPolicy(
            TaskType.SIMPLE_QA,
            preferred_models=["llama-3.1-8b", "gpt-4o-mini", "claude-3-haiku"],
            max_cost_per_request=0.05,
            fallback_models=["mixtral-8x7b"]
        )
        
        self.policies[TaskType.CREATIVE] = RoutingPolicy(
            TaskType.CREATIVE,
            preferred_models=["claude-3.5-sonnet", "mixtral-8x7b", "gpt-4o"],
            max_cost_per_request=0.30,
            fallback_models=["llama-3.1-8b", "claude-3-haiku"]
        )
        
        self.policies[TaskType.ANALYSIS] = RoutingPolicy(
            TaskType.ANALYSIS,
            preferred_models=["claude-3.5-sonnet", "llama-3.3-70b", "gpt-4o"],
            max_cost_per_request=0.75,
            fallback_models=["mixtral-8x7b", "llama-3.1-8b"]
        )
        
        self.policies[TaskType.COMPLEX_REASONING] = RoutingPolicy(
            TaskType.COMPLEX_REASONING,
            preferred_models=["gpt-4o", "claude-3.5-sonnet", "llama-3.3-70b"],
            max_cost_per_request=1.00,
            fallback_models=["mixtral-8x7b", "llama-3.1-8b"]
        )
    
    def add_policy(self, task_type: TaskType, policy: RoutingPolicy):
        """Add or update a routing policy"""
        self.policies[task_type] = policy
        
    def _select_model(self, task_type: TaskType) -> str:
        """Select the best model for a task based on policy and available providers"""
        policy = self.policies.get(task_type)
        if not policy:
            # Find any available model as fallback
            for model_name, model_config in self.registry.models.items():
                if self._is_provider_available(model_config.provider):
                    return model_name
            raise Exception("No available models found")
            
        # Try preferred models first (only if provider is available)
        for model_name in policy.preferred_models:
            if model_name not in self.failed_models:
                model = self.registry.get_model(model_name)
                if model and self._is_provider_available(model.provider) and model.avg_latency <= policy.max_latency:
                    return model_name
                    
        # Try fallback models (only if provider is available)
        if policy.fallback_models:
            for model_name in policy.fallback_models:
                if model_name not in self.failed_models:
                    model = self.registry.get_model(model_name)
                    if model and self._is_provider_available(model.provider):
                        return model_name
                    
        # Last resort - find any available model
        for model_name, model_config in self.registry.models.items():
            if model_name not in self.failed_models and self._is_provider_available(model_config.provider):
                return model_name
                
        raise Exception("No available models found")
    
    def _is_provider_available(self, provider: str) -> bool:
        """Check if a provider is available (has initialized client)"""
        if provider == "openai":
            return self.openai_client is not None
        elif provider == "anthropic":
            return self.anthropic_client is not None
        elif provider == "groq":
            return self.groq_client is not None
        return False
    
    async def _call_openai(self, model_name: str, prompt: str) -> tuple[str, int]:
        """Call OpenAI API"""
        if not self.openai_client:
            raise Exception("OpenAI client not initialized")
            
        response = await asyncio.to_thread(
            self.openai_client.chat.completions.create,
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000
        )
        
        return response.choices[0].message.content, response.usage.total_tokens
    
    async def _call_anthropic(self, model_name: str, prompt: str) -> tuple[str, int]:
        """Call Anthropic API"""
        if not self.anthropic_client:
            raise Exception("Anthropic client not initialized")
            
        response = await asyncio.to_thread(
            self.anthropic_client.messages.create,
            model=model_name,
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Rough token estimation for Anthropic
        estimated_tokens = len(prompt.split()) * 1.3 + len(response.content[0].text.split()) * 1.3
        
        return response.content[0].text, int(estimated_tokens)
    
    async def _call_groq(self, model_name: str, prompt: str) -> tuple[str, int]:
        """Call Groq API"""
        if not self.groq_client:
            raise Exception("Groq client not initialized")
            
        response = await asyncio.to_thread(
            self.groq_client.chat.completions.create,
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.7
        )
        
        return response.choices[0].message.content, response.usage.total_tokens
    
    async def route(self, prompt: str, task_type: Optional[TaskType] = None) -> Dict[str, Any]:
        """Route a request to the best model"""
        start_time = time.time()
        
        # Handle string task_type from web UI
        if isinstance(task_type, str):
            try:
                task_type = TaskType(task_type)
            except ValueError:
                task_type = None
        
        # Classify task if not provided
        if not task_type:
            task_type = self.classifier.classify(prompt)
            
        logger.info(f"Classified task as: {task_type.value}")
        
        # Select model
        selected_model = self._select_model(task_type)
        model_config = self.registry.get_model(selected_model)
        
        logger.info(f"Selected model: {selected_model}")
        
        try:
            # Make API call using the actual model name (not registry key)
            if model_config.provider == "openai":
                response_text, tokens_used = await self._call_openai(model_config.name, prompt)
            elif model_config.provider == "anthropic":
                response_text, tokens_used = await self._call_anthropic(model_config.name, prompt)
            elif model_config.provider == "groq":
                response_text, tokens_used = await self._call_groq(model_config.name, prompt)
            else:
                raise Exception(f"Unknown provider: {model_config.provider}")
            
            # Calculate metrics
            latency = time.time() - start_time
            cost = (tokens_used / 1000) * model_config.cost_per_1k_tokens
            
            # Record metrics
            metrics = RequestMetrics(
                model_used=selected_model,
                latency=latency,
                tokens_used=tokens_used,
                cost=cost,
                timestamp=datetime.now(),
                success=True,
                task_type=task_type
            )
            self.metrics.append(metrics)
            
            # Remove from failed models if it was there
            self.failed_models.discard(selected_model)
            
            return {
                "response": response_text,
                "model_used": selected_model,
                "provider": model_config.provider,  # Add provider info for UI
                "task_type": task_type.value,
                "latency": latency,
                "tokens_used": tokens_used,
                "cost": cost,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Model {selected_model} failed: {str(e)}")
            
            # Add to failed models (circuit breaker)
            self.failed_models.add(selected_model)
            
            # Try fallback
            policy = self.policies.get(task_type)
            if policy and policy.fallback_models:
                for fallback_model in policy.fallback_models:
                    if fallback_model not in self.failed_models:
                        try:
                            fallback_config = self.registry.get_model(fallback_model)
                            if not self._is_provider_available(fallback_config.provider):
                                continue
                                
                            if fallback_config.provider == "openai":
                                response_text, tokens_used = await self._call_openai(fallback_config.name, prompt)
                            elif fallback_config.provider == "anthropic":
                                response_text, tokens_used = await self._call_anthropic(fallback_config.name, prompt)
                            elif fallback_config.provider == "groq":
                                response_text, tokens_used = await self._call_groq(fallback_config.name, prompt)
                            else:
                                raise Exception(f"Unknown provider: {fallback_config.provider}")
                                
                            latency = time.time() - start_time
                            cost = (tokens_used / 1000) * fallback_config.cost_per_1k_tokens
                            
                            metrics = RequestMetrics(
                                model_used=fallback_model,
                                latency=latency,
                                tokens_used=tokens_used,
                                cost=cost,
                                timestamp=datetime.now(),
                                success=True,
                                task_type=task_type
                            )
                            self.metrics.append(metrics)
                            
                            return {
                                "response": response_text,
                                "model_used": fallback_model,
                                "provider": fallback_config.provider,
                                "task_type": task_type.value,
                                "latency": latency,
                                "tokens_used": tokens_used,
                                "cost": cost,
                                "success": True,
                                "fallback_used": True
                            }
                            
                        except Exception as fallback_error:
                            logger.error(f"Fallback model {fallback_model} also failed: {str(fallback_error)}")
                            continue
            
            # All models failed
            return {
                "response": None,
                "error": str(e),
                "model_used": selected_model,
                "success": False
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics"""
        if not self.metrics:
            return {
                "total_requests": 0,
                "total_cost": 0,
                "avg_latency": 0,
                "success_rate": 100
            }
            
        total_cost = sum(m.cost for m in self.metrics)
        total_requests = len(self.metrics)
        successful_metrics = [m for m in self.metrics if m.success]
        
        if successful_metrics:
            avg_latency = sum(m.latency for m in successful_metrics) / len(successful_metrics)
        else:
            avg_latency = 0
            
        success_rate = len(successful_metrics) / total_requests if total_requests > 0 else 1
        
        model_usage = {}
        for metric in self.metrics:
            model_usage[metric.model_used] = model_usage.get(metric.model_used, 0) + 1
            
        task_distribution = {}
        for metric in self.metrics:
            task_type = metric.task_type.value
            task_distribution[task_type] = task_distribution.get(task_type, 0) + 1
        
        return {
            "total_requests": total_requests,
            "total_cost": round(total_cost, 4),
            "avg_latency": round(avg_latency, 2),
            "success_rate": round(success_rate * 100, 2),
            "model_usage": model_usage,
            "task_distribution": task_distribution,
            "failed_models": list(self.failed_models)
        }

# CLI Interface
async def main():
    """Simple CLI for testing"""
    router = AIModelRouter()
    
    print("🚀 AI Model Router v0.1")
    print("Available commands: route <prompt>, stats, quit")
    
    while True:
        try:
            command = input("\n> ").strip()
            
            if command == "quit":
                break
            elif command == "stats":
                stats = router.get_stats()
                print(json.dumps(stats, indent=2))
            elif command.startswith("route "):
                prompt = command[6:]  # Remove "route "
                result = await router.route(prompt)
                
                if result["success"]:
                    print(f"\n✅ Model: {result['model_used']}")
                    print(f"📊 Cost: ${result['cost']:.4f} | Latency: {result['latency']:.2f}s")
                    print(f"🎯 Task: {result['task_type']}")
                    if result.get('fallback_used'):
                        print("⚠️  Used fallback model")
                    print(f"\n📝 Response:\n{result['response']}")
                else:
                    print(f"❌ Error: {result.get('error', 'Unknown error')}")
            else:
                print("Commands: route <prompt>, stats, quit")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    # Set up API keys
    if not any([os.getenv('OPENAI_API_KEY'), os.getenv('ANTHROPIC_API_KEY'), os.getenv('GROQ_API_KEY')]):
        print("⚠️  Please set at least one API key:")
        print("   export OPENAI_API_KEY=your-openai-key")
        print("   export ANTHROPIC_API_KEY=your-anthropic-key") 
        print("   export GROQ_API_KEY=your-groq-key")
    
    asyncio.run(main())