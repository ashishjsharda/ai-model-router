"""AI Model Router - Smart routing for AI models with cost optimization"""

from .router import AIModelRouter, TaskType, RoutingPolicy, ModelConfig, RequestMetrics

__version__ = "0.1.0"
__author__ = "AI Router Team"
__description__ = "Smart routing for AI models with automatic cost optimization and fallbacks"

__all__ = [
    "AIModelRouter",
    "TaskType", 
    "RoutingPolicy",
    "ModelConfig",
    "RequestMetrics"
]