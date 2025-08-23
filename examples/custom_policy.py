#!/usr/bin/env python3
"""Custom policy example for AI Model Router"""

import asyncio
import os
from ai_router import AIModelRouter, TaskType, RoutingPolicy

async def main():
    router = AIModelRouter()
    
    print("🚀 AI Model Router - Custom Policy Example\n")
    
    # Create a budget-conscious policy for coding tasks
    budget_coding_policy = RoutingPolicy(
        task_type=TaskType.CODING,
        preferred_models=["llama-3.1-8b", "mixtral-8x7b"],  # Cheapest first
        max_cost_per_request=0.10,  # Very low budget
        max_latency=5.0,
        fallback_models=["gpt-4o-mini"]
    )
    
    # Create a premium policy for analysis tasks
    premium_analysis_policy = RoutingPolicy(
        task_type=TaskType.ANALYSIS,
        preferred_models=["gpt-4o", "claude-3.5-sonnet"],  # Best quality first
        max_cost_per_request=2.00,  # Higher budget for quality
        max_latency=10.0,
        fallback_models=["llama-3.1-70b"]
    )
    
    # Apply custom policies
    router.add_policy(TaskType.CODING, budget_coding_policy)
    router.add_policy(TaskType.ANALYSIS, premium_analysis_policy)
    
    print("📋 Applied custom policies:")
    print("   • Coding: Budget-focused (prefer cheap, fast models)")
    print("   • Analysis: Premium-focused (prefer high-quality models)")
    print()
    
    # Test with different prompts
    test_cases = [
        "Write a Python function to merge two sorted lists",  # Should use cheap model
        "Analyze the economic impact of remote work on urban centers",  # Should use premium model
        "What's 2+2?",  # Simple QA - should use default policy
    ]
    
    for i, prompt in enumerate(test_cases, 1):
        print(f"Test {i}: {prompt}")
        
        result = await router.route(prompt)
        
        if result["success"]:
            print(f"✅ Selected: {result['model_used']}")
            print(f"💰 Cost: ${result['cost']:.4f}")
            print(f"🎯 Task type: {result['task_type']}")
            
            # Show why this model was chosen
            if result['task_type'] == 'coding':
                print("💡 Used budget policy - prioritized cost savings")
            elif result['task_type'] == 'analysis':
                print("💡 Used premium policy - prioritized quality")
            else:
                print("💡 Used default policy")
        else:
            print(f"❌ Failed: {result.get('error')}")
        
        print("-" * 50)
    
    # Show final stats
    stats = router.get_stats()
    print(f"\n📊 Custom Policy Results:")
    print(f"💰 Total cost: ${stats['total_cost']:.4f}")
    print(f"🤖 Models used: {stats['model_usage']}")
    print(f"🎯 Task distribution: {stats['task_distribution']}")

if __name__ == "__main__":
    if not any([os.getenv('OPENAI_API_KEY'), os.getenv('ANTHROPIC_API_KEY'), os.getenv('GROQ_API_KEY')]):
        print("Please set at least one API key:")
        print("export OPENAI_API_KEY=your-key")
        print("export ANTHROPIC_API_KEY=your-key") 
        print("export GROQ_API_KEY=your-key")
        exit(1)
    
    asyncio.run(main())