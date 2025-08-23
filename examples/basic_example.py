#!/usr/bin/env python3
"""Basic usage example for AI Model Router"""

import asyncio
import os
from ai_router import AIModelRouter

async def main():
    # Initialize the router
    router = AIModelRouter()
    
    print("🚀 AI Model Router - Basic Example\n")
    
    # Example prompts of different types
    test_prompts = [
        ("Simple QA", "What's the capital of Japan?"),
        ("Coding", "Write a Python function to calculate fibonacci numbers"),
        ("Creative", "Write a short poem about artificial intelligence"),
        ("Analysis", "Compare the pros and cons of renewable energy sources"),
    ]
    
    total_cost = 0
    
    for category, prompt in test_prompts:
        print(f"📝 {category}: {prompt}")
        
        # Route the request
        result = await router.route(prompt)
        
        if result["success"]:
            print(f"✅ Model: {result['model_used']}")
            print(f"💰 Cost: ${result['cost']:.4f}")
            print(f"⚡ Latency: {result['latency']:.2f}s")
            print(f"🎯 Classified as: {result['task_type']}")
            
            # Show a preview of the response
            response_preview = result['response'][:100] + "..." if len(result['response']) > 100 else result['response']
            print(f"📖 Response preview: {response_preview}")
            
            total_cost += result['cost']
        else:
            print(f"❌ Failed: {result.get('error')}")
        
        print("-" * 60)
    
    # Show final statistics
    print(f"\n📊 Session Summary:")
    print(f"💰 Total cost: ${total_cost:.4f}")
    
    stats = router.get_stats()
    print(f"📈 Total requests: {stats['total_requests']}")
    print(f"⚡ Average latency: {stats['avg_latency']:.2f}s")
    print(f"✅ Success rate: {stats['success_rate']:.1f}%")
    print(f"🤖 Models used: {list(stats['model_usage'].keys())}")

if __name__ == "__main__":
    # Check for API keys
    if not any([os.getenv('OPENAI_API_KEY'), os.getenv('ANTHROPIC_API_KEY'), os.getenv('GROQ_API_KEY')]):
        print("Please set at least one API key:")
        print("export OPENAI_API_KEY=your-key")
        print("export ANTHROPIC_API_KEY=your-key") 
        print("export GROQ_API_KEY=your-key")
        exit(1)
    
    asyncio.run(main())