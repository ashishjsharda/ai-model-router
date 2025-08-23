#!/usr/bin/env python3
"""Test script to verify Groq model names are working"""

import asyncio
import os
from ai_router import AIModelRouter

async def test_groq_models():
    """Test that Groq models are accessible"""
    
    # Check if GROQ_API_KEY is set
    if not os.getenv('GROQ_API_KEY'):
        print("❌ GROQ_API_KEY environment variable is not set!")
        print("Please set it using:")
        print('$env:GROQ_API_KEY = "your-actual-groq-api-key"')
        return
    
    print("✅ GROQ_API_KEY is set")
    
    try:
        # Initialize the router
        router = AIModelRouter()
        print("✅ Router initialized successfully")
        
        # Test a simple prompt that should use a Groq model
        print("\n🧪 Testing simple QA (should use llama-3.1-8b)...")
        result = await router.route("What is 2+2?")
        
        if result["success"]:
            print(f"✅ Success! Used model: {result['model_used']}")
            print(f"📖 Response: {result['response']}")
            print(f"💰 Cost: ${result['cost']:.4f}")
            print(f"⚡ Latency: {result['latency']:.2f}s")
        else:
            print(f"❌ Failed: {result.get('error')}")
            
    except Exception as e:
        print(f"❌ Error initializing router: {e}")

if __name__ == "__main__":
    asyncio.run(test_groq_models())