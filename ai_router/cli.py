#!/usr/bin/env python3
"""CLI interface for AI Model Router"""

import asyncio
import json
import os
import sys
from .router import AIModelRouter

def main():
    """CLI entry point"""
    # Check API keys
    if not any([os.getenv('OPENAI_API_KEY'), os.getenv('ANTHROPIC_API_KEY'), os.getenv('GROQ_API_KEY')]):
        print("⚠️  Please set at least one API key:")
        print("   export OPENAI_API_KEY=your-openai-key")
        print("   export ANTHROPIC_API_KEY=your-anthropic-key") 
        print("   export GROQ_API_KEY=your-groq-key")
        sys.exit(1)
    
    asyncio.run(interactive_cli())

async def interactive_cli():
    """Interactive CLI loop"""
    router = AIModelRouter()
    
    print("🚀 AI Model Router v0.1")
    print("Available commands: route <prompt>, stats, help, quit")
    
    while True:
        try:
            command = input("\n> ").strip()
            
            if command == "quit" or command == "exit":
                print("👋 Goodbye!")
                break
                
            elif command == "stats":
                stats = router.get_stats()
                print(json.dumps(stats, indent=2))
                
            elif command == "help":
                print_help()
                
            elif command.startswith("route "):
                prompt = command[6:]  # Remove "route "
                if not prompt:
                    print("Usage: route <your prompt here>")
                    continue
                    
                print("🤔 Processing...")
                result = await router.route(prompt)
                
                if result["success"]:
                    print(f"\n✅ Model: {result['model_used']}")
                    print(f"💰 Cost: ${result['cost']:.4f} | ⚡ Latency: {result['latency']:.2f}s")
                    print(f"🎯 Task: {result['task_type']}")
                    if result.get('fallback_used'):
                        print("⚠️  Used fallback model")
                    print(f"\n📝 Response:\n{result['response']}")
                else:
                    print(f"❌ Error: {result.get('error', 'Unknown error')}")
                    
            elif command == "":
                continue
                
            else:
                print("Unknown command. Type 'help' for available commands.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

def print_help():
    """Print help information"""
    help_text = """
🚀 AI Model Router Commands:

route <prompt>    - Route a prompt to the best AI model
                   Example: route Write a Python function to sort a list

stats            - Show usage statistics (cost, latency, model usage)

help             - Show this help message

quit/exit        - Exit the router

🎯 The router automatically selects the best model based on:
   • Task type (coding, creative, analysis, simple Q&A)
   • Cost constraints
   • Latency requirements
   • Model availability

💡 Tip: Try different types of prompts to see how routing changes!
"""
    print(help_text)

if __name__ == "__main__":
    main()