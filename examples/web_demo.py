#!/usr/bin/env python3
"""Launch the AI Router Web UI demo"""

import os
import sys

def main():
    print("🚀 AI Router Web Demo")
    print("=" * 50)
    
    # Check for API keys
    api_keys = {
        'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
        'ANTHROPIC_API_KEY': os.getenv('ANTHROPIC_API_KEY'), 
        'GROQ_API_KEY': os.getenv('GROQ_API_KEY')
    }
    
    available_keys = [name for name, key in api_keys.items() if key]
    
    if not available_keys:
        print("❌ No API keys found!")
        print("\nPlease set at least one API key:")
        print("  Windows PowerShell:")
        print("    $env:OPENAI_API_KEY='your-key'")
        print("    $env:ANTHROPIC_API_KEY='your-key'")
        print("    $env:GROQ_API_KEY='your-key'")
        print("\n  Windows Command Prompt:")
        print("    set OPENAI_API_KEY=your-key")
        print("    set ANTHROPIC_API_KEY=your-key")
        print("    set GROQ_API_KEY=your-key")
        print("\n  Linux/Mac:")
        print("    export OPENAI_API_KEY=your-key")
        print("    export ANTHROPIC_API_KEY=your-key")
        print("    export GROQ_API_KEY=your-key")
        sys.exit(1)
    
    print(f"✅ Found API keys for: {', '.join(available_keys)}")
    print()
    
    try:
        # Import and start the web server
        from ai_router.web.server import start_web_ui
        
        print("🌐 Starting web interface...")
        print("📝 You can test different prompts and see:")
        print("  • Smart model selection")
        print("  • Real-time cost tracking") 
        print("  • Task classification")
        print("  • Performance metrics")
        print()
        
        # Start the server (will open browser automatically)
        start_web_ui(port=8080, open_browser=True)
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure ai_router is installed:")
        print("  pip install -e .")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()