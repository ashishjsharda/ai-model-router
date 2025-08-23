#!/usr/bin/env python3
"""Check what models are actually available in Groq API"""

import requests
import os
import json

def check_groq_models():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("❌ GROQ_API_KEY not set")
        return
    
    url = "https://api.groq.com/openai/v1/models"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        models = response.json()
        print("✅ Available models in your Groq account:")
        print("=" * 50)
        
        for model in models.get('data', []):
            model_id = model.get('id', 'Unknown')
            print(f"📋 {model_id}")
            
        print("=" * 50)
        print(f"Total models: {len(models.get('data', []))}")
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching models: {e}")
        if hasattr(e, 'response') and e.response:
            print(f"Response: {e.response.text}")

if __name__ == "__main__":
    check_groq_models()