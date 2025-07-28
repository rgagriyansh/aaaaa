#!/usr/bin/env python3
"""
Simple test script to verify Anthropic API client functionality
"""

import anthropic
import sys

def test_anthropic_client():
    """Test if the Anthropic client can be initialized and used"""
    try:
        # Import the API key from main.py
        from main import CLAUDE_API_KEY
        
        if not CLAUDE_API_KEY or CLAUDE_API_KEY == "your-api-key-here":
            print("❌ API key not set in main.py")
            return False
        
        print("✅ API key found")
        
        # Test client initialization
        client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
        print("✅ Anthropic client initialized successfully")
        
        # Test if messages attribute exists
        if hasattr(client, 'messages'):
            print("✅ Client has 'messages' attribute")
        else:
            print("❌ Client does not have 'messages' attribute")
            return False
        
        # Test a simple API call
        print("🧪 Testing API call...")
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=50,
            messages=[
                {"role": "user", "content": "Hello! Please respond with just 'Test successful'."}
            ]
        )
        
        print(f"✅ API call successful: {response.content[0].text}")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Anthropic API Client...")
    print("=" * 40)
    
    success = test_anthropic_client()
    
    if success:
        print("\n✅ All tests passed! The API client is working correctly.")
        sys.exit(0)
    else:
        print("\n❌ Tests failed! Please check the error messages above.")
        sys.exit(1) 