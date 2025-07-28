#!/usr/bin/env python3
"""
Simple startup script for the Math Document Converter Web App
"""

import sys
import subprocess
import os

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'flask',
        'anthropic', 
        'PIL',
        'docx',
        'lxml'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n📦 Installing dependencies...")
        
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
            print("✅ Dependencies installed successfully!")
        except subprocess.CalledProcessError:
            print("❌ Failed to install dependencies. Please run:")
            print("   pip install -r requirements.txt")
            return False
    
    # Check if anthropic version is compatible
    try:
        import anthropic
        client = anthropic.Anthropic(api_key="test")
        if not hasattr(client, 'messages'):
            print("⚠️  Anthropic client version issue detected. Updating to latest version...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "anthropic"])
                print("✅ Anthropic updated successfully!")
            except subprocess.CalledProcessError:
                print("❌ Failed to update anthropic. Please run:")
                print("   pip install --upgrade anthropic")
                return False
    except Exception as e:
        print(f"⚠️  Error checking anthropic version: {e}")
    
    return True

def check_api_key():
    """Check if Claude API key is available"""
    try:
        from main import CLAUDE_API_KEY
        if not CLAUDE_API_KEY or CLAUDE_API_KEY == "your-api-key-here":
            print("⚠️  Warning: Claude API key not found or not set!")
            print("   Please update the CLAUDE_API_KEY in main.py")
            return False
        return True
    except ImportError:
        print("❌ Error: Could not import main.py")
        return False

def create_directories():
    """Create necessary directories"""
    directories = ['uploads', 'templates']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def main():
    print("🚀 Starting Math Document Converter Web App...")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Check API key
    if not check_api_key():
        print("\n❌ Cannot start without valid API key")
        return
    
    # Create directories
    create_directories()
    
    print("\n✅ All checks passed!")
    print("🌐 Starting web server...")
    print("📱 Open your browser and go to: http://localhost:5000")
    print("⏹️  Press Ctrl+C to stop the server")
    print("=" * 50)
    
    # Start the Flask app
    try:
        from app import app
        app.run(debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")

if __name__ == "__main__":
    main() 