#!/usr/bin/env python3
"""
Installation script for Math Document Converter Web App
"""

import subprocess
import sys
import os

def install_dependencies():
    """Install all required dependencies"""
    print("📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        return False

def upgrade_anthropic():
    """Upgrade anthropic to latest version"""
    print("🔄 Upgrading anthropic to latest version...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "anthropic"])
        print("✅ Anthropic upgraded successfully!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to upgrade anthropic")
        return False

def test_api_client():
    """Test the API client"""
    print("🧪 Testing API client...")
    try:
        from test_api import test_anthropic_client
        return test_anthropic_client()
    except ImportError:
        print("❌ Could not import test_api.py")
        return False

def create_directories():
    """Create necessary directories"""
    directories = ['uploads', 'templates']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def main():
    print("🚀 Math Document Converter Web App - Installation")
    print("=" * 50)
    
    # Step 1: Install dependencies
    if not install_dependencies():
        print("\n❌ Installation failed at step 1")
        return
    
    # Step 2: Upgrade anthropic
    if not upgrade_anthropic():
        print("\n❌ Installation failed at step 2")
        return
    
    # Step 3: Create directories
    create_directories()
    
    # Step 4: Test API client
    if not test_api_client():
        print("\n❌ Installation failed at step 4")
        print("Please check your API key in main.py")
        return
    
    print("\n" + "=" * 50)
    print("✅ Installation completed successfully!")
    print("\n🎉 You can now run the web app:")
    print("   python run.py")
    print("   or")
    print("   python app.py")
    print("\n🌐 The app will be available at: http://localhost:5000")

if __name__ == "__main__":
    main() 