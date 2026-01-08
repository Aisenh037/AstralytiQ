#!/usr/bin/env python3
"""
🚀 AstralytiQ Deployment Helper
Automated deployment preparation and validation script.
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

class DeploymentHelper:
    """Helper class for deployment preparation."""
    
    def __init__(self):
        self.project_root = Path.cwd()
        self.required_files = [
            "streamlit_app.py",
            "app.py", 
            "requirements.txt",
            ".streamlit/config.toml",
            "README.md"
        ]
    
    def check_prerequisites(self) -> Tuple[bool, List[str]]:
        """Check if all required files exist."""
        missing_files = []
        
        for file_path in self.required_files:
            if not (self.project_root / file_path).exists():
                missing_files.append(file_path)
        
        return len(missing_files) == 0, missing_files
    
    def validate_requirements(self) -> bool:
        """Validate requirements.txt has all necessary packages."""
        req_file = self.project_root / "requirements.txt"
        if not req_file.exists():
            return False
        
        required_packages = [
            "streamlit", "pandas", "plotly", "requests", 
            "networkx", "numpy", "scikit-learn", "websockets"
        ]
        
        try:
            with open(req_file, 'r', encoding='utf-8') as f:
                content = f.read().lower()
        except UnicodeDecodeError:
            try:
                with open(req_file, 'r', encoding='utf-16') as f:
                    content = f.read().lower()
            except UnicodeDecodeError:
                with open(req_file, 'rb') as f:
                    content = f.read().decode('utf-8', errors='ignore').lower()
        
        missing_packages = []
        for pkg in required_packages:
            if pkg not in content:
                missing_packages.append(pkg)
        
        if missing_packages:
            print(f"❌ Missing packages in requirements.txt: {missing_packages}")
            return False
        
        return True
    
    def check_git_status(self) -> Tuple[bool, str]:
        """Check git repository status."""
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"], 
                capture_output=True, 
                text=True,
                cwd=self.project_root
            )
            
            if result.returncode != 0:
                return False, "Not a git repository"
            
            if result.stdout.strip():
                return False, "Uncommitted changes detected"
            
            remote_result = subprocess.run(
                ["git", "remote", "-v"],
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            if not remote_result.stdout.strip():
                return False, "No git remote configured"
            
            return True, "Git repository ready"
            
        except FileNotFoundError:
            return False, "Git not installed"
    
    def run_deployment_check(self):
        """Run complete deployment readiness check."""
        print("🔍 AstralytiQ Deployment Readiness Check")
        print("=" * 45)
        
        # Check prerequisites
        prereq_ok, missing_files = self.check_prerequisites()
        if prereq_ok:
            print("✅ All required files present")
        else:
            print(f"❌ Missing files: {missing_files}")
            return False
        
        # Check requirements
        if self.validate_requirements():
            print("✅ Requirements.txt validated")
        else:
            print("❌ Requirements.txt issues detected")
            return False
        
        # Check git status
        git_ok, git_msg = self.check_git_status()
        if git_ok:
            print(f"✅ Git: {git_msg}")
        else:
            print(f"⚠️  Git: {git_msg}")
        
        print(f"\n📊 Project Summary:")
        print(f"   Name: AstralytiQ Educational MLOps Platform")
        print(f"   Status: ✅ Ready for deployment")
        
        return True
    
    def show_streamlit_instructions(self):
        """Show Streamlit Cloud deployment instructions."""
        print("\n🌟 Streamlit Cloud Deployment (FREE & RECOMMENDED)")
        print("=" * 55)
        print("Your repository is ready! Follow these steps:")
        print()
        print("1. Go to https://share.streamlit.io")
        print("2. Sign in with GitHub")
        print("3. Click 'New app'")
        print("4. Select your repository: sales-forecast-app")
        print("5. Set main file: streamlit_app.py")
        print("6. Click 'Deploy!'")
        print()
        print("✅ Your app will be live at:")
        print("   https://aisenh037-sales-forecast-app-streamlit-app-main-xyz.streamlit.app")
        print()
        print("🎯 This will give you a live demo of your AstralytiQ platform!")


def main():
    """Main deployment helper function."""
    helper = DeploymentHelper()
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "check":
            helper.run_deployment_check()
        elif command == "streamlit":
            helper.show_streamlit_instructions()
        else:
            print(f"Unknown command: {command}")
            print("Available commands: check, streamlit")
    else:
        # Interactive mode
        print("\n🚀 Welcome to AstralytiQ Deployment Helper!")
        print("=" * 50)
        
        if helper.run_deployment_check():
            print("\n🎉 Your repository is deployment-ready!")
            helper.show_streamlit_instructions()
        else:
            print("\n❌ Please fix the issues above before deploying.")


if __name__ == "__main__":
    main()