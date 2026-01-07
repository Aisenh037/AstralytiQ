#!/usr/bin/env python3
"""
Local development startup script for Enterprise SaaS Platform
"""
import subprocess
import sys
import time
import requests
import os
from concurrent.futures import ThreadPoolExecutor
import signal

def check_service_health(service_name, url, max_retries=30):
    """Check if a service is healthy."""
    for i in range(max_retries):
        try:
            response = requests.get(f"{url}/health", timeout=5)
            if response.status_code == 200:
                print(f"✅ {service_name} is healthy at {url}")
                return True
        except requests.exceptions.RequestException:
            pass
        
        if i < max_retries - 1:
            print(f"⏳ Waiting for {service_name} to start... ({i+1}/{max_retries})")
            time.sleep(2)
    
    print(f"❌ {service_name} failed to start at {url}")
    return False

def start_service(service_name, port, service_path):
    """Start a microservice."""
    print(f"🚀 Starting {service_name} on port {port}...")
    
    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd()
    
    cmd = [
        sys.executable, "-m", "uvicorn",
        f"src.services.{service_path}.main:app",
        "--host", "0.0.0.0",
        "--port", str(port),
        "--reload"
    ]
    
    return subprocess.Popen(cmd, env=env)

def start_databases():
    """Start required databases using Docker Compose."""
    print("🗄️ Starting databases...")
    
    try:
        # Check if Docker is running
        subprocess.run(["docker", "--version"], check=True, capture_output=True)
        
        # Start databases
        subprocess.run([
            "docker-compose", "up", "-d", 
            "postgres", "mongodb", "redis"
        ], check=True)
        
        print("✅ Databases started successfully")
        return True
        
    except subprocess.CalledProcessError:
        print("❌ Failed to start databases. Make sure Docker is installed and running.")
        return False
    except FileNotFoundError:
        print("❌ Docker not found. Please install Docker and Docker Compose.")
        return False

def run_tests():
    """Run basic tests to verify setup."""
    print("🧪 Running basic tests...")
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            "tests/test_basic_structure.py", "-v"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Basic tests passed")
            return True
        else:
            print("❌ Some tests failed:")
            print(result.stdout)
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Failed to run tests: {e}")
        return False

def main():
    """Main startup function."""
    print("🚀 Enterprise SaaS Platform - Local Development Setup")
    print("=" * 60)
    
    # Store process references for cleanup
    processes = []
    
    def signal_handler(sig, frame):
        print("\n🛑 Shutting down services...")
        for process in processes:
            process.terminate()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        # Step 1: Start databases
        if not start_databases():
            return False
        
        # Wait for databases to be ready
        print("⏳ Waiting for databases to be ready...")
        time.sleep(10)
        
        # Step 2: Run tests
        if not run_tests():
            print("⚠️ Tests failed, but continuing with service startup...")
        
        # Step 3: Start microservices
        services = [
            ("API Gateway", 8000, "api_gateway"),
            ("User Service", 8001, "user_service"),
            ("Tenant Service", 8002, "tenant_service"),
            ("Data Service", 8003, "data_service")
        ]
        
        print("\n🚀 Starting microservices...")
        
        for service_name, port, service_path in services:
            process = start_service(service_name, port, service_path)
            processes.append(process)
            time.sleep(3)  # Stagger startup
        
        # Step 4: Health checks
        print("\n🏥 Performing health checks...")
        all_healthy = True
        
        for service_name, port, _ in services:
            url = f"http://localhost:{port}"
            if not check_service_health(service_name, url):
                all_healthy = False
        
        if all_healthy:
            print("\n🎉 All services are running successfully!")
            print("\n📚 Available Services:")
            print("=" * 40)
            
            for service_name, port, _ in services:
                print(f"• {service_name:15} http://localhost:{port}")
                print(f"  {'Docs:':15} http://localhost:{port}/docs")
            
            print(f"\n🎨 Streamlit Frontend: http://localhost:8501")
            print("   Run: streamlit run streamlit_app.py")
            
            print(f"\n🔧 Development Tips:")
            print("• API Documentation available at /docs endpoint for each service")
            print("• Use Ctrl+C to stop all services")
            print("• Check logs in terminal for debugging")
            print("• Modify code and services will auto-reload")
            
            # Start Streamlit app
            print(f"\n🎨 Starting Streamlit frontend...")
            streamlit_process = subprocess.Popen([
                sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
                "--server.port", "8501",
                "--server.address", "0.0.0.0"
            ])
            processes.append(streamlit_process)
            
            print("\n✨ Platform is ready for development!")
            print("Press Ctrl+C to stop all services")
            
            # Keep the script running
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass
        
        else:
            print("\n❌ Some services failed to start. Check the logs above.")
            return False
    
    except Exception as e:
        print(f"\n❌ Startup failed: {e}")
        return False
    
    finally:
        # Cleanup
        print("\n🧹 Cleaning up...")
        for process in processes:
            try:
                process.terminate()
                process.wait(timeout=5)
            except:
                process.kill()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)