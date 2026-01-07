#!/usr/bin/env python3
"""
Startup script for the Enterprise SaaS Platform services.
"""
import asyncio
import subprocess
import sys
import time
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.shared.infrastructure.migrations import create_tables


async def setup_database():
    """Set up the database tables."""
    print("Setting up database tables...")
    try:
        await create_tables()
        print("✅ Database setup completed!")
    except Exception as e:
        print(f"❌ Database setup failed: {e}")
        return False
    return True


def start_service(service_name: str, port: int, module_path: str):
    """Start a microservice."""
    print(f"Starting {service_name} on port {port}...")
    cmd = [
        "uvicorn",
        f"{module_path}:app",
        "--host", "0.0.0.0",
        "--port", str(port),
        "--reload"
    ]
    return subprocess.Popen(cmd)


def main():
    """Main startup function."""
    print("🚀 Starting Enterprise SaaS Platform...")
    
    # Setup database
    if not asyncio.run(setup_database()):
        print("❌ Failed to setup database. Exiting.")
        return
    
    # Start services
    services = [
        ("API Gateway", 8000, "src.services.api_gateway.main"),
        ("User Service", 8001, "src.services.user_service.main"),
        ("Tenant Service", 8002, "src.services.tenant_service.main"),
        ("Data Service", 8003, "src.services.data_service.main"),
        ("ML Service", 8004, "src.services.ml_service.main"),
    ]
    
    processes = []
    
    try:
        for service_name, port, module_path in services:
            process = start_service(service_name, port, module_path)
            processes.append((service_name, process))
            time.sleep(2)  # Stagger startup
        
        print("\n✅ All services started successfully!")
        print("\n📋 Service URLs:")
        for service_name, port, _ in services:
            print(f"  • {service_name}: http://localhost:{port}")
        
        print("\n📚 API Documentation:")
        print("  • API Gateway: http://localhost:8000/docs")
        
        print("\nPress Ctrl+C to stop all services...")
        
        # Wait for all processes
        for service_name, process in processes:
            process.wait()
    
    except KeyboardInterrupt:
        print("\n🛑 Stopping all services...")
        for service_name, process in processes:
            print(f"Stopping {service_name}...")
            process.terminate()
        
        # Wait for graceful shutdown
        time.sleep(2)
        
        # Force kill if needed
        for service_name, process in processes:
            if process.poll() is None:
                process.kill()
        
        print("✅ All services stopped.")


if __name__ == "__main__":
    main()