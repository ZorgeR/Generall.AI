import os
import sys
import logging
from pathlib import Path
import importlib
from .tool_integrator import ToolIntegrator

# Set up logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def initialize_secure_containers(base_data_path="./data"):
    """
    Initialize the secure container system.
    This function should be called when the application starts.
    
    Args:
        base_data_path: Base path for user data directories
        
    Returns:
        True if initialization was successful, False otherwise
    """
    try:
        logger.info("Initializing secure container system...")
        
        # Import the agents module
        import agents
        
        # Check if Docker is available
        try:
            import docker
            
            # Try to connect to Docker
            docker_client = docker.from_env()
            docker_version = docker_client.version()
            logger.info(f"Docker is available. Version: {docker_version.get('Version', 'unknown')}")
            
            from .container_manager import ContainerManager
            container_manager = ContainerManager(base_data_path=base_data_path)
            # Store the container manager as a global variable
            globals()['_container_manager'] = container_manager
            
            # Create a tool integrator and patch all tools
            tool_integrator = ToolIntegrator(base_data_path=base_data_path)
            if tool_integrator.patch_all_tools(agents):
                logger.info("Successfully patched all tools to use secure containers")
            else:
                logger.warning("Failed to patch some tools, secure container functionality may be limited")
                
            return True
        except Exception as docker_error:
            logger.error(f"Docker initialization failed: {str(docker_error)}")
            logger.warning("Continuing without secure container functionality")
            return False
    except Exception as e:
        logger.error(f"Error initializing secure container system: {str(e)}")
        return False

def cleanup_containers():
    """
    Clean up all containers created by the secure container system.
    This function should be called when the application shuts down.
    
    Returns:
        True if cleanup was successful, False otherwise
    """
    try:
        logger.info("Cleaning up secure containers...")
        
        # Check if container manager exists
        if '_container_manager' not in globals():
            logger.warning("No container manager found, skipping cleanup")
            return True
            
        # Get the container manager
        container_manager = globals()['_container_manager']
        
        # Clean up all containers
        try:
            # Stop and remove all containers with our prefix
            for container in container_manager.docker_client.containers.list(all=True):
                if container.name.startswith(container_manager.container_prefix):
                    logger.info(f"Stopping and removing container {container.name}")
                    try:
                        container.stop(timeout=1)
                    except:
                        pass  # Container might already be stopped
                    try:
                        container.remove(force=True)
                    except:
                        pass  # Container might already be removed
                        
            logger.info("All secure containers cleaned up successfully")
            return True
        except Exception as e:
            logger.error(f"Error cleaning up containers: {str(e)}")
            return False
    except Exception as e:
        logger.error(f"Error in cleanup_containers: {str(e)}")
        return False 