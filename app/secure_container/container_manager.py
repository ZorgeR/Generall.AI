import os
import subprocess
import uuid
import logging
import tempfile
import json
import shutil
from pathlib import Path
import docker
from docker.errors import DockerException, ImageNotFound, ContainerError
import time

# Set up logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class ContainerManager:
    """
    Manages secure Docker containers for running potentially dangerous operations.
    Each user gets their own container that is created on demand and destroyed after use.
    """
    
    def __init__(self, base_data_path="./data"):
        """
        Initialize the container manager.
        
        Args:
            base_data_path: Base path for user data directories
        """
        self.base_data_path = Path(base_data_path)
        self.image_name = "secure-container"
        self.image_tag = "latest"
        self.container_prefix = "secure-container-"
        
        # Initialize Docker client
        try:
            self.docker_client = docker.from_env()
            logger.info("Docker client initialized successfully")
        except DockerException as e:
            logger.error(f"Failed to initialize Docker client: {str(e)}")
            raise
        
        # Ensure the secure container image is built
        self._ensure_image_exists()
    
    def _ensure_image_exists(self):
        """Ensure the secure container image exists, build it if it doesn't"""
        try:
            self.docker_client.images.get(f"{self.image_name}:{self.image_tag}")
            logger.info(f"Image {self.image_name}:{self.image_tag} already exists")
        except ImageNotFound:
            logger.info(f"Building image {self.image_name}:{self.image_tag}")
            
            # Get the directory of the Dockerfile
            dockerfile_dir = Path(__file__).parent.absolute()
            logger.info(f"Dockerfile directory: {dockerfile_dir}")
            
            # List the contents of the directory to ensure files are present
            try:
                files = list(dockerfile_dir.glob('*'))
                logger.info(f"Files in Dockerfile directory: {[f.name for f in files]}")
                
                # Check if Dockerfile exists
                if not (dockerfile_dir / 'Dockerfile').exists():
                    logger.error(f"Dockerfile not found in {dockerfile_dir}")
                    raise FileNotFoundError(f"Dockerfile not found in {dockerfile_dir}")
                
                # Check if entrypoint.sh exists
                if not (dockerfile_dir / 'entrypoint.sh').exists():
                    logger.error(f"entrypoint.sh not found in {dockerfile_dir}")
                    raise FileNotFoundError(f"entrypoint.sh not found in {dockerfile_dir}")
                
                # Check requirements.txt
                if not (dockerfile_dir / 'requirements.txt').exists():
                    logger.warning(f"requirements.txt not found in {dockerfile_dir}, creating empty one")
                    with open(dockerfile_dir / 'requirements.txt', 'w') as f:
                        f.write("# Secure container requirements\n")
            except Exception as e:
                logger.error(f"Error checking Dockerfile directory: {str(e)}")
            
            # Build the image
            try:
                logger.info(f"Starting build of image {self.image_name}:{self.image_tag}")
                build_result = self.docker_client.images.build(
                    path=str(dockerfile_dir),
                    tag=f"{self.image_name}:{self.image_tag}",
                    rm=True,
                    forcerm=True
                )
                logger.info(f"Successfully built image {self.image_name}:{self.image_tag}")
                return build_result
            except Exception as e:
                logger.error(f"Failed to build image: {str(e)}")
                # Print more detailed error information
                if hasattr(e, 'stderr'):
                    logger.error(f"Build stderr: {e.stderr}")
                raise
    
    def ensure_user_directory(self, user_id):
        """
        Ensure the user directory exists.
        
        Args:
            user_id: User ID
            
        Returns:
            Path to the user directory
        """
        user_dir = self.base_data_path / str(user_id)
        user_dir.mkdir(parents=True, exist_ok=True)
        return user_dir
    
    def run_command(self, user_id, command, timeout=60, network_enabled=False):
        """
        Run a command in a secure container for the specified user.
        
        Args:
            user_id: User ID
            command: Command to run
            timeout: Timeout in seconds
            network_enabled: Whether to enable network access for the container
            
        Returns:
            Command output
        """
        # Ensure user directory exists
        user_dir = self.ensure_user_directory(user_id)
        
        # Check for user packages file and prepare installation script if needed
        packages_file = user_dir / "installed_packages.txt"
        package_installation_prefix = ""
        
        if packages_file.exists():
            try:
                with open(packages_file, 'r') as f:
                    packages = [line.strip() for line in f if line.strip()]
                
                if packages:
                    logger.info(f"Found {len(packages)} installed packages for user {user_id}")
                    packages_str = " ".join(packages)
                    # Create a prefix script to install packages before running the actual command
                    package_installation_prefix = f"""
#!/bin/bash
echo "Installing user packages: {packages_str}"
sudo apt-get update -qq
sudo apt-get install -y -qq {packages_str} > /dev/null 2>&1
echo "Package installation completed"
"""
                    # If we're installing packages, we need network access
                    network_enabled = True
                    logger.info("Network access enabled for package installation")
            except Exception as e:
                logger.error(f"Error reading packages file for user {user_id}: {str(e)}")
        
        # Generate a unique container name
        container_name = f"{self.container_prefix}{user_id}-{uuid.uuid4().hex[:8]}"
        
        try:
            # Run the container with the user directory mounted
            logger.info(f"Running command in container {container_name}: {command}")
            
            # Determine network mode based on parameter
            network_mode = "bridge" if network_enabled else "none"
            
            # If we have packages to install, create a script to do that first
            if package_installation_prefix:
                # Create a temporary script file in the user directory
                script_path = user_dir / "temp_setup.sh"
                with open(script_path, 'w') as f:
                    f.write(package_installation_prefix)
                    f.write(f"\n# Now run the actual command\n{command}\n")
                
                # Make the script executable
                os.chmod(script_path, 0o755)
                
                # Update the command to run our script
                command = "/home/runner/workspace/temp_setup.sh"
            
            # volume prefix
            # TODO: make this configurable
            user_dir_short = str(user_dir.absolute())
            user_dir_fixed = user_dir_short.replace("/app/data", "")
            prefix = "/Users/zorg/dev/git/generall.ai/generall.ai/data"
            user_dir_inside_host = f"{prefix}/{user_dir_fixed}"

            container = self.docker_client.containers.run(
                image=f"{self.image_name}:{self.image_tag}",
                command=command,
                volumes={
                    str(user_dir_inside_host): {
                        'bind': '/home/runner/workspace',
                        'mode': 'rw',
                        'propagation': 'shared'
                    }
                },
                name=container_name,
                detach=True,
                remove=False,  # We'll remove it manually after getting logs
                network_mode=network_mode,  # Network access based on parameter
                cap_add=["SYS_PTRACE"] if "strace" in command else None,  # Add capabilities for strace if needed
                mem_limit="512m",  # Limit memory to prevent resource exhaustion
                cpu_quota=100000,  # Limit CPU usage (100% of one CPU)
                cpu_period=100000
            )
            
            # Wait for the container to finish or timeout
            try:
                exit_code = container.wait(timeout=timeout)["StatusCode"]
                logs = container.logs().decode('utf-8')
                
                if exit_code != 0:
                    logger.warning(f"Command exited with non-zero status: {exit_code}")
                    return f"Error (exit code {exit_code}):\n{logs}"
                
                return logs
            except Exception as e:
                logger.error(f"Error waiting for container: {str(e)}")
                return f"Error executing command: {str(e)}"
            
        except ContainerError as e:
            logger.error(f"Container error: {str(e)}")
            return f"Container error: {str(e)}"
        except Exception as e:
            logger.error(f"Error running command in container: {str(e)}")
            return f"Error: {str(e)}"
        finally:
            # Always clean up the container
            try:
                container = self.docker_client.containers.get(container_name)
                container.remove(force=True)
                logger.info(f"Container {container_name} removed")
            except Exception as e:
                logger.error(f"Error removing container {container_name}: {str(e)}")
            
            # Clean up the temporary script if it was created
            if package_installation_prefix and script_path.exists():
                try:
                    script_path.unlink()
                except Exception as e:
                    logger.error(f"Error removing temporary script: {str(e)}")
    
    def run_python_code(self, user_id, code, timeout=60, network_enabled=False):
        """
        Run Python code in a secure container for the specified user.
        
        Args:
            user_id: User ID
            code: Python code to run
            timeout: Timeout in seconds
            network_enabled: Whether to enable network access for the container
            
        Returns:
            Code execution output
        """
        # Create a temporary file with the code
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
            temp_file.write(code)
            temp_file_path = temp_file.name
        
        try:
            # Copy the file to the user directory
            user_dir = self.ensure_user_directory(user_id)
            target_file = user_dir / "temp_code.py"
            shutil.copy(temp_file_path, target_file)
            
            # Run the code in the container
            command = "python /home/runner/workspace/temp_code.py"
            result = self.run_command(user_id, command, timeout, network_enabled)
            
            # Clean up the temporary file in the user directory
            if target_file.exists():
                target_file.unlink()
            
            return result
        finally:
            # Clean up the local temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
    
    def run_file_operation(self, user_id, operation, args, timeout=30):
        """
        Run a file operation in a secure container.
        
        Args:
            user_id: User ID
            operation: Operation name (list_files, read_file, etc.)
            args: Operation arguments
            timeout: Timeout in seconds
            
        Returns:
            Operation result
        """
        # Create a Python script to perform the file operation
        script = f"""
import os
import json
import sys
from pathlib import Path

def list_files(path='.'):
    result = []
    for item in Path(path).iterdir():
        if item.is_file():
            result.append({{"name": item.name, "type": "file", "size": item.stat().st_size}})
        elif item.is_dir():
            result.append({{"name": item.name, "type": "directory"}})
    return result

def read_file(filename):
    try:
        with open(filename, 'r') as f:
            return f.read()
    except Exception as e:
        return str(e)

def create_file(filename, content):
    try:
        with open(filename, 'w') as f:
            f.write(content)
        return f"File {{filename}} created successfully"
    except Exception as e:
        return str(e)

def delete_file(filename):
    try:
        os.remove(filename)
        return f"File {{filename}} deleted successfully"
    except Exception as e:
        return str(e)

def create_directory(dirname):
    try:
        os.makedirs(dirname, exist_ok=True)
        return f"Directory {{dirname}} created successfully"
    except Exception as e:
        return str(e)

def delete_directory(dirname):
    try:
        import shutil
        shutil.rmtree(dirname)
        return f"Directory {{dirname}} deleted successfully"
    except Exception as e:
        return str(e)

# Parse arguments
operation = "{operation}"
args = {json.dumps(args)}

# Execute the operation
if operation == "list_files":
    path = args.get("path", ".")
    result = list_files(path)
elif operation == "read_file":
    result = read_file(args["filename"])
elif operation == "create_file":
    result = create_file(args["filename"], args["content"])
elif operation == "delete_file":
    result = delete_file(args["filename"])
elif operation == "create_directory":
    result = create_directory(args["dirname"])
elif operation == "delete_directory":
    result = delete_directory(args["dirname"])
else:
    result = f"Unknown operation: {{operation}}"

# Print the result as JSON
print(json.dumps(result))
"""
        
        return self.run_python_code(user_id, script, timeout)
    
    def run_shell_script(self, user_id, script_content, timeout=60, network_enabled=False):
        """
        Run a shell script in a secure container.
        
        Args:
            user_id: User ID
            script_content: Content of the shell script
            timeout: Timeout in seconds
            network_enabled: Whether to enable network access for the container
            
        Returns:
            Script execution output
        """
        # Create a temporary file with the script
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as temp_file:
            temp_file.write(script_content)
            temp_file_path = temp_file.name
        
        try:
            # Copy the file to the user directory
            user_dir = self.ensure_user_directory(user_id)
            target_file = user_dir / "temp_script.sh"
            shutil.copy(temp_file_path, target_file)
            
            # Make the script executable
            os.chmod(target_file, 0o755)
            
            # Run the script in the container
            command = "/bin/bash /home/runner/workspace/temp_script.sh"
            result = self.run_command(user_id, command, timeout, network_enabled)
            
            # Clean up the temporary file in the user directory
            if target_file.exists():
                target_file.unlink()
            
            return result
        finally:
            # Clean up the local temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
    
    def install_package(self, user_id, package_name, timeout=300):
        """
        Install a package in the secure container and save it to the user's package list.
        
        Args:
            user_id: User ID
            package_name: Name of the package to install
            timeout: Timeout in seconds
            
        Returns:
            Installation output
        """
        # Ensure user directory exists
        user_dir = self.ensure_user_directory(user_id)
        
        # Create a script to install the package
        script = f"""#!/bin/bash
set -e
echo "Installing package: {package_name}"
sudo apt-get update
sudo apt-get install -y {package_name}
if [ $? -eq 0 ]; then
    echo "Package {package_name} installed successfully"
    exit 0
else
    echo "Failed to install package {package_name}"
    exit 1
fi
"""
        
        # Run the script with network access enabled
        result = self.run_shell_script(user_id, script, timeout, network_enabled=True)
        
        # If installation was successful, add the package to the user's package list
        if "installed successfully" in result:
            packages_file = user_dir / "installed_packages.txt"
            
            # Read existing packages
            existing_packages = set()
            if packages_file.exists():
                with open(packages_file, 'r') as f:
                    existing_packages = set(line.strip() for line in f if line.strip())
            
            # Add the new package if it's not already in the list
            if package_name not in existing_packages:
                existing_packages.add(package_name)
                
                # Write the updated package list
                with open(packages_file, 'w') as f:
                    for pkg in sorted(existing_packages):
                        f.write(f"{pkg}\n")
                
                logger.info(f"Added {package_name} to user {user_id}'s package list")
        
        return result
    
    def cleanup_all_containers(self):
        """Clean up all containers created by this manager"""
        try:
            containers = self.docker_client.containers.list(
                all=True,
                filters={"name": self.container_prefix}
            )
            
            for container in containers:
                try:
                    container.remove(force=True)
                    logger.info(f"Container {container.name} removed during cleanup")
                except Exception as e:
                    logger.error(f"Error removing container {container.name}: {str(e)}")
            
            return f"Cleaned up {len(containers)} containers"
        except Exception as e:
            logger.error(f"Error cleaning up containers: {str(e)}")
            return f"Error cleaning up containers: {str(e)}" 