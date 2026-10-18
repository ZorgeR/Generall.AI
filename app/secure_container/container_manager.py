import os
import subprocess
import uuid
import logging
import json
import shlex
import threading
from pathlib import Path
import docker
from docker.errors import DockerException, ImageNotFound, ContainerError
import time

from . import sessions

# Cap on sandbox containers running at the same time across all users. Each container is
# limited to 512 MB / one CPU; without a cap a burst of users could exhaust the host.
MAX_SANDBOX_CONTAINERS = max(1, int(os.getenv("MAX_SANDBOX_CONTAINERS", "4") or "4"))
_SANDBOX_SLOTS = threading.BoundedSemaphore(MAX_SANDBOX_CONTAINERS)

# Memory limit of a sandbox container unless the caller asks for more (run_script(mem_limit=...)
# for the headless browser and the PDF renderer, which need about 1 GB).
DEFAULT_MEM_LIMIT = "512m"

# Scripts shipped with the bot that run_script() copies into a user's workspace and runs inside
# the sandbox (browse.py: headless Chromium page reader; render_document.py: pdf/docx/xlsx).
SCRIPTS_DIR = Path(__file__).parent / "scripts"
IMAGE_NAME = "secure-container"
# The files baked into the sandbox image. Scripts under scripts/ are copied into the workspace at
# run time (run_script), so they are not part of the image and not of its tag.
IMAGE_BUILD_FILES = ("Dockerfile", "entrypoint.sh", "requirements.txt")
IMAGE_LATEST_TAG = "latest"  # alias kept for people: the content tag is what the bot runs; also the
                             # fallback when building a new content tag fails (_ensure_image_exists)
# Content tags whose build failed in this process: the other ContainerManager instances
# (SecureToolWrapper's, web_tools') use the fallback instead of starting another long build.
_FAILED_BUILDS: set[str] = set()


def image_build_fingerprint(dockerfile_dir: Path | None = None) -> str:
    """12 hex chars of the sandbox build context (IMAGE_BUILD_FILES): the image tag.

    A deploy that changes the Dockerfile or the requirements gets a new tag, so
    ``_ensure_image_exists`` builds the new image on the first startup instead of silently
    keeping the old one (the ``latest`` tag used to be built once and never again)."""
    import hashlib

    directory = Path(dockerfile_dir) if dockerfile_dir is not None else Path(__file__).parent
    digest = hashlib.sha256()
    for name in IMAGE_BUILD_FILES:
        path = directory / name
        digest.update(name.encode("utf-8") + b"\0")
        try:
            digest.update(path.read_bytes())
        except OSError:
            digest.update(b"<missing>")
        digest.update(b"\0")
    return digest.hexdigest()[:12]

# Set up logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class ContainerManager:
    """
    Manages secure Docker containers for running potentially dangerous operations.

    Commands (run_command and everything built on it: Python code, shell scripts, package installs,
    the patched delete_file / create_directory) go to the user's PERSISTENT sandbox session
    (``secure_container.sessions``) when SANDBOX_SESSIONS is on and the session manager is up.
    When no session can start (memory, caps, a refused egress filter, a Docker error) or sessions
    are off, the command runs in a one-off container created for the call and removed afterwards,
    as before (with SANDBOX_EGRESS_FILTER=required and the filter not active: without network, see
    ``_one_shot_network``). ``run_script`` (browse_webpage, document rendering) is always one-off.
    """
    
    def __init__(self, base_data_path="./data", host_data_path=None):
        """
        Initialize the container manager.
        
        Args:
            base_data_path: Base path for user data directories (internal path)
            host_data_path: Base path on the host machine for Docker volume mounting.
                           If None, will be auto-detected or use base_data_path
        """
        self.base_data_path = Path(base_data_path)
        self.image_name = IMAGE_NAME
        self.image_tag = image_build_fingerprint()  # content hash: a changed build context = a rebuild
        self.container_prefix = "secure-container-"
        
        # Determine the host data path for volume mounting
        if host_data_path:
            self.host_data_path = Path(host_data_path)
        else:
            # Auto-detect if we're running in a container
            self.host_data_path = self._detect_host_data_path()
        
        logger.info(f"Container manager initialized with base_data_path={self.base_data_path}, host_data_path={self.host_data_path}")
        
        # Initialize Docker client
        try:
            self.docker_client = docker.from_env()
            logger.info("Docker client initialized successfully")
        except DockerException as e:
            logger.error(f"Failed to initialize Docker client: {str(e)}")
            raise
        
        # Ensure the secure container image is built
        self._ensure_image_exists()

        # The process-wide persistent-session manager (idempotent: this class is constructed more
        # than once at startup). A failure leaves sessions off: every call is then a one-off container.
        self.session_manager = None
        try:
            self.session_manager = sessions.init_manager(
                self.docker_client, f"{self.image_name}:{self.image_tag}", self.host_data_path, self.base_data_path
            )
        except Exception as e:  # noqa: BLE001
            logger.error(f"Persistent sandbox sessions are unavailable, using one-off containers: {e}")
    
    def _detect_host_data_path(self):
        """
        Detect the host data path for Docker volume mounting.
        If running inside a container with /app/data, translate to host path.
        Otherwise, use the absolute path of base_data_path.
        """
        base_absolute = self.base_data_path.absolute()
        
        # Check if we're running inside a container and using /app/data
        if str(base_absolute).startswith('/app/data'):            
            # Check if there's a .dockerenv file (indicates we're in a container)
            if Path('/.dockerenv').exists():
                # Try to read from an environment variable first
                workspace_root = os.environ.get('WORKSPACE_ROOT')
                if workspace_root:
                    host_path = Path(workspace_root) / 'data'
                    logger.info(f"Detected host data path from WORKSPACE_ROOT: {host_path}")
                    return host_path
                logger.warning("Running in container but WORKSPACE_ROOT not set, using heuristic")
                return Path('/opt/generall.ai/Generall.AI/data')
            
        return base_absolute
    
    def _ensure_image_exists(self):
        """Ensure the secure container image exists, build it if it doesn't"""
        try:
            self.docker_client.images.get(f"{self.image_name}:{self.image_tag}")
            logger.info(f"Image {self.image_name}:{self.image_tag} already exists")
            return None
        except ImageNotFound:
            if self.image_tag in _FAILED_BUILDS and self._fall_back_to_latest():
                return None
            logger.info(f"Building image {self.image_name}:{self.image_tag} (the build context changed or no image yet)")

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
                self._retag_latest_and_prune()
                return build_result
            except Exception as e:
                logger.error(f"Failed to build image: {str(e)}")
                # Print more detailed error information
                if hasattr(e, 'stderr'):
                    logger.error(f"Build stderr: {e.stderr}")
                # A failed build (network, a removed upstream package) must not take the whole bot
                # down while the previous image is still there: run on that one, rebuild next start.
                _FAILED_BUILDS.add(self.image_tag)
                if self._fall_back_to_latest():
                    return None
                raise

    def _fall_back_to_latest(self) -> bool:
        """Run on ``secure-container:latest`` (the last image built successfully) when the current
        content tag could not be built. False when there is no such image either."""
        wanted = f"{self.image_name}:{self.image_tag}"
        try:
            self.docker_client.images.get(f"{self.image_name}:{IMAGE_LATEST_TAG}")
        except Exception:  # noqa: BLE001 - ImageNotFound or Docker trouble: nothing to fall back to
            return False
        logger.error(
            f"Could not build {wanted}; running on the previous sandbox image "
            f"{self.image_name}:{IMAGE_LATEST_TAG} until a restart builds it (tools that need the new "
            f"image, e.g. browse_webpage or the pdf render, may fail meanwhile)"
        )
        self.image_tag = IMAGE_LATEST_TAG
        return True

    def _retag_latest_and_prune(self):
        """Point ``secure-container:latest`` at the new image and drop the older content-tagged
        images (they only take disk). Best effort: a failure here never fails startup."""
        current = f"{self.image_name}:{self.image_tag}"
        try:
            image = self.docker_client.images.get(current)
            image.tag(self.image_name, IMAGE_LATEST_TAG, force=True)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Could not tag {current} as {IMAGE_LATEST_TAG}: {e}")
            return
        try:
            for old in self.docker_client.images.list(name=self.image_name):
                if old.id == image.id:
                    continue
                tags = [t for t in (old.tags or []) if t.startswith(f"{self.image_name}:")]
                if not tags:
                    continue
                logger.info(f"Removing stale sandbox image {', '.join(tags)}")
                self.docker_client.images.remove(old.id, force=True)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Could not prune old sandbox images: {e}")

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
        # Make the directory world-writable so the non-root `runner` user inside
        # the secure container can write files (e.g. output files, temp scripts).
        # The container itself is the security boundary, not the directory permissions.
        # Temporary fix to allow the non-root `runner` user to write to the directory.
        os.chmod(user_dir, 0o777)
        return user_dir
    
    def _sessions(self):
        """The session manager commands go to, or None (SANDBOX_SESSIONS off or no manager).

        Read from the instance, not the module: a ContainerManager built without ``__init__``
        (tests) has none and always uses one-off containers."""
        if not sessions.SESSIONS_ENABLED:
            return None
        return getattr(self, "session_manager", None)

    def run_command(self, user_id, command, timeout=60, network_enabled=False, mem_limit=None):
        """
        Run a command for the specified user: in the persistent sandbox session when possible,
        else in a one-off container.

        In the session the command runs as ``timeout -k 5 <timeout> bash -lc <command>``, so shell
        syntax works, and ``network_enabled`` / ``mem_limit`` are ignored (the session always has
        filtered internet and its own memory limit). When no session can start (CapacityError) or
        sessions are off, a one-off container runs ``bash -lc <command>`` with ``network_enabled``
        and ``mem_limit`` as given (blocking until a MAX_SANDBOX_CONTAINERS slot is free); a note
        says so. Callers run this in a worker thread.

        Args:
            user_id: User ID
            command: Command to run
            timeout: Timeout in seconds
            network_enabled: Network access for a one-off container
            mem_limit: Docker memory limit for a one-off container (e.g. "1g"); None = DEFAULT_MEM_LIMIT

        Returns:
            Command output
        """
        manager = self._sessions()
        reason = None
        if manager is not None:
            try:
                return manager.exec(user_id, command, timeout=timeout)
            except sessions.CapacityError as e:
                reason = str(e)
                logger.warning(f"No sandbox session for user {user_id} ({reason}); using a one-off container")
        result = self.run_one_shot(user_id, command, timeout, network_enabled, mem_limit, shell=True)
        if reason:
            result += (f"\n[Note: this ran in a one-off container because the persistent sandbox session is "
                       f"unavailable: {reason} Background processes and packages installed there do not stay.]")
        return result

    def run_one_shot(self, user_id, command, timeout=60, network_enabled=False, mem_limit=None, *,
                     shell=False, replay_packages=True):
        """Run ``command`` in a fresh container that is removed afterwards (the pre-session path).

        Blocks until a sandbox slot is free (see MAX_SANDBOX_CONTAINERS). ``shell=True`` runs it
        through ``bash -lc`` (otherwise Docker splits the string itself, no shell syntax);
        ``replay_packages`` reinstalls the user's saved packages first (the validated names of
        ``sessions.persisted_packages``; forces network on)."""
        with _SANDBOX_SLOTS:
            return self._run_command_in_slot(user_id, command, timeout, network_enabled, mem_limit,
                                             shell=shell, replay_packages=replay_packages)

    def run_script(self, user_id, script_path, args=(), timeout=90, network_enabled=False, mem_limit=None):
        """Run one of the bundled sandbox scripts (``secure_container/scripts/``) for a user.

        Always a one-off container (:meth:`run_one_shot`), never the persistent session: the
        browser reader and the PDF renderer need their own memory limit, and the user's saved
        packages are NOT reinstalled first (the scripts only use the image's libraries, and a
        replay would add network and minutes to every call).
        The script is copied into the user's workspace under a unique temp name (tool calls
        of one message run concurrently) and executed as ``python -I <file> <args>``.
        ``script_path`` is the bare file name of a bundled script (anything
        with a directory part, ``../`` included, is refused: the model never chooses what runs
        here, but the tools must not be able to ship arbitrary files either); ``args`` are
        quoted for the shell. The temp file is removed afterwards; the return value is the raw
        container output (entrypoint banner first, so callers that expect JSON take the last
        line that parses, see ``agents.web_tools.last_json_line``).
        """
        name = str(script_path or "")
        if not name or Path(name).name != name:
            return f"Error: sandbox script not found: {script_path}"
        source = SCRIPTS_DIR / name
        if not source.is_file():
            return f"Error: sandbox script not found: {script_path}"
        user_dir = self.ensure_user_directory(user_id)
        target_file, file_in_container = self._workspace_file(user_dir, "temp_tool", ".py")
        try:
            # created with its mode in one step, never through a link: the user's sandbox session
            # can swap a workspace path for a link at any moment, and the bot is root
            sessions.write_new_file(user_dir, target_file.name, source.read_bytes(), 0o755)
            # -I (isolated): the script's directory is the user's workspace, where the model writes
            # files; without it a workspace json.py/random.py or a `pip install --user` copy of
            # openpyxl/playwright would shadow what the script imports. The image installs the
            # script dependencies system-wide, and PLAYWRIGHT_BROWSERS_PATH is not a PYTHON* variable.
            command = " ".join([f"python -I {file_in_container}"] + [shlex.quote(str(a)) for a in args])
            return self.run_one_shot(user_id, command, timeout, network_enabled, mem_limit, replay_packages=False)
        finally:
            self._remove_workspace_file(target_file)

    @staticmethod
    def _remove_workspace_file(path):
        """Remove a per-call workspace file (unlink never follows a link; a missing one is fine)."""
        try:
            Path(path).unlink(missing_ok=True)
        except Exception as e:
            logger.error(f"Error removing temporary file {path}: {str(e)}")

    @staticmethod
    def _workspace_file(user_dir, prefix, suffix):
        """A per-call file in the user's workspace: (host path, path inside the container).

        The name carries a uuid because tool calls of one assistant message run
        CONCURRENTLY (AgentAnthropic.run_tool_batch). A fixed name like
        ``temp_setup.sh`` was overwritten by whichever call wrote last, so every
        container then executed that one command and all callers got its output.
        """
        name = f"{prefix}_{uuid.uuid4().hex[:8]}{suffix}"
        return user_dir / name, f"/home/runner/workspace/{name}"

    def _one_shot_network(self, network_enabled):
        """docker run kwargs for a one-off container's network: none, or with network the sandbox
        network when the session manager set it up (egress-filtered), else Docker's default bridge.

        With SANDBOX_EGRESS_FILTER=required a one-off container gets network only while the filter
        is on (``manager.egress_state``): the fallback for a session refused because the filter
        failed must not run unfiltered instead."""
        if not network_enabled:
            return {"network_mode": "none"}
        manager = getattr(self, "session_manager", None)
        if sessions.EGRESS_FILTER == "required" and getattr(manager, "egress_state", None) != "on":
            logger.warning("No network for a one-off sandbox container: the egress filter is not active "
                           "(SANDBOX_EGRESS_FILTER=required)")
            return {"network_mode": "none"}
        if manager is not None:
            kwargs = {"network": manager.network_name}
            if sessions.DNS_SERVERS:
                kwargs["dns"] = list(sessions.DNS_SERVERS)
            return kwargs
        return {"network_mode": "bridge"}

    def _run_command_in_slot(self, user_id, command, timeout=60, network_enabled=False, mem_limit=None, *,
                             shell=False, replay_packages=True):
        # Ensure user directory exists
        user_dir = self.ensure_user_directory(user_id)

        # The saved packages, reinstalled first. Read like a session start does
        # (sessions.persisted_packages): never through a link, never blocking on a FIFO (we hold a
        # MAX_SANDBOX_CONTAINERS slot here), and only names that pass validation reach the script.
        package_installation_prefix = ""
        packages = sessions.persisted_packages(user_dir) if replay_packages else {"apt": [], "pip": []}
        if packages["apt"] or packages["pip"]:
            logger.info(f"Reinstalling saved packages for user {user_id} in a one-off container: {packages}")
            # sessions.replay_script: apt via sudo, the best-effort pip install of python3-* apt
            # packages, then the pip list; it ends with `exit`, so it runs in a subshell and a failed
            # install does not stop the command after it
            package_installation_prefix = (
                "#!/bin/bash\n(\n" + sessions.replay_script(packages["apt"], packages["pip"]) + ")\n"
            )
            # Network access is required to install packages
            network_enabled = True
            logger.info("Network access enabled for package installation")

        # Generate a unique container name
        container_name = f"{self.container_prefix}{user_id}-{uuid.uuid4().hex[:8]}"

        command_text = command  # what the logs and the strace check look at
        script_path = None
        try:
            # Run the container with the user directory mounted
            logger.info(f"Running command in container {container_name}: {command}")
            
            # Network: none, the filtered sandbox network, or the default bridge (see _one_shot_network)
            network_kwargs = self._one_shot_network(network_enabled)
            
            # If we have packages to install, create a script to do that first
            if package_installation_prefix:
                # A temporary script in the user directory (unique per call), created executable
                # in one step and never through a link (the sandbox can plant one at any time)
                script_path, script_in_container = self._workspace_file(user_dir, "temp_setup", ".sh")
                script = f"{package_installation_prefix}\n# Now run the actual command\n{command}\n"
                sessions.write_new_file(user_dir, script_path.name, script.encode("utf-8"), 0o755)

                # Update the command to run our script
                command = script_in_container
            elif shell:
                # Through bash, like the persistent session: pipes, &&, cd and globs work
                command = ["bash", "-lc", command]
            
            # Get the host path for the user directory to mount into the container
            # user_dir is relative to base_data_path, so we need to construct the host path
            relative_user_path = user_dir.relative_to(self.base_data_path)
            host_user_dir = self.host_data_path / relative_user_path
            host_user_dir_str = str(host_user_dir.absolute())
            
            logger.info(f"User directory (internal): {user_dir}")
            logger.info(f"Host directory (for mounting): {host_user_dir_str}")
            logger.info(f"Mounting to container: /home/runner/workspace")

            container = self.docker_client.containers.run(
                image=f"{self.image_name}:{self.image_tag}",
                command=command,
                volumes={
                    host_user_dir_str: {
                        'bind': '/home/runner/workspace',
                        'mode': 'rw',
                        'propagation': 'shared'
                    }
                },
                name=container_name,
                detach=True,
                remove=False,  # We'll remove it manually after getting logs
                **network_kwargs,  # Network access based on parameter
                cap_add=["SYS_PTRACE"] if "strace" in command_text else None,  # Add capabilities for strace if needed
                mem_limit=mem_limit or DEFAULT_MEM_LIMIT,  # Limit memory to prevent resource exhaustion
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
            if script_path is not None:
                self._remove_workspace_file(script_path)
    
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
        user_dir = self.ensure_user_directory(user_id)
        target_file, file_in_container = self._workspace_file(user_dir, "temp_code", ".py")
        try:
            # Written into the workspace readable and executable by all, mode set on the new file
            # itself: a chmod by path would follow a link the sandbox swapped in meanwhile
            sessions.write_new_file(user_dir, target_file.name, str(code).encode("utf-8"), 0o755)

            # Run the code in the container
            command = f"python {file_in_container}"
            return self.run_command(user_id, command, timeout, network_enabled)
        finally:
            # Clean up the temporary file in the user directory
            self._remove_workspace_file(target_file)
    
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
        # Serialize args as JSON string, then parse in the script via json.loads
        # to avoid JSON true/false/null being invalid Python literals
        args_json = json.dumps(args)
        
        # Create a Python script to perform the file operation
        script = f"""
import os
import json
import sys
from pathlib import Path

MAX_FILES = 200

def list_files(path='.', page=1):
    all_items = []
    base = Path(path)
    for item in sorted(base.iterdir(), key=lambda x: (x.is_file(), x.name)):
        if item.is_file():
            all_items.append({{"name": item.name, "type": "file", "size": item.stat().st_size}})
        elif item.is_dir():
            sub_count = sum(1 for _ in item.iterdir()) if item.exists() else 0
            all_items.append({{"name": item.name, "type": "directory", "items": sub_count}})

    total = len(all_items)
    start = (page - 1) * MAX_FILES
    end = start + MAX_FILES
    page_items = all_items[start:end]
    total_pages = (total + MAX_FILES - 1) // MAX_FILES if total > 0 else 1

    result = {{"files": page_items, "total": total, "page": page, "total_pages": total_pages}}
    if total > MAX_FILES:
        result["note"] = f"Showing {{len(page_items)}} of {{total}} items (page {{page}}/{{total_pages}}). Use page parameter to see more."
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

args = json.loads({repr(args_json)})

operation = "{operation}"

if operation == "list_files":
    path = args.get("path", ".")
    page = args.get("page", 1)
    result = list_files(path, page)
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
        user_dir = self.ensure_user_directory(user_id)
        target_file, file_in_container = self._workspace_file(user_dir, "temp_script", ".sh")
        try:
            # Created executable in one step, never through a link (see run_python_code)
            sessions.write_new_file(user_dir, target_file.name, str(script_content).encode("utf-8"), 0o755)

            # Run the script in the container
            command = f"/bin/bash {file_in_container}"
            return self.run_command(user_id, command, timeout, network_enabled)
        finally:
            # Clean up the temporary file in the user directory
            self._remove_workspace_file(target_file)
    
    def install_package(self, user_id, package_name, timeout=300, manager="apt"):
        """
        Install apt or pip packages for the user and save them so new sessions reinstall them.

        In the persistent session (``SessionManager.install_package``): ``sudo apt-get install`` or
        ``pip install --user``, then the names go to installed_packages.txt (apt) or
        installed_python_packages.txt (pip) in the workspace on success. Without a session
        (CapacityError, sessions off) the install runs in a one-off container with network, which
        only saves the names; the next session start reinstalls them.

        Args:
            user_id: User ID
            package_name: One or more package names separated by spaces (validated, never a shell string)
            timeout: Timeout in seconds
            manager: "apt" (default) or "pip"

        Returns:
            Installation output, or "Error: ..."
        """
        manager_name = str(manager or "apt").strip().lower()
        if manager_name not in sessions.PACKAGE_FILES:
            return 'Error: manager must be "apt" or "pip".'
        try:
            packages = sessions.parse_packages(manager_name, package_name)
        except ValueError as e:
            return f"Error: {e}"
        session_manager = self._sessions()
        if session_manager is not None:
            try:
                return session_manager.install_package(user_id, " ".join(packages), manager_name, timeout=timeout)
            except sessions.CapacityError as e:
                logger.warning(f"No sandbox session for user {user_id} ({e}); installing in a one-off container")
        command = sessions.install_command(manager_name, packages)
        result = self.run_one_shot(user_id, command, timeout, True, None, shell=True, replay_packages=False)
        if result.startswith(("Error", "Container error")):
            return result
        added = sessions.add_persisted_packages(self.ensure_user_directory(user_id), manager_name, packages)
        logger.info(f"Saved {manager_name} packages {added or packages} for user {user_id}")
        return f"{result.rstrip()}\n{sessions.persisted_note(manager_name, packages, added)}"

    def install_python_package(self, user_id, package_name, timeout=300):
        """``install_package(..., manager="pip")`` (kept for SystemTools and older callers)."""
        return self.install_package(user_id, package_name, timeout=timeout, manager="pip")

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