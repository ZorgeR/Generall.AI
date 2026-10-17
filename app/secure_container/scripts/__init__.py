"""Scripts that run INSIDE the sandbox container, not in the bot process.

``ContainerManager.run_script`` copies one of these files into the user's workspace and runs
it with the sandbox's Python (``python <file> <args>``), so each script must be a single
self-contained file: only the standard library plus what ``secure_container/requirements.txt``
installs in the image (playwright, trafilatura, markdown, python-docx, openpyxl). They talk
back to the host by printing exactly one JSON line as their last line of output; the host
(``agents.web_tools.last_json_line``) ignores the entrypoint banner and anything else before it.
"""
