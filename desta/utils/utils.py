import subprocess
from typing import Optional


def run(cmd: str) -> Optional[str]:
    """
    Run a shell command and return its output.
    
    Args:
        cmd: Shell command to execute
        
    Returns:
        Command output as string, or None if command fails
    """
    try:
        return subprocess.check_output(cmd, shell=True, text=True).strip()
    except subprocess.SubprocessError:
        return None