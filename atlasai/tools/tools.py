# atlasai/tools/tools.py
import platform
import logging
import os
import subprocess
from typing import List
from langchain_community.tools import DuckDuckGoSearchRun, ShellTool
from langchain_community.agent_toolkits import FileManagementToolkit

logger = logging.getLogger(__name__)

def get_os():
    """Return the current operating system."""
    system = platform.uname()
    return system.system

def search(query: str) -> str:
    """Search the web using DuckDuckGo."""
    try:
        search = DuckDuckGoSearchRun()
        return search.invoke(query)
    except Exception as e:
        logger.error(f"Error searching with DuckDuckGo: {str(e)}")
        return f"Error performing search: {str(e)}"

def list_directory(directory: str = ".") -> str:
    """List files in the specified directory.
    
    Args:
        directory: Path to list (default: current directory)
    
    Returns:
        String with directory contents
    """
    try:
        if not os.path.isdir(directory):
            return f"Error: '{directory}' is not a valid directory"
        
        # Use 'ls' or 'dir' command directly depending on the system
        if platform.system() == "Windows":
            result = subprocess.run(["dir", "/b", directory], capture_output=True, text=True)
        else:
            result = subprocess.run(["ls", "-la", directory], capture_output=True, text=True)
            
        if result.returncode != 0:
            return f"Error listing directory: {result.stderr}"
        
        return result.stdout
    except Exception as e:
        logger.error(f"Error listing directory {directory}: {str(e)}")
        return f"Error listing directory: {str(e)}"

def read_file(file_path: str) -> str:
    """Read the contents of a file.
    
    Args:
        file_path: Path to the file to read
    
    Returns:
        Contents of the file
    """
    try:
        toolkit = FileManagementToolkit(root_dir=None)
        read_tool = None
        
        # Get the read file tool
        for tool in toolkit.get_tools():
            if tool.__class__.__name__ == "ReadFileTool":
                read_tool = tool
                break
                
        if read_tool:
            return read_tool.invoke({"file_path": file_path})
        else:
            return f"Error: Could not create file reading tool"
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {str(e)}")
        return f"Error reading file: {str(e)}"

def execute_command(commands: List[str]) -> str:
    """Execute shell commands.
    
    Args:
        commands: List of commands to execute
        
    Returns:
        Command output
    """
    try:
        # Limit commands to safe read-only operations
        safe_commands = ["ls", "dir", "cat", "type", "find", "grep", "head", "tail"]
        
        # Verify commands are safe
        for cmd in commands:
            cmd_parts = cmd.split()
            if not cmd_parts:
                continue
                
            base_cmd = cmd_parts[0].lower()
            if base_cmd not in safe_commands:
                return f"Error: Command not allowed '{base_cmd}'. Only read commands are permitted."
        
        # Use ShellTool to execute commands
        shell_tool = ShellTool()
        return shell_tool.run({"commands": commands})
    except Exception as e:
        logger.error(f"Error executing commands {commands}: {str(e)}")
        return f"Error executing commands: {str(e)}"

def write_file(file_path: str, content: str, overwrite: bool = False) -> str:
    """Crea un nuevo archivo o sobrescribe uno existente.
    
    Args:
        file_path: Ruta al archivo que se va a escribir
        content: Contenido a escribir en el archivo
        overwrite: Si es True, sobrescribirá un archivo existente
    
    Returns:
        Mensaje de éxito o error
    """
    try:
        # Verificar si el archivo ya existe y si se debe sobrescribir
        if os.path.exists(file_path) and not overwrite:
            return f"Error: El archivo '{file_path}' ya existe. Usa overwrite=True para sobrescribir."
        
        # Asegurarse de que el directorio existe
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            
        # Escribir el archivo
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
            
        return f"Archivo '{file_path}' escrito correctamente."
    except Exception as e:
        logger.error(f"Error escribiendo archivo {file_path}: {str(e)}")
        return f"Error escribiendo archivo: {str(e)}"

def append_file(file_path: str, content: str) -> str:
    """Añade contenido al final de un archivo existente.
    
    Args:
        file_path: Ruta al archivo al que se va a añadir contenido
        content: Contenido a añadir al archivo
    
    Returns:
        Mensaje de éxito o error
    """
    try:
        # Verificar si el archivo existe
        if not os.path.exists(file_path):
            return f"Error: El archivo '{file_path}' no existe."
        
        # Añadir al archivo
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(content)
            
        return f"Contenido añadido al archivo '{file_path}' correctamente."
    except Exception as e:
        logger.error(f"Error añadiendo contenido al archivo {file_path}: {str(e)}")
        return f"Error añadiendo contenido al archivo: {str(e)}"

def edit_file(file_path: str, search_text: str, replace_text: str, regex: bool = False) -> str:
    """Edita partes de un archivo existente.
    
    Args:
        file_path: Ruta al archivo que se va a editar
        search_text: Texto a buscar
        replace_text: Texto de reemplazo
        regex: Si es True, search_text se interpretará como una expresión regular
    
    Returns:
        Mensaje de éxito o error
    """
    try:
        import re
        
        # Verificar si el archivo existe
        if not os.path.exists(file_path):
            return f"Error: El archivo '{file_path}' no existe."
        
        # Leer el contenido actual
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Realizar el reemplazo
        occurrences = 0
        if regex:
            new_content, count = re.subn(search_text, replace_text, content)
            occurrences = count
        else:
            new_content = content.replace(search_text, replace_text)
            occurrences = content.count(search_text)
        
        # Escribir el nuevo contenido
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(new_content)
            
        return f"Archivo '{file_path}' editado correctamente. {occurrences} ocurrencias reemplazadas."
    except Exception as e:
        logger.error(f"Error editando archivo {file_path}: {str(e)}")
        return f"Error editando archivo: {str(e)}"