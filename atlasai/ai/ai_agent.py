import os
import logging
import re
import json
from typing import Dict, Any, Optional, Callable

from ollama import chat

from atlasai.tools import list_directory, read_file

logger = logging.getLogger(__name__)

class AgentCLI:
    """AI Agent for command line operations."""
    
    def __init__(self, model="codellama:7b", stream=False, system_prompt=None, language="en"):
        """Initialize the AI Agent.
        
        Args:
            model: The Ollama model to use
            stream: Whether to stream responses
            system_prompt: Custom system prompt
            language: Response language (en/es)
        """
        self.model = model
        self.stream = stream
        self.language = language
        self.system_prompt = system_prompt or self._get_default_system_prompt()
        
    def _get_default_system_prompt(self) -> str:
        """Get the default system prompt."""
        base_prompt = """You are an AI assistant specialized in analyzing software projects.
        Your task is to determine the type of project (like Flask, FastAPI, Django, etc.) 
        and suggest the optimal deployment command for AtlasServer.
        
        Use the tools provided to explore the project structure:
        - list_directory: To see the contents of directories
        - read_file: To examine key files like main.py, requirements.txt
        - execute_command: To run shell commands that help explore the filesystem
        
        Start by using execute_command with "ls -la" (or "dir" on Windows) to see the project contents.
        Then locate and read important files like:
        - app/main.py or main.py
        - requirements.txt
        - pyproject.toml
        - setup.py
        - app/__init__.py
        
        Be thorough in your analysis and explain your reasoning clearly.
        When you've completed your analysis, respond in JSON format with these fields:
        {
            "type": "Project type (Flask, FastAPI, Django, etc.)",
            "command": "Exact command to run the application",
            "environment_vars": {"ENV_VAR1": "value1", "ENV_VAR2": "value2"},
            "port": "Recommended port number",
            "reasoning": "Detailed explanation of your analysis and recommendations"
        }
        
        In your reasoning, use Markdown formatting with **bold** for important concepts, 
        numbered lists (1., 2., etc.) for steps, and organize information clearly.
        """
        
        # Add language instruction
        if self.language == "es":
            base_prompt += "\n\nIMPORTANT: Provide your explanation in the 'reasoning' field in Spanish, while keeping the JSON structure and field names in English."
        
        return base_prompt
    
    def _format_directory_output(self, content):
        """Format directory listing output into a rich table."""
        try:
            from rich.table import Table
            
            # Create table for files
            table = Table(show_header=True)
            table.add_column("Permissions", style="cyan")
            table.add_column("Size", justify="right", style="green")
            table.add_column("Modified", style="magenta")
            table.add_column("Name", style="blue")
            
            # Process each line
            lines = content.split("\n")
            for line in lines:
                if not line.strip() or line.startswith("total"):
                    continue
                    
                parts = line.split()
                if len(parts) >= 8:
                    perms = parts[0]
                    size = parts[4]
                    date = f"{parts[5]} {parts[6]}"
                    name = " ".join(parts[7:])
                    
                    # Add icon based on type
                    if perms.startswith("d"):
                        name = "📁 " + name
                    elif perms.startswith("-"):
                        name = "📄 " + name
                    
                    table.add_row(perms, size, date, name)
            
            return table
        except ImportError:
            # If rich is not available, return plain content
            return content
    
    def _format_file_content(self, content, file_path=""):
        """Format file content with syntax highlighting."""
        try:
            from rich.syntax import Syntax
            
            # Detect language
            lang = "text"
            if file_path.endswith(".py") or ("import " in content and ("def " in content or "class " in content)):
                lang = "python"
            elif file_path.endswith(".json") or (content.startswith("{") and content.endswith("}")):
                lang = "json"
            elif file_path.endswith(".md"):
                lang = "markdown"
            elif file_path.endswith(".html") or ("<html" in content.lower() and "</html>" in content.lower()):
                lang = "html"
            elif file_path.endswith(".js"):
                lang = "javascript"
            elif file_path.endswith(".css"):
                lang = "css"
            elif file_path.endswith(".yml") or file_path.endswith(".yaml"):
                lang = "yaml"
            elif file_path.endswith(".sh"):
                lang = "bash"
            elif file_path.endswith(".txt"):
                lang = "text"
            
            # Create syntax highlighted content
            return Syntax(content, lang, theme="monokai", line_numbers=True, word_wrap=True)
        except ImportError:
            # If rich is not available, return plain content
            return content
    
    async def analyze_project(self, project_dir: str, callback: Optional[Callable[[str], None]] = None) -> str:
        """Analyze a project directory using tools to explore files.
        
        Args:
            project_dir: Path to the project directory
            callback: Optional callback for streaming output
            
        Returns:
            Analysis result as a string
        """
        try:
            from rich.console import Console
            from rich.panel import Panel
            console = Console()
            
            # Verify directory exists
            if not os.path.isdir(project_dir):
                error_msg = f"Directory '{project_dir}' does not exist"
                logger.error(error_msg)
                if callback:
                    callback(f"[bold red]❌ {error_msg}[/bold red]")
                return error_msg
            
            # Configure tools
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "list_directory",
                        "description": "List files and folders in a directory",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "directory": {
                                    "type": "string",
                                    "description": "Path to the directory to list"
                                }
                            }
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "description": "Read the contents of a file",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "file_path": {
                                    "type": "string",
                                    "description": "Path to the file to read"
                                }
                            },
                            "required": ["file_path"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "execute_command",
                        "description": "Execute a shell command to navigate or explore the filesystem",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "commands": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "List of shell commands to execute"
                                }
                            },
                            "required": ["commands"]
                        }
                    }
                }
            ]
            
            # Check if model supports tools
            try:
                # Verify if the model supports tools with a simple test
                test_response = chat(
                    model=self.model,
                    messages=[{"role": "user", "content": "Test tools support"}],
                    tools=tools,
                    stream=False
                )
            except Exception as e:
                if "does not support tools" in str(e) or "400" in str(e):
                    error_msg = f"Model '{self.model}' does not support function calling/tools. Please use a compatible model like 'qwen3:8b' or 'codellama:7b'."
                    logger.error(error_msg)
                    if callback:
                        callback(f"[bold red]❌ {error_msg}[/bold red]\n")
                    return error_msg
                    
            # Initial message
            if callback:
                callback(Panel(
                    "[bold blue]Starting project analysis...[/bold blue]",
                    title="🔍 Project Explorer",
                    border_style="blue"
                ))
            
            # Initial message for exploration
            prompt = f"""
            I need you to thoroughly analyze the project in this directory: {project_dir}
            
            Your first steps should be:
            1. List the directory structure to understand the project organization
            2. Find and READ the main Python files (especially app/main.py if it exists)
            3. Read requirements.txt or package.json if they exist
            
            Use the available tools to explore directories and read files.
            DO NOT make assumptions about file contents - READ key files first.
            
            After exploring and reading key files, determine the project type and suggest
            the optimal deployment command for AtlasServer.
            """
            
            # Configure messages
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            # Maximum iterations
            max_iterations = 10
            
            # Main analysis loop
            for iteration in range(max_iterations):
                if callback:
                    callback(Panel(
                        f"[bold]Phase {iteration+1}[/bold]: Scanning project structure and files",
                        title="🔄 Analysis Progress",
                        border_style="cyan"
                    ))
                
                # Call the model (always without streaming for tools)
                response = chat(
                    model=self.model,
                    messages=messages,
                    tools=tools,
                    stream=False  # Important: use False for tools
                )
                
                # If there are no tool calls, it's the final response
                if 'tool_calls' not in response.get('message', {}):
                    final_response = response['message'].get('content', "")
                    # Filter <think> tags
                    final_response = re.sub(r'<think>[\s\S]*?</think>', '', final_response).strip()
                    
                    if callback:
                        # Try to format as JSON if possible
                        try:
                            if final_response.startswith("{") and "type" in final_response:
                                from rich.syntax import Syntax
                                data = json.loads(final_response)
                                json_str = json.dumps(data, indent=2)
                                callback(Panel(
                                    Syntax(json_str, "json", theme="monokai"),
                                    title="[bold green]🔍 Analysis Complete[/bold green]",
                                    border_style="green"
                                ))
                            else:
                                callback(f"\n[bold green]Analysis Complete[/bold green]\n{final_response}\n")
                        except:
                            callback(f"\n[bold green]Analysis Complete[/bold green]\n{final_response}\n")
                    
                    return final_response
                
                # Process tool calls
                tool_calls = response['message'].get('tool_calls', [])
                if not tool_calls:
                    break  # Exit if there are no tools to call
                
                # Add assistant message
                messages.append(response['message'])
                
                # Process each tool
                for tool_call in tool_calls:
                    tool_name = tool_call.get('function', {}).get('name')
                    tool_id = tool_call.get('id')
                    
                    # Parse arguments
                    try:
                        arguments_value = tool_call.get('function', {}).get('arguments', '{}')
                        if isinstance(arguments_value, dict):
                        # If it's already a dictionary, use it directly
                            arguments = arguments_value
                        else:
                        # If it's a string, try to parse it
                            arguments = json.loads(arguments_value)
                    except json.JSONDecodeError:
                        error_msg = f"Error parsing tool arguments: {arguments_value}"
                        logger.error(error_msg)
                        
                        # Add error as tool result
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_id,
                            "name": tool_name,
                            "content": error_msg
                        })
                        continue
                    
                    if callback:
                        tool_info = json.dumps(arguments, indent=2)
                        callback(Panel(
                            f"[bold yellow]Running:[/bold yellow] {tool_name}\n[dim]{tool_info}[/dim]",
                            title="🛠️ Tool Execution",
                            border_style="yellow"
                        ))
                    
                    # Execute tool
                    try:
                        result = await self._execute_tool(tool_name, arguments, project_dir)
                        
                        # Format results for display
                        if callback:
                            if tool_name == "list_directory":
                                # Format directory listing as a table
                                dir_table = self._format_directory_output(result)
                                callback(Panel(
                                    dir_table,
                                    title="📂 Directory Contents",
                                    border_style="blue"
                                ))
                            elif tool_name == "read_file":
                                # Format file content with syntax highlighting
                                file_path = arguments.get("file_path", "")
                                file_content = self._format_file_content(result, file_path)
                                callback(Panel(
                                    file_content,
                                    title=f"📑 File Content: {os.path.basename(file_path)}",
                                    border_style="green"
                                ))
                            else:
                                # Generic result display
                                max_display = 500
                                if len(result) > max_display:
                                    display_result = result[:max_display] + "... [truncated]"
                                else:
                                    display_result = result
                                callback(Panel(
                                    display_result,
                                    title="🔹 Command Result",
                                    border_style="cyan"
                                ))
                        
                        # Add result to the message
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_id,
                            "name": tool_name,
                            "content": result
                        })
                    except Exception as e:
                        error_msg = f"Error executing tool: {str(e)}"
                        logger.error(error_msg)
                        
                        # Add error as result
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_id,
                            "name": tool_name,
                            "content": error_msg
                        })
                        
                        if callback:
                            callback(Panel(
                                f"[bold red]{error_msg}[/bold red]",
                                title="❌ Error",
                                border_style="red"
                            ))
            
            # If we get here, we've reached the maximum iterations
            return "Analysis has exceeded the maximum number of iterations without reaching a conclusion."
            
        except Exception as e:
            error_msg = f"Error during analysis: {str(e)}"
            logger.error(error_msg)
            if callback:
                callback(f"[bold red]❌ {error_msg}[/bold red]")
            return error_msg
            
    async def _execute_tool(self, tool_name: str, arguments: Dict[str, Any], working_dir: str) -> str:
        """Execute the specified tool with the given arguments."""
        try:
            if tool_name == "list_directory":
                directory = arguments.get("directory", working_dir)
            
                # Convert relative paths to absolute
                if not os.path.isabs(directory):
                    directory = os.path.join(working_dir, directory)
            
                # Verify directory exists
                if not os.path.isdir(directory):
                    return f"Error: Directory '{directory}' does not exist"
                
                return list_directory(directory)
        
            elif tool_name == "read_file":
                file_path = arguments.get("file_path")
                if not file_path:
                    return "Error: No file specified to read"
            
                # Convert relative paths to absolute
                if not os.path.isabs(file_path):
                    file_path = os.path.join(working_dir, file_path)
            
                # Verify file exists
                if not os.path.isfile(file_path):
                    return f"Error: File '{file_path}' does not exist"
                
                return read_file(file_path)
        
            elif tool_name == "execute_command":
                commands = arguments.get("commands", [])
                if not commands:
                    return "Error: No commands specified to execute"
            
                from atlasai.tools import execute_command
                # Execute in the project directory
                result = execute_command(commands)
                return result
        
            else:
                return f"Error: Unknown tool '{tool_name}'"
    
        except Exception as e:
            logger.error(f"Error executing tool '{tool_name}': {str(e)}")
            return f"Error executing tool '{tool_name}': {str(e)}"