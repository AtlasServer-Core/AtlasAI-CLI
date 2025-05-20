# atlasai/ai/general_agent.py
import os
import logging
import json
import re
from typing import Dict, Any, Optional, Callable, List

from ollama import chat
from openai import OpenAI

from atlasai.tools import get_os, search, list_directory, read_file, execute_command
from atlasai.ai.prompts import BASE_PROMPTS

logger = logging.getLogger(__name__)

class GeneralAgent:
    """General AI agent to process user queries with tools."""
    
    def __init__(self, model="qwen3:8b", provider="ollama", api_key=None, stream=False, language="en"):
        """Initialize the general AI agent."""
        assert model, "Model cannot be empty"
        assert provider in ["ollama", "openai"], "Provider must be 'ollama' or 'openai'"
        if provider == "openai":
            assert api_key, "API key is required to use OpenAI"
        
        self.model = model
        self.provider = provider
        self.api_key = api_key
        self.stream = stream
        self.language = language
        general_prompt = BASE_PROMPTS.get_general_agent_prompt(language)
        advanced_prompt = BASE_PROMPTS.get_advanced_agent_prompt(language)
        if language == "es":
            general_prompt = general_prompt.split("IMPORTANT:")[0].strip()
    
        self.system_prompt = f"{general_prompt}\n\n{advanced_prompt}"

        
        # Initialize OpenAI client only when needed
        if self.provider == "openai":
            self.client = OpenAI(api_key=self.api_key)
        

    def _get_available_tools(self) -> List[Dict[str, Any]]:
        """Get the list of available tools."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_os",
                    "description": "Get information about the current operating system",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                        "additionalProperties": False
                    },
                    "strict": True
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search the web for information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query"
                            }
                        },
                        "required": ["query"],
                        "additionalProperties": False
                    },
                    "strict": True
                }
            },
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
                        },
                        "required": ["directory"],
                        "additionalProperties": False
                    },
                    "strict": True
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
                        "required": ["file_path"],
                        "additionalProperties": False
                    },
                    "strict": True
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
                        "required": ["commands"],
                        "additionalProperties": False
                    },
                    "strict": True
                }
            },
            {
                    "type": "function",
                    "function": {
                        "name": "write_file",
                        "description": "Create a new file or overwrite an existing one",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "file_path": {
                                    "type": "string",
                                    "description": "Path to the file to be written"
                                },
                                "content": {
                                    "type": "string",
                                    "description": "Content to write to the file"
                                },
                                "overwrite": {
                                    "type": "boolean",
                                    "description": "If True, will overwrite an existing file"
                                }
                            },
                            "required": ["file_path", "content", "overwrite"],
                            "additionalProperties": False
                        },
                        "strict": True
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "edit_file",
                        "description": "Edit parts of an existing file",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "file_path": {
                                    "type": "string",
                                    "description": "Path to the file to be edited"
                                },
                                "search_text": {
                                    "type": "string",
                                    "description": "Text to search for"
                                },
                                "replace_text": {
                                    "type": "string",
                                    "description": "Replacement text"
                                },
                                "regex": {
                                    "type": "boolean",
                                    "description": "If True, search_text will be interpreted as a regular expression"
                                }
                            },
                            "required": ["file_path", "search_text", "replace_text", "regex"],
                            "additionalProperties": False
                        },
                        "strict": True
                    }
                }
        ]
    
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
    
    async def process_query(self, query: str, callback: Optional[Callable[[str], None]] = None) -> str:
        """Process a general user query.
        
        Args:
            query: User query
            callback: Function to call with each chunk in streaming mode
            
        Returns:
            Complete response
        """
        try:
            from rich.console import Console
            from rich.panel import Panel
            from rich.markdown import Markdown
            console = Console()
        
            # Verify that the query is not empty
            assert query and query.strip(), "Query cannot be empty"
            
            # Get available tools
            tools = self._get_available_tools()
            
            # Check if model supports tools based on provider
            try:
                if self.provider == "openai":
                    # Validate client exists
                    assert hasattr(self, 'client') and self.client is not None, "OpenAI client not initialized"
                    
                    # Test function support
                    test_response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": "Test tools support"}],
                        tools=tools
                    )
                    
                    # Verify valid response
                    assert hasattr(test_response, 'choices') and len(test_response.choices) > 0, "Invalid response from OpenAI API"
                    
                else:
                    # Ollama implementation
                    test_response = chat(
                        model=self.model,
                        messages=[{"role": "user", "content": "Test tools support"}],
                        tools=tools,
                        stream=False
                    )
            except Exception as e:
                error_msg = f"Error checking tool support: {str(e)}"
                logger.error(error_msg)
                if "does not support tools" in str(e) or "400" in str(e):
                    error_msg = f"Model '{self.model}' does not support function calling/tools. Please use a compatible model."
                logger.error(error_msg)
                if callback:
                    callback(f"[bold red]❌ {error_msg}[/bold red]\n")
                return error_msg
            
            # Initial message
            if callback:
                callback(Panel(
                    "[bold blue]Starting query processing...[/bold blue]",
                    title="🔍 AI Assistant",
                    border_style="blue"
                ))
            
            # Configure messages
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": query}
            ]
            
            # Maximum iterations to prevent infinite loops
            max_iterations = 20
            
            # Main processing loop
            for iteration in range(max_iterations):
                if callback:
                    callback(Panel(
                        f"[bold]Phase {iteration+1}[/bold]: Processing query",
                        title="🔄 Progress",
                        border_style="cyan"
                    ))
                
                # Call the model based on provider
                if self.provider == "openai":
                    # Use chat.completions API for OpenAI function calling
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        tools=tools
                    )
                    
                    # Extract assistant message content
                    assistant_message = response.choices[0].message
                    
                else:
                    # Ollama implementation
                    response = chat(
                        model=self.model,
                        messages=messages,
                        tools=tools,
                        stream=False
                    )
                    assistant_message = response['message']
                
                # Check if the assistant message has tool calls
                has_tool_calls = False
                if self.provider == "openai":
                    has_tool_calls = (hasattr(response.choices[0].message, 'tool_calls') and 
                                     response.choices[0].message.tool_calls and 
                                     len(response.choices[0].message.tool_calls) > 0)
                else:
                    has_tool_calls = ("tool_calls" in assistant_message and 
                                     assistant_message["tool_calls"] and 
                                     len(assistant_message["tool_calls"]) > 0)
                
                # If there are no tool calls, it's the final response
                if not has_tool_calls:
                    if self.provider == "openai":
                        final_response = getattr(assistant_message, "content", "")
                    else:
                        final_response = assistant_message.get("content", "")
                    
                    # Filter <think> tags
                    final_response = re.sub(r'<think>[\s\S]*?</think>', '', final_response).strip()
                
                    if callback:
                        # Try to format as JSON if possible
                        try:
                            from rich.markdown import Markdown
                            callback(Panel(
                            Markdown(final_response),
                            title="[bold green]🔍 Results[/bold green]",
                            border_style="green"
                            ))
                        except ImportError:
                        # Fallback if rich.markdown is not available
                            callback(Panel(
                                final_response,
                                title="[bold green]🔍 Results[/bold green]",
                                border_style="green"
                            ))
    
                    return final_response
                
                # Add assistant message to history
                messages.append(assistant_message)
                
                # Process tool calls
                if self.provider == "openai":
                    tool_calls = response.choices[0].message.tool_calls
                else:
                    tool_calls = assistant_message.get("tool_calls", [])
                
                if not tool_calls:
                    break  # Exit if there are no tools to call
                
                # Process each tool call
                for tool_call in tool_calls:
                    # Extract tool information based on provider
                    if self.provider == "openai":
                        tool_id = tool_call.id
                        tool_name = tool_call.function.name
                        arguments_value = tool_call.function.arguments
                    else:
                        tool_id = tool_call.get('id')
                        tool_name = tool_call.get('function', {}).get('name')
                        arguments_value = tool_call.get('function', {}).get('arguments', '{}')
                    
                    # Validate tool name
                    valid_tools = ["get_os", "search", "list_directory", "read_file", "execute_command", "write_file", "edit_file"]
                    if tool_name not in valid_tools:
                        error_msg = f"Unknown tool: '{tool_name}'"
                        logger.error(error_msg)
                        
                        # Add error as tool result
                        if self.provider == "openai":
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_id,
                                "content": error_msg
                            })
                        else:
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_id,
                                "name": tool_name,
                                "content": error_msg
                            })
                        continue
                    
                    # Parse arguments
                    try:
                        if isinstance(arguments_value, dict):
                            arguments = arguments_value
                        else:
                            arguments = json.loads(arguments_value)
                            
                        # Validate arguments based on tool
                        if tool_name == "get_os":
                            # No arguments required
                            pass
                        elif tool_name == "search":
                            assert "query" in arguments, "Missing 'query' argument"
                            assert isinstance(arguments["query"], str), "The 'query' argument must be a string"
                        elif tool_name == "list_directory":
                            assert "directory" in arguments, "Missing 'directory' argument"
                            assert isinstance(arguments["directory"], str), "The 'directory' argument must be a string"
                        elif tool_name == "read_file":
                            assert "file_path" in arguments, "Missing 'file_path' argument"
                            assert isinstance(arguments["file_path"], str), "The 'file_path' argument must be a string"
                        elif tool_name == "execute_command":
                            assert "commands" in arguments, "Missing 'commands' argument"
                            assert isinstance(arguments["commands"], list), "The 'commands' argument must be a list"
                            assert all(isinstance(cmd, str) for cmd in arguments["commands"]), "All commands must be strings"
                            
                    except json.JSONDecodeError:
                        error_msg = f"Error parsing tool arguments: {arguments_value}"
                        logger.error(error_msg)
                        
                        # Add error as tool result
                        if self.provider == "openai":
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_id,
                                "content": error_msg
                            })
                        else:
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_id,
                                "name": tool_name,
                                "content": error_msg
                            })
                        continue
                    except AssertionError as e:
                        error_msg = f"Error in arguments: {str(e)}"
                        logger.error(error_msg)
                        
                        # Add error as tool result
                        if self.provider == "openai":
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_id,
                                "content": error_msg
                            })
                        else:
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_id,
                                "name": tool_name,
                                "content": error_msg
                            })
                        continue
                    
                    if callback:
                        tool_info = json.dumps(arguments, indent=2) if arguments else "{}"
                        callback(Panel(
                            f"[bold yellow]Running:[/bold yellow] {tool_name}\n[dim]{tool_info}[/dim]",
                            title="🛠️ Tool Execution",
                            border_style="yellow"
                        ))
                    
                    # Execute tool
                    try:
                        result = await self._execute_tool(tool_name, arguments)
                        
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
                            elif tool_name == "get_os":
                                callback(Panel(
                                    result,
                                    title="💻 Operating System Information",
                                    border_style="magenta"
                                ))
                            elif tool_name == "search":
                                max_display = 500
                                if len(result) > max_display:
                                    display_result = result[:max_display] + "... [truncated]"
                                else:
                                    display_result = result
                                callback(Panel(
                                    display_result,
                                    title="🔎 Search Results",
                                    border_style="green"
                                ))
                            elif tool_name == "write_file" or tool_name == "append_file":
                                callback(Panel(
                                    result,
                                    title=f"✏️ File Operation: {os.path.basename(arguments.get('file_path', ''))}",
                                    border_style="green"
                                ))
                            elif tool_name == "edit_file":
                                callback(Panel(
                                result,
                                title=f"🔄 Edited File: {os.path.basename(arguments.get('file_path', ''))}",
                                border_style="blue"
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
                        
                        # Add result to message history - correct format for OpenAI
                        if self.provider == "openai":
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_id,
                                "content": result
                            })
                        else:
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
                        if self.provider == "openai":
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_id,
                                "content": error_msg
                            })
                        else:
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
            
            # If we reach here, we've reached the maximum iterations
            return "Analysis has exceeded the maximum number of iterations without reaching a conclusion."
            
        except Exception as e:
            error_msg = f"Error during query processing: {str(e)}"
            logger.error(error_msg)
            if callback:
                callback(f"[bold red]❌ {error_msg}[/bold red]")
            return error_msg
    
    async def _execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Execute the specified tool with the given arguments."""
        try:
            if tool_name == "get_os":
                return get_os()
                
            elif tool_name == "search":
                query = arguments.get("query", "")
                if not query:
                    return "Error: No search query specified"
                
                return search(query)
                
            elif tool_name == "list_directory":
                directory = arguments.get("directory", ".")
                
                # Convert relative paths to absolute
                if not os.path.isabs(directory):
                    directory = os.path.abspath(directory)
                
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
                    file_path = os.path.abspath(file_path)
                
                # Verify file exists
                if not os.path.isfile(file_path):
                    return f"Error: File '{file_path}' does not exist"
                
                return read_file(file_path)
                
            elif tool_name == "execute_command":
                commands = arguments.get("commands", [])
                if not commands:
                    return "Error: No commands specified to execute"
                
                return execute_command(commands)

            elif tool_name == "write_file":
                file_path = arguments.get("file_path")
                content = arguments.get("content", "")
                overwrite = arguments.get("overwrite", False)
        
                if not file_path:
                    return "Error: No se especificó ningún archivo para escribir"
        
                # Convertir rutas relativas a absolutas
                if not os.path.isabs(file_path):
                    file_path = os.path.abspath(file_path)
        
                from atlasai.tools import write_file
                return write_file(file_path, content, overwrite)

            elif tool_name == "edit_file":
                file_path = arguments.get("file_path")
                search_text = arguments.get("search_text")
                replace_text = arguments.get("replace_text", "")
                regex = arguments.get("regex", False)
        
                if not file_path:
                    return "Error: No se especificó ningún archivo para editar"
        
                if search_text is None:
                    return "Error: No se especificó el texto a buscar"
        
                # Convertir rutas relativas a absolutas
                if not os.path.isabs(file_path):
                    file_path = os.path.abspath(file_path)
        
                # Verificar que el archivo existe
                if not os.path.isfile(file_path):
                    return f"Error: El archivo '{file_path}' no existe"
        
                from atlasai.tools import edit_file
                return edit_file(file_path, search_text, replace_text, regex)
                
            else:
                return f"Error: Unknown tool '{tool_name}'"
        
        except Exception as e:
            logger.error(f"Error executing tool '{tool_name}': {str(e)}")
            return f"Error executing tool '{tool_name}': {str(e)}"