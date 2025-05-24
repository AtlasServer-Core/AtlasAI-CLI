# atlasai/task/task_executor.py
import os
import logging
from typing import Optional, Callable
from rich.console import Console
from rich.panel import Panel
from atlasai.task.task_parser import TaskGraph
from atlasai.ai.general_agent import GeneralAgent
from atlasai.tools import execute_command
import datetime

logger = logging.getLogger(__name__)
console = Console()

class TaskExecutor:
    """Executes tasks defined in a task graph."""
    
    def __init__(self, task_graph: TaskGraph, 
                 working_dir: str,
                 model: str = "gpt-4.1", 
                 provider: str = "openai",
                 api_key: Optional[str] = None,
                 language: str = "en",
                 prompt_level: str = "combined",
                 verify_commands: bool = True):
        self.task_graph = task_graph
        self.working_dir = os.path.abspath(working_dir)
        self.model = model
        self.provider = provider
        self.api_key = api_key
        self.language = language
        self.prompt_level = prompt_level
        self.verify_commands = verify_commands
        
    async def execute_tasks(self, callback: Optional[Callable[[str], None]] = None) -> bool:
        """Executes all tasks in the correct order."""
        if not self.task_graph.tasks:
            console.print("[bold red]No tasks found to execute[/]")
            return False
            
        # Initialize agent
        agent = GeneralAgent(
            model=self.model,
            provider=self.provider,
            api_key=self.api_key,
            stream=True,
            language=self.language
        )
        
        # Show execution order
        try:
            execution_order = self.task_graph.get_execution_order()
            
            console.print(Panel(
                f"[bold cyan]Executing {len(execution_order)} tasks[/]\n"
                f"Execution order: [blue]{' → '.join(execution_order)}[/]",
                title="[bold blue]🔄 Execution Plan[/]",
                border_style="blue"
            ))
            
            # Execute tasks
            while not self.task_graph.all_tasks_completed():
                next_tasks = self.task_graph.get_next_tasks()
                
                if not next_tasks:
                    console.print("[bold red]❌ No more tasks to execute, but tasks remain pending[/]")
                    return False
                
                for task_id in next_tasks:
                    task = self.task_graph.tasks[task_id]
                    
                    console.print(Panel(
                        f"[bold cyan]{task.title}[/]\n\n{task.description}",
                        title=f"[bold blue]🔄 Task {task_id}[/]",
                        border_style="blue"
                    ))
                    
                    # Execute commands
                    for command in task.commands:
                        if command.startswith("atlasai"):
                            # AtlasAI-specific commands
                            await self._execute_atlasai_command(command, agent, callback)
                        else:
                            # System commands - using existing execute_command function
                            await self._execute_system_command(command, callback)
                    
                    # Mark as completed
                    self.task_graph.mark_as_completed(task_id)
                    console.print(f"[bold green]✅ Task {task_id} completed[/]")
            
            # All tasks completed
            console.print(Panel(
                "[bold green]All tasks have been completed successfully![/]",
                title="[bold green]✅ Process Completed[/]",
                border_style="green"
            ))
            
            return True
            
        except Exception as e:
            console.print(f"[bold red]❌ Error during execution:[/] {str(e)}")
            logger.error(f"Error during execution: {str(e)}", exc_info=True)
            return False
    
    async def _execute_atlasai_command(self, command: str, agent: GeneralAgent, 
                                      callback: Optional[Callable[[str], None]] = None) -> None:
        """Executes an AtlasAI-specific command."""
        console.print(f"[bold blue]Executing AtlasAI command:[/] [yellow]{command}[/]")
        
        # Extract query for agent
        parts = command.split(' ', 2)
        if len(parts) < 3 or parts[1] != '--query':
            console.print("[bold red]❌ Invalid AtlasAI command format[/]")
            return
            
        query = parts[2].strip('"\'')
        
        # Execute query
        full_response = []
        
        def collect_response(chunk):
            if callback:
                callback(chunk)
            full_response.append(chunk)
            console.print(chunk, end="", highlight=False)
            
        await agent.process_query(query, callback=collect_response)
        console.print("\n")
    
    async def _execute_system_command(self, command: str, 
                                     callback: Optional[Callable[[str], None]] = None) -> None:
        """Executes a system command using the existing execute_command function."""
        console.print(f"[bold blue]Executing command:[/] [yellow]{command}[/]")
        
        # Change to working directory
        prev_dir = os.getcwd()
        os.chdir(self.working_dir)
        
        try:
            # Execute the command using existing functionality
            result = execute_command([command])
            
            if "Error" in result and "not allowed" in result:
                # Command not allowed due to security checks
                console.print(Panel(
                    result,
                    title="[bold yellow]⚠️ Command Restricted[/]",
                    border_style="yellow"
                ))
                
                # Ask user if they want to execute anyway
                if self.verify_commands:
                    from rich.prompt import Confirm
                    if Confirm.ask("Do you want to execute this command anyway?"):
                        # This would require additional implementation
                        # For now, just inform the user
                        console.print("[bold yellow]⚠️ Manual override not implemented yet[/]")
            else:
                # Command executed successfully
                console.print(Panel(
                    result,
                    title="[bold green]✅ Command Output[/]",
                    border_style="green"
                ))
                
            if callback:
                callback(result)
                    
        except Exception as e:
            console.print(f"[bold red]❌ Error executing command:[/] {str(e)}")
            if callback:
                callback(f"Error: {str(e)}")
        finally:
            # Restore original directory
            os.chdir(prev_dir)

def generate_task_template(num_tasks=3):
    """Generate a task template with specified number of tasks."""
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    
    template = f"""# AtlasAI Task: My Project Workflow

## Metadata
- author: Your Name
- date: {today}
- priority: high
- allow_file_operations: true

## Description
This is a template task workflow. Edit this description to explain what this workflow will accomplish.
Replace the task definitions below with your actual tasks.

## Tasks
"""

    # Generate example tasks
    for i in range(1, num_tasks + 1):
        task_id = f"task{i}"
        depends = f"task{i-1}" if i > 1 else ""
        
        task = f"""
{i}. [TASK id="{task_id}" depends="{depends}"]
   ### Task {i} Title
   Description of task {i}. Explain what this task does and why it's important.
   
   ```bash
   # Replace with actual commands for this task
   echo "Executing task {i}"
   ```

"""
        template += task
    
    # Add a final task that depends on multiple previous tasks
    if num_tasks > 2:
        dependencies = ", ".join([f"task{i}" for i in range(1, num_tasks)])
        template += f"""
{num_tasks + 1}. [TASK id="final" depends="{dependencies}"]
   ### Final Task
   This task depends on all previous tasks and runs only when they are complete.
   
   ```bash
   # Replace with your final commands
   echo "All tasks completed, running final task"
   ```
"""
    
    # Add some helpful comments at the end
    template += """
## Notes
- Each task must have a unique ID
- The "depends" attribute lists the IDs of tasks that must complete before this task
- Multiple dependencies should be comma-separated
- Commands are executed in the order they appear
- Use atlasai commands for AI-powered operations
"""
    
    return template