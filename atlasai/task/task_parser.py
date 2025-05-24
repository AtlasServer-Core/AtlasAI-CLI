# atlasai/task/task_parser.py
import re
import os
import yaml
from typing import Dict, List, Any
import networkx as nx

class TaskDefinition:
    """Represents an individual task."""
    def __init__(self, id: str, title: str, description: str, 
                 depends_on: List[str], commands: List[str]):
        self.id = id
        self.title = title
        self.description = description
        self.depends_on = depends_on
        self.commands = commands
        self.is_completed = False

class TaskGraph:
    """Manages a graph of tasks with their dependencies."""
    
    def __init__(self):
        self.tasks: Dict[str, TaskDefinition] = {}
        self.metadata: Dict[str, Any] = {}
        self.graph = nx.DiGraph()
        
    def add_task(self, task: TaskDefinition):
        """Adds a task to the graph."""
        self.tasks[task.id] = task
        self.graph.add_node(task.id)
        
        # Add dependencies
        for dep_id in task.depends_on:
            if dep_id:  # Ignore empty dependencies
                self.graph.add_edge(dep_id, task.id)
    
    def get_next_tasks(self) -> List[str]:
        """Gets the next tasks that can be executed."""
        available_tasks = []
        
        for task_id, task in self.tasks.items():
            if task.is_completed:
                continue
                
            # Check if all dependencies are completed
            deps_completed = all(
                self.tasks.get(dep_id).is_completed 
                for dep_id in task.depends_on 
                if dep_id and dep_id in self.tasks
            )
            
            if deps_completed:
                available_tasks.append(task_id)
                
        return available_tasks
    
    def mark_as_completed(self, task_id: str):
        """Marks a task as completed."""
        if task_id in self.tasks:
            self.tasks[task_id].is_completed = True
    
    def all_tasks_completed(self) -> bool:
        """Checks if all tasks are completed."""
        return all(task.is_completed for task in self.tasks.values())
    
    def get_execution_order(self) -> List[str]:
        """Gets the execution order of tasks."""
        try:
            # Check for cycles
            cycles = list(nx.simple_cycles(self.graph))
            if cycles:
                raise ValueError(f"Cycles detected in dependencies: {cycles}")
            
            # Get topological order
            return list(nx.topological_sort(self.graph))
        except nx.NetworkXUnfeasible:
            raise ValueError("Task graph contains cycles and cannot be sorted")

def parse_task_file(file_path: str) -> TaskGraph:
    """Parses a task file and creates a task graph."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Task file not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Create task graph
    task_graph = TaskGraph()
    
    # Extract metadata
    metadata_match = re.search(r'## Metadata\s*\n(.*?)(?=##)', content, re.DOTALL)
    if metadata_match:
        metadata_text = metadata_match.group(1)
        # Convert metadata lines to YAML format
        yaml_text = "\n".join([line.strip().replace("- ", "") for line in metadata_text.split('\n') if line.strip()])
        try:
            task_graph.metadata = yaml.safe_load(yaml_text)
        except yaml.YAMLError:
            # If YAML loading fails, use simplified approach
            metadata_lines = [line.strip() for line in metadata_text.split('\n') if line.strip()]
            for line in metadata_lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    task_graph.metadata[key.strip()] = value.strip()
    
    # Extract tasks
    task_pattern = r'\[TASK id="([^"]+)" depends="([^"]*)"\](.*?)(?=\[\s*TASK|$)'
    task_matches = re.finditer(task_pattern, content, re.DOTALL)
    
    for match in task_matches:
        task_id = match.group(1)
        depends_str = match.group(2)
        task_content = match.group(3).strip()
        
        # Split dependencies
        depends_on = [dep.strip() for dep in depends_str.split(',') if dep.strip()]
        
        # Extract title
        title_match = re.search(r'###\s*(.*?)$', task_content, re.MULTILINE)
        title = title_match.group(1).strip() if title_match else "Task " + task_id
        
        # Extract description (everything before the first code block)
        desc_text = re.sub(r'###\s*.*?\n', '', task_content, 1)
        description = re.split(r'```', desc_text, 1)[0].strip()
        
        # Extract commands
        commands = []
        code_blocks = re.finditer(r'```(?:\w+)?\s*(.*?)```', task_content, re.DOTALL)
        for code_match in code_blocks:
            cmd_lines = code_match.group(1).strip().split('\n')
            commands.extend([cmd.strip() for cmd in cmd_lines if cmd.strip()])
        
        # Create task
        task = TaskDefinition(
            id=task_id,
            title=title,
            description=description,
            depends_on=depends_on,
            commands=commands
        )
        
        # Add to graph
        task_graph.add_task(task)
    
    return task_graph