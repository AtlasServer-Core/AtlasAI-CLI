# atlasai/ai/interactive_agent.py
import asyncio
import os
from typing import Optional, List, Dict, Any
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.markdown import Markdown
from rich.syntax import Syntax

from atlasai.ai.general_agent import GeneralAgent
from atlasai.ai.prompts import BASE_PROMPTS

class InteractiveAgent:
    """Interactive CLI agent that maintains conversation context."""
    
    def __init__(self, model="qwen3:8b", provider="ollama", api_key=None, language="en", prompt_level="general"):
        """Initialize the interactive agent.
        
        Args:
            model: AI model to use
            provider: Provider (ollama/openai)
            api_key: API key (required for OpenAI)
            language: Language (en/es)
            prompt_level: Prompt level (general/advanced/combined)
        """
        self.console = Console()
        self.model = model
        self.provider = provider
        self.api_key = api_key
        self.language = language
        self.prompt_level = prompt_level
        
        # Create custom system prompt based on prompt_level
        system_prompt = self._get_combined_prompt(prompt_level, language)
        
        # Create the underlying general agent with custom prompt
        self.agent = GeneralAgent(
            model=model,
            provider=provider,
            api_key=api_key,
            stream=True,  # Always use streaming for better experience
            language=language
        )
        
        # Override agent's system prompt
        self.agent.system_prompt = system_prompt
        
        # Message history for context
        self.messages = []
        
        # Session information
        self.session_dir = os.path.expanduser("~/.atlasai/chat_sessions")
        self.current_dir = os.getcwd()
    
    def _get_combined_prompt(self, prompt_level, language):
        """Get the appropriate prompt based on level."""
        if prompt_level == "general":
            return BASE_PROMPTS.get_general_agent_prompt(language)
        elif prompt_level == "advanced":
            return BASE_PROMPTS.get_advanced_agent_prompt(language)
        elif prompt_level == "combined":
            general_prompt = BASE_PROMPTS.get_general_agent_prompt(language)
            advanced_prompt = BASE_PROMPTS.get_advanced_agent_prompt(language)
            
            # Eliminate duplicate language instructions
            if language == "es":
                general_prompt = general_prompt.split("IMPORTANT:")[0].strip()
            
            return f"{general_prompt}\n\n{advanced_prompt}"
        else:
            return BASE_PROMPTS.get_general_agent_prompt(language)
    
    def _display_welcome(self):
        """Display welcome message."""
        self.console.print(Panel(
            "[bold blue]Welcome to AtlasAI Interactive Mode![/bold blue]\n"
            "Ask me anything about your system, projects, or development needs.\n"
            "Type [bold green]'exit'[/bold green] or [bold green]'quit'[/bold green] to end the session.",
            title="🤖 AtlasAI Interactive",
            border_style="blue"
        ))
    
    def _collect_chunks(self, chunk):
        """Callback for streaming chunks."""
        self.console.print(chunk, end="", highlight=False)
    
    async def start_interactive_session(self):
        """Start an interactive session with the agent."""
        self._display_welcome()
        
        # Create session directory if it doesn't exist
        os.makedirs(self.session_dir, exist_ok=True)
        
        while True:
            # Get user input
            user_input = Prompt.ask("\n[bold cyan]You[/]")
            
            # Check for exit command
            if user_input.lower() in ['exit', 'quit']:
                self.console.print("[bold green]Goodbye! Session ended.[/]")
                break
            
            # Display processing indicator
            self.console.print("\n[bold magenta]AtlasAI[/]")
            
            # Process the query with the agent
            full_response = []
            
            def collect_response(chunk):
                self._collect_chunks(chunk)
                full_response.append(chunk)
            
            response = await self.agent.process_query(
                user_input,
                callback=collect_response
            )
            
            # If response is empty but we collected chunks, use those
            if not response and full_response:
                response = ''.join(full_response)
            
            # Store the interaction in history
            self.messages.append({"role": "user", "content": user_input})
            self.messages.append({"role": "assistant", "content": response})
            
            # Add a separator for readability
            self.console.print("\n" + "-" * 50)
    
    def save_session(self):
        """Save the current session to a file."""
        if not self.messages:
            return
            
        import time
        import json
        
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        session_file = os.path.join(self.session_dir, f"chat_session_{timestamp}.json")
        
        try:
            with open(session_file, "w") as f:
                json.dump({
                    "timestamp": timestamp,
                    "model": self.model,
                    "provider": self.provider,
                    "language": self.language,
                    "prompt_level": self.prompt_level,
                    "messages": self.messages
                }, f, indent=2)
            
            self.console.print(f"[green]Session saved to {session_file}[/]")
        except Exception as e:
            self.console.print(f"[red]Failed to save session: {str(e)}[/]")

async def start_interactive_cli(model="qwen3:8b", provider="ollama", api_key=None, language="en", prompt_level="general"):
    """Start an interactive CLI session with AtlasAI."""
    interactive_agent = InteractiveAgent(
        model=model,
        provider=provider,
        api_key=api_key,
        language=language,
        prompt_level=prompt_level
    )
    
    try:
        await interactive_agent.start_interactive_session()
    except KeyboardInterrupt:
        print("\nSession interrupted")
    finally:
        # Save the session when done
        interactive_agent.save_session()