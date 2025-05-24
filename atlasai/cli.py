import click
import os
import json
import re
import asyncio
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax
from rich.prompt import Confirm
from rich.markdown import Markdown

# Crear consola global
console = Console()

@click.group()
def cli():
    """AtlasAI CLI - AI-powered tools for AtlasServer deployments."""
    pass

@click.group()
def ai():
    pass

@ai.command("setup")
@click.option("--provider", type=click.Choice(["ollama", "openai"]), default="ollama", 
              help="AI provider (ollama or openai)")
@click.option("--model", default="qwen3:8b", help="Model to use (e.g.: qwen3:8b or gpt-4)")
@click.option("--api-key", help="API key (required for OpenAI)")
def ai_setup(provider, model, api_key):
    """Configure the AI service for CLI."""
    from atlasai.ai.ai_cli import AtlasServerAICLI
    ai_cli = AtlasServerAICLI()
    
    # Verificar API key para OpenAI
    if provider == "openai" and not api_key:
        console.print("[bold red]❌ API key is required for OpenAI[/]")
        return
        
    success = ai_cli.setup(provider, model, api_key)
    
    if success:
        config_info = f"[blue]{provider}[/] / [blue]{model}[/]"
        if provider == "openai":
            config_info += " / [blue]API key saved[/]"
        console.print(f"[bold green]✅ AI configuration saved:[/] {config_info}")
    else:
        console.print("[bold red]❌ Error saving AI configuration[/]")

@ai.command("suggest")
@click.argument("app_directory", type=click.Path(exists=True))
@click.option("--stream/--no-stream", default=True, help="Stream the AI response")
@click.option("--interactive/--no-interactive", default=True, 
              help="Use interactive file exploration")
@click.option("--debug/--no-debug", default=False, help="Show debug information")
@click.option("--language", type=click.Choice(["en", "es"]), default="en",
              help="Response language (English or Spanish)")
def ai_suggest_command(app_directory, stream, interactive, debug, language):
    """Suggest deployment commands for an application."""
    try:
        app_directory = os.path.abspath(app_directory)
        
        # Load AI configuration
        from atlasai.ai.ai_cli import AtlasServerAICLI
        ai_cli = AtlasServerAICLI()
        configured_model = ai_cli.ai_config.get("model", "codellama:7b")
        
        console.print(f"[bold cyan]🤖 Using AI model:[/] [blue]{configured_model}[/]")
        
        if ai_cli.ai_config.get("provider", "ollama") == "ollama":
            import requests
            try:
                with console.status("[bold blue]Connecting to Ollama server...[/]", spinner="dots"):
                    response = requests.get("http://localhost:11434/api/version", timeout=2)
        
                if response.status_code != 200:
                    console.print("[bold red]❌ Error: Could not connect to Ollama server[/]")
                    return
                else:
                    if debug:
                        console.print(Panel.fit(
                        f"Version: {response.json().get('version', 'unknown')}",
                        title="[bold green]✅ Connected to Ollama[/]",
                        border_style="green"
                    ))
            except Exception as e:
                console.print("[bold red]❌ Error: Ollama server is not running[/]")
                console.print(f"   [italic]{str(e)}[/]")
                console.print("   Run [bold]'ollama serve'[/] or ensure the Ollama service is running.")
                return
        elif ai_cli.ai_config.get("provider") == "openai":
            # Verificar que tenemos API key
            if not ai_cli.ai_config.get("api_key"):
                console.print("[bold red]❌ Error: No API key configured for OpenAI[/]")
                console.print("   Run [bold]'atlasai ai setup --provider openai --api-key YOUR_API_KEY'[/]")
                return
    
            console.print(Panel.fit(
                "Using OpenAI for AI services",
                title="[bold green]✅ OpenAI configured[/]",
                border_style="green"
            ))
        
        if interactive:
            # Use the simplified approach (without complex tools)
            from atlasai.ai.ai_agent import AgentCLI
            agent = AgentCLI(
                model=configured_model, 
                provider=ai_cli.ai_config.get("provider", "ollama"),
                api_key=ai_cli.ai_config.get("api_key"),
                stream=stream, 
                language=language
            )
            
            console.print(Panel(
                f"[cyan]Directory:[/] [bold]{app_directory}[/]",
                title="[bold blue]🔍 Project Analysis[/]",
                border_style="blue"
            ))
            
            # Define callback for streaming if needed
            if stream:
                full_response_text = []
                
                def collect_response(chunk):
                    full_response_text.append(chunk)
                    console.print(chunk, end="", highlight=False)
                
                # Execute with streaming
                response = asyncio.run(agent.analyze_project(
                    app_directory, 
                    callback=collect_response
                ))
                
                # If response is empty but we have text, use it
                if not response and full_response_text:
                    response = ''.join(full_response_text)
                console.print("\n")
            else:
                # Execute without streaming
                with console.status("[bold blue]Analyzing project structure...[/]", spinner="dots"):
                    response = asyncio.run(agent.analyze_project(app_directory))
            
            # Show complete response in debug mode
            if debug:
                console.print("\n[bold blue]🔧 DEBUG - Raw response:[/]")
                console.print(Syntax(response, "markdown", theme="monokai", line_numbers=True))
            
            # Process response to extract JSON
            try:
                # Look for JSON block in markdown format
                json_match = re.search(r'```(?:json)?\s*({[\s\S]*?})\s*```', response, re.DOTALL)
                if json_match:
                    try:
                        result = json.loads(json_match.group(1))
                    except:
                        # Try to clean JSON before parsing
                        json_str = json_match.group(1)
                        # Remove comment lines or non-JSON text
                        json_str = re.sub(r'^\s*//.*$', '', json_str, flags=re.MULTILINE)
                        try:
                            result = json.loads(json_str)
                        except:
                            result = {"type": "Unknown", "reasoning": response}
                else:
                    # Look for JSON outside markdown blocks
                    json_match = re.search(r'({[\s\S]*})', response)
                    if json_match:
                        try:
                            result = json.loads(json_match.group(1))
                        except:
                            result = {"type": "Unknown", "reasoning": response}
                    else:
                        # No JSON found, use the full text
                        result = {"type": "Unknown", "reasoning": response}
            except Exception as e:
                if debug:
                    console.print(f"[bold red]Error parsing JSON:[/] {str(e)}")
                result = {"type": "Unknown", "reasoning": response}
            
        else:
            # Use the original non-interactive approach
            if stream:
                # Callback for streaming
                console.print("[bold blue]🤖 Analyzing project structure...[/]")
                
                def stream_callback(chunk):
                    console.print(chunk, end="", highlight=False)
                
                # Execute with streaming
                result = asyncio.run(ai_cli.suggest_deployment_command(
                    app_directory, 
                    stream=True, 
                    callback=stream_callback
                ))
                console.print("\n")
            else:
                # Execute without streaming
                with console.status("[bold blue]Analyzing project structure...[/]", spinner="dots"):
                    result = asyncio.run(ai_cli.suggest_deployment_command(app_directory))
        
        # Display formatted results
        console.print("\n")
        console.print(Panel(
            "[bold cyan]AI has analyzed your project and has recommendations for deployment[/]",
            title="[bold green]📊 DEPLOYMENT RECOMMENDATIONS[/]",
            border_style="green",
            expand=False
        ))
        
        if isinstance(result, dict):
            # If it's a dictionary (successfully parsed JSON)
            recommendations_table = Table(show_header=False, box=None)
            recommendations_table.add_column("Property", style="cyan")
            recommendations_table.add_column("Value", style="green")
            
            recommendations_table.add_row("📂 Project Type", str(result.get('type', 'Unknown')))
            
            if result.get("command"):
                recommendations_table.add_row("🚀 Command", str(result['command']))
            
            if result.get("port"):
                recommendations_table.add_row("🔌 Port", str(result['port']))
            
            console.print(recommendations_table)
                
            if result.get("environment_vars"):
                env_table = Table(title="📋 Recommended Environment Variables", show_header=True)
                env_table.add_column("Variable", style="cyan")
                env_table.add_column("Value", style="green")
                
                for key, value in result["environment_vars"].items():
                    env_table.add_row(key, str(value))
                
                console.print(env_table)
            
            if result.get("reasoning"):
                markdown_content = Markdown(result["reasoning"])
                console.print(Panel(
                    markdown_content,
                    title="[bold blue]🔍 Analysis Details[/]",
                    border_style="blue",
                    width=100,
                    expand=False
                ))
        else:
            # If it's not a dictionary (string or other type)
            console.print(f"[bold cyan]📂 Detected project type:[/] [yellow]Unknown[/]")
            console.print(Panel(
                str(result),
                title="[bold blue]🔍 Analysis Details[/]",
                border_style="blue", 
                width=100
            ))
        
        console.print()
        
        if Confirm.ask("Would you like to register this application with this configuration?"):
            # Code for automatic registration
            console.print("[yellow]Automatic registration implementation pending.[/]")
            
    except Exception as e:
        console.print(f"[bold red]❌ Error during analysis:[/] {str(e)}")
        import traceback
        console.print(Syntax(traceback.format_exc(), "python", theme="monokai"))

@cli.command("query")
@click.option("--query", "-q", required=True, help="Query for the AI assistant")
@click.option("--stream/--no-stream", default=True, help="Show response in real-time")
@click.option("--debug/--no-debug", default=False, help="Show debug information")
@click.option("--language", type=click.Choice(["en", "es"]), default="en",
              help="Response language (English or Spanish)")
def query_command(query, stream, debug, language):
    """Make a general query to the AI assistant."""
    process_ai_query(query, stream, debug, language)

def process_ai_query(query, stream=True, debug=False, language="en"):
    """Process an AI query with the given parameters."""
    try:
        # Load AI configuration
        from atlasai.ai.ai_cli import AtlasServerAICLI
        ai_cli = AtlasServerAICLI()
        configured_model = ai_cli.ai_config.get("model", "qwen3:8b")
        
        console.print(f"[bold cyan]🤖 Using AI model:[/] [blue]{configured_model}[/]")
        
        # Check connection depending on provider
        if ai_cli.ai_config.get("provider", "ollama") == "ollama":
            import requests
            try:
                with console.status("[bold blue]Connecting to Ollama server...[/]", spinner="dots"):
                    response = requests.get("http://localhost:11434/api/version", timeout=2)
        
                if response.status_code != 200:
                    console.print("[bold red]❌ Error: Could not connect to Ollama server[/]")
                    return
                else:
                    if debug:
                        console.print(Panel.fit(
                        f"Version: {response.json().get('version', 'unknown')}",
                        title="[bold green]✅ Connected to Ollama[/]",
                        border_style="green"
                    ))
            except Exception as e:
                console.print("[bold red]❌ Error: Ollama server is not running[/]")
                console.print(f"   [italic]{str(e)}[/]")
                console.print("   Run [bold]'ollama serve'[/] or ensure the Ollama service is running.")
                return
        elif ai_cli.ai_config.get("provider") == "openai":
            # Verify API key exists
            if not ai_cli.ai_config.get("api_key"):
                console.print("[bold red]❌ Error: No API key configured for OpenAI[/]")
                console.print("   Run [bold]'atlasai ai setup --provider openai --api-key YOUR_API_KEY'[/]")
                return
    
            console.print(Panel.fit(
                "Using OpenAI for AI services",
                title="[bold green]✅ OpenAI configured[/]",
                border_style="green"
            ))
        
        # Initialize general agent
        from atlasai.ai.general_agent import GeneralAgent
        agent = GeneralAgent(
            model=configured_model, 
            provider=ai_cli.ai_config.get("provider", "ollama"),
            api_key=ai_cli.ai_config.get("api_key"),
            stream=stream, 
            language=language
        )
        
        console.print(Panel(
            f"[cyan]Query:[/] [bold]{query}[/]",
            title="[bold blue]🔍 AI Query[/]",
            border_style="blue"
        ))
        
        # Execute query with or without streaming
        if stream:
            full_response_text = []
            
            def collect_response(chunk):
                full_response_text.append(chunk)
                console.print(chunk, end="", highlight=False)
            
            # Execute with streaming
            response = asyncio.run(agent.process_query(
                query, 
                callback=collect_response
            ))
            
            # If response is empty but we have text, use it
            if not response and full_response_text:
                response = ''.join(full_response_text)
            console.print("\n")
        else:
            # Execute without streaming
            with console.status("[bold blue]Processing query...[/]", spinner="dots"):
                response = asyncio.run(agent.process_query(query))
        
        # Show complete response in debug mode
        if debug:
            console.print("\n[bold blue]🔧 DEBUG - Complete response:[/]")
            console.print(Syntax(response, "markdown", theme="monokai", line_numbers=True))
        
    except Exception as e:
        console.print(f"[bold red]❌ Error during query:[/] {str(e)}")
        import traceback
        console.print(Syntax(traceback.format_exc(), "python", theme="monokai"))

@cli.command("chat")
@click.option("--model", help="Model to use (defaults to configured model)")
@click.option("--provider", type=click.Choice(["ollama", "openai"]), 
              help="AI provider (ollama or openai, defaults to configured provider)")
@click.option("--api-key", help="API key (required for OpenAI if not configured)")
@click.option("--language", type=click.Choice(["en", "es"]), default="en",
              help="Response language (English or Spanish)")
@click.option("--prompt-level", type=click.Choice(["general", "advanced", "combined"]), 
              default="general", help="Prompt sophistication level")
def chat_command(model, provider, api_key, language, prompt_level):
    """Start an interactive chat session with the AI assistant."""
    try:
        # Load configuration
        from atlasai.ai.ai_cli import AtlasServerAICLI
        ai_cli = AtlasServerAICLI()
        
        # Use configured values if not specified
        model = model or ai_cli.ai_config.get("model", "qwen3:8b")
        provider = provider or ai_cli.ai_config.get("provider", "ollama")
        api_key = api_key or ai_cli.ai_config.get("api_key")
        
        console.print(f"[bold cyan]🤖 Using AI model:[/] [blue]{model}[/]")
        
        # Check connection based on provider
        if provider == "ollama":
            import requests
            try:
                with console.status("[bold blue]Connecting to Ollama server...[/]", spinner="dots"):
                    response = requests.get("http://localhost:11434/api/version", timeout=2)
            
                if response.status_code != 200:
                    console.print("[bold red]❌ Error: Could not connect to Ollama server[/]")
                    return
                else:
                    console.print(Panel.fit(
                        "Connected to Ollama server",
                        title="[bold green]✅ Connection Ready[/]",
                        border_style="green"
                    ))
            except Exception as e:
                console.print("[bold red]❌ Error: Ollama server is not running[/]")
                console.print(f"   [italic]{str(e)}[/]")
                console.print("   Run [bold]'ollama serve'[/] or ensure the Ollama service is running.")
                return
                
        elif provider == "openai":
            # Verify API key exists
            if not api_key:
                console.print("[bold red]❌ Error: No API key configured for OpenAI[/]")
                console.print("   Run [bold]'atlasai ai setup --provider openai --api-key YOUR_API_KEY'[/]")
                return
            
            console.print(Panel.fit(
                "OpenAI API configured",
                title="[bold green]✅ Connection Ready[/]",
                border_style="green"
            ))
        
        # Import and start the interactive chat
        from atlasai.ai.interactive_agent import start_interactive_cli
        
        console.print(Panel(
            f"Starting interactive chat mode with [bold blue]{model}[/] ({provider})\n"
            f"Language: [bold]{language}[/], Prompt level: [bold]{prompt_level}[/]\n"
            "Type [bold green]'exit'[/] or [bold green]'quit'[/] to end the session.",
            title="[bold blue]🤖 AtlasAI Interactive Chat[/]",
            border_style="blue"
        ))
        
        # Start the interactive session
        asyncio.run(start_interactive_cli(
            model=model,
            provider=provider,
            api_key=api_key,
            language=language,
            prompt_level=prompt_level
        ))
        
    except Exception as e:
        from rich.text import Text
        error_text = Text("❌ Error starting chat: ", style="bold red")
        error_text.append(str(e))
        console.print(error_text)
        import traceback
        console.print(Syntax(traceback.format_exc(), "python", theme="monokai"))


@cli.command("task")
@click.argument("task_file", type=click.Path(exists=True))
@click.option("--working-dir", type=click.Path(exists=True), default=".",
              help="Working directory for executing tasks")
@click.option("--model", help="Model to use (defaults to configured model)")
@click.option("--provider", type=click.Choice(["ollama", "openai"]), 
              help="AI provider (ollama or openai, defaults to configured provider)")
@click.option("--api-key", help="API key (required for OpenAI if not configured)")
@click.option("--language", type=click.Choice(["en", "es"]), default="en",
              help="Response language (English or Spanish)")
@click.option("--verify-commands/--no-verify-commands", default=True,
              help="Verify commands before execution")
def task_command(task_file, working_dir, model, provider, api_key, language, verify_commands):
    """Execute a task workflow defined in a markdown file."""
    try:
        # Load configuration
        from atlasai.ai.ai_cli import AtlasServerAICLI
        ai_cli = AtlasServerAICLI()
        
        # Use configured values if not specified
        model = model or ai_cli.ai_config.get("model", "gpt-4.1")
        provider = provider or ai_cli.ai_config.get("provider", "openai")
        api_key = api_key or ai_cli.ai_config.get("api_key")
        
        console.print(f"[bold cyan]🤖 Using AI model:[/] [blue]{model}[/]")
        
        # Check connection based on provider
        if provider == "ollama":
            # (Same code as in chat_command)
            pass
        elif provider == "openai":
            # (Same code as in chat_command)
            pass
        
        # Parse the task file
        from atlasai.task.task_parser import parse_task_file
        
        console.print(Panel(
            f"[cyan]Analyzing task file:[/] [bold]{task_file}[/]",
            title="[bold blue]📄 Task Analysis[/]",
            border_style="blue"
        ))
        
        task_graph = parse_task_file(task_file)
        
        # Show task workflow information
        console.print(f"[bold cyan]Title:[/] {task_graph.metadata.get('title', 'Untitled')}")
        console.print(f"[bold cyan]Tasks:[/] {len(task_graph.tasks)}")
        
        # Create and execute the task executor
        from atlasai.task.task_executor import TaskExecutor
        
        executor = TaskExecutor(
            task_graph=task_graph,
            working_dir=working_dir,
            model=model,
            provider=provider,
            api_key=api_key,
            language=language,
            verify_commands=verify_commands
        )
        
        asyncio.run(executor.execute_tasks())
        
    except Exception as e:
        console.print(f"[bold red]❌ Error executing task workflow:[/] {str(e)}")
        import traceback
        console.print(Syntax(traceback.format_exc(), "python", theme="monokai"))

@cli.command("task-template")
@click.argument("output_file", type=click.Path(exists=False))
@click.option("--tasks", type=int, default=3,
              help="Number of task templates to include")
@click.option("--overwrite/--no-overwrite", default=False,
              help="Overwrite existing file if it exists")
def task_template_command(output_file, tasks, overwrite):
    """Generate a template task file with the correct format."""
    try:
        # Check if file already exists
        if os.path.exists(output_file) and not overwrite:
            console.print(f"[bold red]❌ File already exists: {output_file}[/]")
            console.print("Use --overwrite to replace the existing file.")
            return

        # Generate template content
        from atlasai.task.task_executor import generate_task_template
        template = generate_task_template(tasks)
        
        # Create directory if it doesn't exist
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Write the template file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(template)
            
        console.print(Panel(
            f"[bold green]✅ Task template created:[/] [cyan]{output_file}[/]\n\n"
            f"This template includes {tasks} example tasks with dependencies.\n"
            f"Edit it according to your needs and run it with:\n\n"
            f"[bold]atlasai task {output_file}[/]",
            title="[bold green]Template Generated[/]",
            border_style="green"
        ))
        
    except Exception as e:
        console.print(f"[bold red]❌ Error creating template:[/] {str(e)}")


cli.add_command(ai)

def main():
    """Main entry point for the CLI."""
    # Check if using pattern atlasai --query or --chat
    import sys
    if len(sys.argv) >= 2:
        if sys.argv[1] == "--query":
            # Direct invocation with --query flag
            query = sys.argv[2]
            language = "en"
            
            # Check if --language is specified
            for i in range(3, len(sys.argv)):
                if sys.argv[i] == "--language" and i + 1 < len(sys.argv):
                    language = sys.argv[i + 1]
                    break
            
            # Process the query directly
            process_ai_query(query, stream=True, debug=False, language=language)
            return
        elif sys.argv[1] == "--chat":
            # Direct invocation with --chat flag
            language = "en"
            prompt_level = "general"
            
            # Check if options are specified
            for i in range(2, len(sys.argv)):
                if sys.argv[i] == "--language" and i + 1 < len(sys.argv):
                    language = sys.argv[i + 1]
                elif sys.argv[i] == "--prompt-level" and i + 1 < len(sys.argv):
                    prompt_level = sys.argv[i + 1]
            
            # Start chat session
            chat_command(None, None, None, language, prompt_level)
            return
    cli()

if __name__ == "__main__":
    main()