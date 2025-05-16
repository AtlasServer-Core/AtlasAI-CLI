import click
import os
import json
import re
import asyncio

@click.group()
def cli():
    """AtlasAI CLI - AI-powered tools for AtlasServer deployments."""
    pass

@click.group()
def ai():
    pass

@ai.command("setup")
@click.option("--provider", type=click.Choice(["ollama"]), default="ollama", 
              help="AI provider (only ollama in Core)")
@click.option("--model", default="qwen3:8b", help="Model to use (e.g.: qwen3:8b)")
def ai_setup(provider, model):
    """Configure the AI service for CLI."""
    from atlasai.ai.ai_cli import AtlasServerAICLI
    ai_cli = AtlasServerAICLI()
    success = ai_cli.setup(provider, model, None)
    
    if success:
        click.echo(f"✅ AI configuration saved: {provider} / {model}")
    else:
        click.echo("❌ Error saving AI configuration")

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
        
        # Cargar configuración AI
        from atlasai.ai.ai_cli import AtlasServerAICLI
        ai_cli = AtlasServerAICLI()
        configured_model = ai_cli.ai_config.get("model", "codellama:7b")
        
        click.echo(f"🤖 Using AI model: {configured_model}")
        
        # Verificar Ollama
        import requests
        try:
            response = requests.get("http://localhost:11434/api/version", timeout=2)
            if response.status_code != 200:
                click.echo("❌ Error: Could not connect to Ollama server")
                return
            else:
                if debug:
                    click.echo(f"✅ Connected to Ollama: {response.json()}")
        except Exception as e:
            click.echo(f"❌ Error: Ollama server is not running. {str(e)}")
            click.echo("   Run 'ollama serve' or ensure the Ollama service is running.")
            return
        
        if interactive:
            # Use el nuevo enfoque simplificado (sin herramientas complejas)
            from atlasai.ai.ai_agent import AgentCLI
            agent = AgentCLI(model=configured_model, stream=stream, language=language)
            
            click.echo(f"🔍 Analyzing project at: {app_directory}")
            
            # Define callback para streaming si es necesario
            if stream:
                full_response_text = []
                
                def collect_response(chunk):
                    full_response_text.append(chunk)
                    click.echo(chunk, nl=False)
                
                # Ejecutar con streaming
                response = asyncio.run(agent.analyze_project(
                    app_directory, 
                    callback=collect_response
                ))
                
                # Si la respuesta está vacía pero tenemos texto, úsalo
                if not response and full_response_text:
                    response = ''.join(full_response_text)
                click.echo("\n")
            else:
                # Ejecutar sin streaming
                click.echo("⏳ This may take a moment...")
                response = asyncio.run(agent.analyze_project(app_directory))
            
            # Mostrar respuesta completa en modo debug
            if debug:
                click.echo("\n🔧 DEBUG - Raw response:")
                click.echo("-"*50)
                click.echo(response)
                click.echo("-"*50)
            
            # Procesar la respuesta para extraer JSON
            try:
                # Buscar bloque JSON en formato markdown
                json_match = re.search(r'```(?:json)?\s*({[\s\S]*?})\s*```', response, re.DOTALL)
                if json_match:
                    try:
                        result = json.loads(json_match.group(1))
                    except:
                        # Intentar limpiar el JSON antes de parsearlo
                        json_str = json_match.group(1)
                        # Eliminar líneas de comentarios o texto no-JSON
                        json_str = re.sub(r'^\s*//.*$', '', json_str, flags=re.MULTILINE)
                        try:
                            result = json.loads(json_str)
                        except:
                            result = {"type": "Unknown", "reasoning": response}
                else:
                    # Buscar JSON fuera de bloques markdown
                    json_match = re.search(r'({[\s\S]*})', response)
                    if json_match:
                        try:
                            result = json.loads(json_match.group(1))
                        except:
                            result = {"type": "Unknown", "reasoning": response}
                    else:
                        # No se encontró JSON, usar el texto completo
                        result = {"type": "Unknown", "reasoning": response}
            except Exception as e:
                if debug:
                    click.echo(f"Error parsing JSON: {str(e)}")
                result = {"type": "Unknown", "reasoning": response}
            
        else:
            # Usar el enfoque no interactivo original
            if stream:
                # Callback para streaming
                click.echo("🤖 Analyzing project structure...")
                
                def stream_callback(chunk):
                    click.echo(chunk, nl=False)
                
                # Ejecutar con streaming
                result = asyncio.run(ai_cli.suggest_deployment_command(
                    app_directory, 
                    stream=True, 
                    callback=stream_callback
                ))
                click.echo("\n")
            else:
                # Ejecutar sin streaming
                click.echo("🤖 Analyzing project structure...")
                result = asyncio.run(ai_cli.suggest_deployment_command(app_directory))
        
        # Mostrar resultados formateados
        click.echo("\n" + "="*50)
        click.echo("📊 DEPLOYMENT RECOMMENDATIONS")
        click.echo("="*50)
        
        if isinstance(result, dict):
            # Si es un diccionario (JSON parseado exitosamente)
            click.echo(f"📂 Detected project type: {result.get('type', 'Unknown')}")
            
            if result.get("command"):
                click.echo(f"🚀 Recommended command: {result['command']}")
            
            if result.get("port"):
                click.echo(f"🔌 Recommended port: {result['port']}")
                
            if result.get("environment_vars"):
                click.echo("\n📋 Recommended environment variables:")
                for key, value in result["environment_vars"].items():
                    click.echo(f"  {key}={value}")
            
            if result.get("reasoning"):
                click.echo("\n🔍 Analysis details:")
                click.echo("-"*50)
                reasoning = result["reasoning"]
                if isinstance(reasoning, str):
                    # Limitar longitud de líneas para mejor visualización
                    for line in reasoning.split("\n"):
                        if len(line) > 80:
                            parts = [line[i:i+80] for i in range(0, len(line), 80)]
                            for part in parts:
                                click.echo(f"  {part}")
                        else:
                            click.echo(f"  {line}")
                else:
                    click.echo(f"  {reasoning}")
        else:
            # Si no es un diccionario (string u otro tipo)
            click.echo(f"📂 Detected project type: Unknown")
            click.echo("\n🔍 Analysis details:")
            click.echo("-"*50)
            click.echo(f"  {result}")
        
        click.echo("\n" + "="*50)
                
        if click.confirm("Would you like to register this application with this configuration?"):
            # Código para registrar automáticamente
            click.echo("Automatic registration implementation pending.")
            
    except Exception as e:
        click.echo(f"❌ Error during analysis: {str(e)}")
        import traceback
        click.echo(traceback.format_exc())

cli.add_command(ai)

def main():
    """Punto de entrada principal para el CLI."""
    cli()

if __name__ == "__main__":
    main()