# atlasai/ai/ai_service.py 
import logging
from typing import Optional, Callable
from openai import OpenAI

logger = logging.getLogger(__name__)

class AIService:
    """Base class for AI services."""
    
    def __init__(self, model: str):
        self.model = model
    
    async def generate_response(self, prompt: str, structured_output: bool = False) -> str:
        """Generate a response from the AI service."""
        raise NotImplementedError("The base service does not implement generate_response")
        
    async def generate_response_stream(self, prompt: str, callback: Callable[[str], None]) -> str:
        """Generate a streaming response with callback for each chunk."""
        raise NotImplementedError("The base service does not implement streaming")

class OpenAIService(AIService):
    def __init__(self, model: str, api_key: str):
        super().__init__(model)
        
        assert api_key is not None and api_key.strip() != "", "OpenAI API key cannot be empty"
        assert model is not None and model.strip() != "", "Model name cannot be empty"
        
        self.client = OpenAI(api_key=api_key)

    async def generate_response(self, prompt: str, structured_output: bool = False) -> str:
        assert prompt is not None, "Prompt cannot be None"
        
        try:
            system_prompt = """You are an AI assistant specialized in application deployment.
                Your task is to analyze code and configurations to suggest deployment strategies.
                Be precise and provide detailed reasoning for your suggestions."""
            
            if structured_output:
                system_prompt += """ Include a 'reasoning' field in your JSON response that 
                explains the rationale behind your suggestions in detail."""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        'role': 'system',
                        'content': system_prompt
                    },
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ]
            )
            
            assert response.choices and len(response.choices) > 0, "No se recibió respuesta del modelo"
            return str(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"Error generating response with OpenAI: {str(e)}")
            raise RuntimeError(f"Error generating response with OpenAI: {str(e)}")

    async def generate_response_stream(self, prompt: str, callback: Callable[[str], None]) -> str:
        assert prompt is not None, "Prompt cannot be None"
        assert callback is not None, "Callback function cannot be None"
        
        try:
            # Prepare system prompt
            system_prompt = """You are an AI assistant specialized in application deployment.
            Your task is to analyze code and configurations to suggest deployment strategies.
            Be precise and provide detailed reasoning for your suggestions."""

            # Correctly implement streaming with OpenAI
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        'role': 'system',
                        'content': system_prompt
                    },
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ],
                stream=True
            )
            
            full_response = ""
            for chunk in response:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, 'content') and delta.content is not None:
                        text_chunk = delta.content
                        full_response += text_chunk
                        callback(text_chunk)
            
            return full_response
        except Exception as e:
            error_msg = f"Error generating streaming response: {str(e)}"
            logger.error(error_msg)
            callback(error_msg)
            raise RuntimeError(f"Error generating streaming response with OpenAI: {str(e)}")

class OllamaService(AIService):
    """AI service using local Ollama models."""
    
    def __init__(self, model: str = "qwen3:8b"):
        super().__init__(model)
        
    async def generate_response(self, prompt: str, structured_output: bool = False) -> str:
        try:
            from ollama import chat
            
            # Prepare system prompt
            system_prompt = """You are an AI assistant specialized in application deployment.
            Your task is to analyze code and configurations to suggest deployment strategies.
            Be precise and provide detailed reasoning for your suggestions."""
            
            if structured_output:
                system_prompt += """ Include a 'reasoning' field in your JSON response that 
                explains the rationale behind your suggestions in detail."""
                
            # Call the Ollama API
            response = chat(
                model=self.model,
                messages=[
                    {
                        'role': 'system',
                        'content': system_prompt
                    },
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ]
            )
            
            return response['message']['content']
        except Exception as e:
            logger.error(f"Error generating response with Ollama: {str(e)}")
            return f"Error generating response: {str(e)}"
            
    async def generate_response_stream(self, prompt: str, callback: Callable[[str], None]) -> str:
        """Generate streaming response with Ollama.
        
        Args:
            prompt: User query or instruction
            callback: Function to call with each chunk of text
            
        Returns:
            Complete response when finished
        """
        try:
            from ollama import chat
            
            # Prepare system prompt
            system_prompt = """You are an AI assistant specialized in application deployment.
            Your task is to analyze code and configurations to suggest deployment strategies.
            Be precise and provide detailed reasoning for your suggestions."""
            
            # Call Ollama API with streaming
            full_response = ""
            for chunk in chat(
                model=self.model,
                messages=[
                    {
                        'role': 'system',
                        'content': system_prompt
                    },
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ],
                stream=True
            ):
                if 'message' in chunk and 'content' in chunk['message']:
                    text_chunk = chunk['message']['content']
                    full_response += text_chunk
                    callback(text_chunk)
            
            return full_response
        except Exception as e:
            error_msg = f"Error generating streaming response: {str(e)}"
            logger.error(error_msg)
            callback(error_msg)
            return error_msg

async def get_ai_service(provider: str = "ollama", model: str = "qwen3:8b", api_key: Optional[str] = None) -> AIService:
    """Factory to get the appropriate AI service implementation."""
    
    assert provider is not None, "Provider cannot be None"
    assert model is not None, "Model name cannot be None"
    
    provider = provider.lower()
    
    if provider == "ollama":
        return OllamaService(model=model)
    elif provider == "openai":
        assert api_key is not None and api_key.strip() != "", "API key is required for OpenAI"
        return OpenAIService(model=model, api_key=api_key)
    else:
        raise ValueError(f"Unsupported provider: {provider}")