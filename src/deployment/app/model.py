import requests
import os
from typing import Optional

class Model:
    def __init__(self, endpoint: Optional[str] = None):
        """
        Initialize the model with NIM endpoint.
        
        Args:
            endpoint (str, optional): NIM API endpoint. Defaults to environment variable NIM_API_ENDPOINT.
        """
        self.endpoint = endpoint or os.getenv("NIM_API_ENDPOINT", "http://localhost:8001/generate")
        
    def generate(self, prompt: str, max_tokens: int = 100, temperature: float = 0.7) -> str:
        """
        Generate text using the NIM API.
        
        Args:
            prompt (str): Input prompt
            max_tokens (int): Maximum number of tokens to generate
            temperature (float): Sampling temperature
            
        Returns:
            str: Generated text
        """
        try:
            response = requests.post(
                self.endpoint,
                json={
                    "inputs": prompt,
                    "parameters": {
                        "max_tokens": max_tokens,
                        "temperature": temperature
                    }
                }
            )
            response.raise_for_status()
            return response.json()["outputs"]
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Error calling NIM API: {str(e)}")
