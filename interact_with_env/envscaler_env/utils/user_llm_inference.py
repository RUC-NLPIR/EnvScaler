"""
LLM inference utilities for user agent.
"""
import time
import os
from openai import OpenAI
from dotenv import load_dotenv

from typing import Optional, List, Dict, Any, Union

# Load environment variables
load_dotenv()
 
def openai_llm_inference(
        model: str, 
        messages: List[dict],
        temperature: float = None, 
        stop_strs: Optional[List[str]] = None,
        max_tokens: int = None,
        api_key: str = None,
        base_url: str = None):
    """Call OpenAI API with retry mechanism."""
    client = OpenAI(api_key=api_key or os.getenv("USER_OPENAI_API_KEY"), base_url=base_url or os.getenv("USER_OPENAI_BASE_URL"))
    retries = 0
    max_retries = 10
    while retries < max_retries:
        try:
            response = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        stop=stop_strs,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
            output=response.choices[0].message.content
            return output
        except KeyboardInterrupt:
            print("Operation canceled by user.")
            break
        except Exception as e:
            print(f"Someting wrong:{e}. Retrying in {retries*10+10} seconds...")
            time.sleep(retries*10) 
            retries += 1
    return ''
    
    
def llm_inference(model, messages, provider, api_key=None, base_url=None):
    """Unified LLM inference interface based on provider."""
    if provider == "openai":
        return openai_llm_inference(
            model=model, 
            messages=messages,
            temperature=0.7,
            api_key=api_key,
            base_url=base_url
        )
    else:
        raise ValueError(f"Invalid provider: {provider}.")