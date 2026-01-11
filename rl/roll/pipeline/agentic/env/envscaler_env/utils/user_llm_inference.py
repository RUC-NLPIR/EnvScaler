import time
import os
from openai import OpenAI
from dotenv import load_dotenv
from typing import Optional, List


def openai_llm_inference(
        model: str, 
        messages: List[dict],
        temperature: float = None, 
        stop_strs: Optional[List[str]] = None,
        max_tokens: int = None):
    load_dotenv()
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )
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


def llm_inference(model, messages, provider):
    if provider == "openai":
        return openai_llm_inference(
            model=model, 
            messages=messages,
            temperature=0.7, 
        )
    else:
        # add other provider support here
        raise ValueError(f"Unknown provider {provider}")
 
    