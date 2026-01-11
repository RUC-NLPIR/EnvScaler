"""
Call LLM with different providers.
"""
import time
import os
from openai import OpenAI
from typing import Optional, List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def llm_inference(provider: str, model: str, messages: List[dict], temperature: float = None, stop_strs: Optional[List[str]] = None, max_tokens: int = None):
    """Call LLM with different providers."""
    if provider == "openai":
        return openai_llm_inference(model, messages, temperature, stop_strs, max_tokens)
    else:
        # add other provider here
        raise ValueError(f"Invalid provider: {provider}")

def openai_llm_inference(
    model: str,
    messages: List[dict],
    temperature: float = None,
    stop_strs: Optional[List[str]] = None,
    max_tokens: int = None
):
    """Call OpenAI LLM API with retry mechanism."""
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL")
    )
    retries = 0
    max_retries = 5
    while retries < max_retries:
        try:
            if 'gpt-5' in model:
                print("Think model cannot set stop_strs, temperature, max_tokens, ignoring these settings")
                response = client.responses.create(
                    model=model,
                    input=messages,
                )
                output = response.output_text
                return output
            else:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    stop=stop_strs,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                output = response.choices[0].message.content
                return output
        # except KeyboardInterrupt:
        #     print("Operation canceled by user.")
        #     raise
        except Exception as e:
            print(f"Something wrong: {e}. Retrying in {retries*10+10} seconds...")
            time.sleep(retries*10)
            retries += 1
    print(f"Failed to get response after {max_retries} retries, return empty string")
    return ''

def openai_single_embedding_inference(model: str, text: str) -> List[float]:
    """Get embedding for a single text using OpenAI API with retry mechanism."""
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )
    retries = 0
    max_retries = 5
    while retries < max_retries:
        try:
            response = client.embeddings.create(
                model=model,
                input=text
            )
            return response.data[0].embedding
        except KeyboardInterrupt:
            print("Operation canceled by user.")
            break
        except Exception as e:
            print(f"Something wrong: {e}. Retrying in {retries*10+10} seconds...")
            time.sleep(retries*10)
            retries += 1
    print(f"Failed to get embedding after {max_retries} retries, return empty list")
    return []

def openai_batch_embedding_inference(model: str, texts: List[str]) -> List[List[float]]:
    """Get embeddings for multiple texts using OpenAI API with retry mechanism."""
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )
    retries = 0
    max_retries = 5
    while retries < max_retries:
        try:
            response = client.embeddings.create(
                model=model,
                input=texts
            )
            return [d.embedding for d in response.data]
        except Exception as e:
            print(f"Something wrong: {e}. Retrying in {retries*10+10} seconds...")
            time.sleep(retries*10)
            retries += 1
    print(f"Failed to get embeddings after {max_retries} retries, return empty list")
    return []


# if __name__ == "__main__":
#     # Test LLM inference
#     # messages = [{'role': 'user', 'content': "Introduce yourself."}]
#     # response = openai_llm_inference(model="gpt-4.1", messages=messages, temperature=0)
#     # print(response)
#
#     # Test embedding inference
#     # text = "Hello, world!"
#     # embedding = openai_single_embedding_inference(model="text-embedding-3-large", text=text)
#     # print(embedding)
#     # print(type(embedding))
#     # print(len(embedding))
#
#     # Test batch embedding inference
#     # texts = ["Hello, world!", "Hello, china!", "Hello, america!"]
#     # embeddings = openai_batch_embedding_inference(model="text-embedding-3-large", texts=texts)
#     # print(embeddings)
#     # print(type(embeddings))
#     # print(len(embeddings))