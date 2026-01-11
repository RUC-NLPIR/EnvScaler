"""
LLM inference utilities for action agent inference.
"""

import os
import time
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict, Any, Tuple, Optional

# Load environment variables
load_dotenv()


def openai_inference_prompt(
    model: str, 
    messages: List[Dict[str, Any]], 
    temperature: float = None,
    enable_thinking: bool = False,
    api_key: str = None,
    base_url: str = None
    ) -> str:
    """Non-streaming inference for prompt mode."""
    client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"), base_url=base_url or os.getenv("OPENAI_BASE_URL"))
    retries = 0
    max_retries = 10
    while retries < max_retries:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                stream=False,
                temperature=temperature,
                max_tokens=10000,
                n=1,
                extra_body={"chat_template_kwargs": {"enable_thinking": enable_thinking}},
            )
            content = response.choices[0].message.content
            # Get reasoning content if available
            if hasattr(response.choices[0].message, "reasoning_content"):
                reasoning_content = response.choices[0].message.reasoning_content
            else:
                reasoning_content = ""
            # Prepend reasoning content if not empty (Qwen3 template style)
            if reasoning_content:
                reasoning_content = reasoning_content.strip()
                content = f"<think>\n{reasoning_content}\n</think>\n\n{content}"
            return content

        except Exception as e:
            print(f"Something wrong: {e}. Retrying in {retries * 10 + 10} seconds...")
            time.sleep(retries * 10)
            
            retries += 1
            
    print(f"Failed to get response after {max_retries} retries.")
    return ''

def openai_stream_inference_prompt(
    model: str,
    messages: List[Dict[str, Any]],
    temperature: float = None,
    enable_thinking: bool = False,
    api_key: str = None,
    base_url: str = None
) -> str:
    """Streaming inference for prompt mode."""
    client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"), base_url=base_url or os.getenv("OPENAI_BASE_URL"))

    retries = 0
    max_retries = 10
    max_tokens = 10000
    while retries < max_retries:
        params = {
            "model": model,
            "messages": messages,
            "stream": True,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": enable_thinking}},
            "n": 1
        }
        try:
            completion = client.chat.completions.create(**params)

            reasoning_content = ""
            content = ""

            for chunk in completion:
                if not getattr(chunk, "choices", None):
                    continue

                choice = chunk.choices[0]
                delta = choice.delta

                # Accumulate reasoning content
                if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                    reasoning_content += delta.reasoning_content

                # Accumulate content
                if hasattr(delta, "content") and delta.content:
                    content += delta.content

            reasoning_content = reasoning_content.strip()
            content = content.strip()

            # Check if <think> tag is present in content
            if not reasoning_content and content and '</think>' in content:
                reasoning_content = content.split('</think>')[0].strip()
                if '<think>' in reasoning_content:
                    reasoning_content = reasoning_content.split('<think>')[1].strip()
                content = content.split('</think>')[1].strip()
            
            # Prepend reasoning content if not empty (Qwen3 template style)
            if reasoning_content:
                content = f"<think>\n{reasoning_content}\n</think>\n\n{content}"

            if content == "":
                raise ValueError("content is empty.")
            return content
        
        except Exception as e:
            print(f"Something wrong: {e}. Retrying in {retries * 10 + 10} seconds...")
            time.sleep(retries * 10)
            if retries >= 5:
                max_tokens = 5000
                print(f"max_tokens: {max_tokens}")
            retries += 1

    print(f"Failed to get response after {max_retries} retries.")
    return ""

def openai_stream_inference_fc(
    model: str,
    messages: List[Dict[str, Any]],
    temperature: float = None,
    tools: Optional[List[Dict]] = None,
    enable_thinking: bool = False,
    api_key: str = None,
    base_url: str = None
) -> Dict[str, Any]:
    """
    Streaming inference using official Model tool interface (function calling mode).
    Returns:
        {
            "reasoning_content": str,
            "tool_calls": list,
            "content": str
        }
    """
    client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"), base_url=base_url or os.getenv("OPENAI_BASE_URL"))

    retries = 0
    max_retries = 10
    while retries < max_retries:
        try:
            if tools:
                completion = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    stream=True,
                    temperature=temperature,
                    max_tokens=10000,
                    tools=tools,
                    tool_choice="auto",
                    top_p=0.95,
                    n=1,
                    extra_body={"chat_template_kwargs": {"enable_thinking": enable_thinking}}
                )
            else:
                completion = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    stream=True,
                    temperature=temperature,
                    max_tokens=10000,
                    n=1,
                    extra_body={"chat_template_kwargs": {"enable_thinking": enable_thinking}}
                )

            reasoning_content = ""
            content = ""
            # Accumulate tool calls by index
            tool_calls_accum: Dict[int, Dict[str, Any]] = {}

            for chunk in completion:
                if not getattr(chunk, "choices", None):
                    continue

                choice = chunk.choices[0]
                delta = choice.delta

                # Accumulate reasoning content
                if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                    reasoning_content += delta.reasoning_content

                # Accumulate content
                if hasattr(delta, "content") and delta.content:
                    content += delta.content

                # Accumulate tool call information
                if hasattr(delta, "tool_calls") and delta.tool_calls:
                    for tool_call in delta.tool_calls:
                        idx = tool_call.index
                        if idx not in tool_calls_accum:
                            tool_calls_accum[idx] = {
                                "id": tool_call.id or "",
                                "type": tool_call.type or "function",
                                "function": {
                                    "name": "",
                                    "arguments": ""
                                }
                            }
                        if tool_call.id:
                            tool_calls_accum[idx]["id"] = tool_call.id
                        if tool_call.type:
                            tool_calls_accum[idx]["type"] = tool_call.type
                        if tool_call.function:
                            if tool_call.function.name:
                                tool_calls_accum[idx]["function"]["name"] += tool_call.function.name
                            if tool_call.function.arguments:
                                tool_calls_accum[idx]["function"]["arguments"] += tool_call.function.arguments

            # Final tool_calls list
            tool_calls = list(tool_calls_accum.values())
            if len(tool_calls) > 1:
                print("warning: more than one tool_call, only keep the first one.")
                tool_calls = [tool_calls[0]]

            # Check if <think> tag is present in content
            if not reasoning_content and content and '</think>' in content:
                reasoning_content = content.split('</think>')[0].strip()
                if '<think>' in reasoning_content:
                    reasoning_content = reasoning_content.split('<think>')[1].strip()
                content = content.split('</think>')[1].strip()
                
            if not content and not tool_calls and not reasoning_content:
                raise ValueError("all content is empty.")
        
            result = {
                "reasoning_content": reasoning_content,
                "tool_calls": tool_calls,
                "content": content
            }
        
            return result
        
        except Exception as e:
            print(f"Something wrong: {e}. Retrying in {retries * 10 + 10} seconds...")
            time.sleep(retries * 10)
            retries += 1

    print(f"Failed to get response after {max_retries} retries.")
    return {"reasoning_content": "", "tool_calls": [], "content": ""}


def llm_inference_fc(provider: str, model: str, messages: List[Dict[str, Any]], temperature: float = None, tools: Optional[List[Dict]] = None, enable_thinking: bool = False, api_key: str = None, base_url: str = None) -> Dict[str, Any]:
    """
    Unified LLM inference interface for FC mode.
    """
    if provider == "openai":
        return openai_stream_inference_fc(model=model, messages=messages, temperature=temperature, tools=tools, enable_thinking=enable_thinking, api_key=api_key, base_url=base_url)
    else:
        # add other provider support here
        raise ValueError(f"Invalid provider: {provider}")


def llm_inference_prompt(provider: str, model: str, messages: List[Dict[str, Any]], temperature: float = None, enable_thinking: bool = False, api_key: str = None, base_url: str = None) -> str:
    """
    Unified LLM inference interface for Prompt mode.
    """
    if provider == "openai":
        return openai_stream_inference_prompt(model=model, messages=messages, temperature=temperature, enable_thinking=enable_thinking, api_key=api_key, base_url=base_url)
    else:
        # add other provider support here
        raise ValueError(f"Invalid provider: {provider}")


if __name__ ==  "__main__":
    # Test FC mode with tools
    msgs = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the weather in Beijing?"}
    ]

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather of a city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name"}
                    },
                    "required": ["city"]
                }
            }
        }
    ]

    model = "gpt-4.1"
    provider = "openai"
    result = llm_inference_fc(
        provider=provider,
        model=model, 
        messages=msgs, 
        tools=tools,
    )
    print(result)