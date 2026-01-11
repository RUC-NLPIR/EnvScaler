"""
step1: 
To ensure alignment with the model's original tool-calling format during training, 
we first convert the structured messages (containing user, assistant, tool, content, reasoning_content, etc.) 
into messages that only include (user, assistant, content) according to the chat_template.

After processing, you no longer need to use LlamaFactory's tool-training mode, and you dont't have to worry about tool format alignment issues.

example:

# Input: tools
[
    {
        "type": "function",
        "function": {
            "name": "text2image-image_generation",
            "description": "\n    Image generation assistant, please imagine and describe a complete picture in detail based on my simple description. Then translate your detailed description into English\n\n    :param width: \u56fe\u7247\u5bbd\u5ea6\uff0c\u9ed8\u8ba41024\n    :param height: \u56fe\u7247\u9ad8\u5ea6\uff0c\u9ed8\u8ba41024\n    :param image_prompt:\u56fe\u7247\u63cf\u8ff0\uff0c\u9700\u8981\u82f1\u6587\n    :return:\u56fe\u7247\u7684url\u5730\u5740\n    ",
            "parameters": {
                "properties": {
                    "image_prompt": {
                        "title": "Image Prompt",
                        "type": "string"
                    },
                    "width": {
                        "default": 1024,
                        "title": "Width",
                        "type": "integer"
                    },
                    "height": {
                        "default": 1024,
                        "title": "Height",
                        "type": "integer"
                    }
                },
                "required": [
                    "image_prompt"
                ],
                "title": "image_generationArguments",
                "type": "object"
            }
        }
    }
]

# Input: structured messages
[
    {
        "role": "user",
        "content": "I need to create a promotional banner for my upcoming conference. The banner should feature a futuristic cityscape at dusk with a glowing neon conference logo that reads \"Tech Summit 2024\" overlaid in the center. The image needs to be high resolution (at least 2000 by 1000 pixels) for print quality. Can you help me generate this visual from my description?"
    },
    {
        "role": "assistant",
        "content": "I'll create your promotional banner for Tech Summit 2024 with the futuristic cityscape at dusk and glowing neon logo. Based on your requirements for print quality, I'll use the higher resolution you specified.",
        "tool_calls": [
            {
                "function": {
                    "name": "text2image-image_generation",
                    "arguments": {
                        "width": 2000,
                        "height": 1000,
                        "image_prompt": "A futuristic cyberpunk cityscape at dusk with towering skyscrapers covered in holographic advertisements and neon lights in blue, purple, and cyan colors. The sky has a gradient from deep violet at the top to warm orange and pink near the horizon where the sun is setting. In the center of the image, a glowing neon sign reads \"Tech Summit 2024\" in bright electric blue and cyan with a subtle glitch effect and light trails. The city below has sleek modern architecture with glass and metal surfaces reflecting the neon glow. Digital billboards and tech company logos are visible on building facades. The overall atmosphere is high-tech and professional with a dark futuristic aesthetic perfect for a technology conference banner. Sharp focus, ultra high resolution, HDR lighting, cinematic quality."
                    }
                }
            }
        ]
    },
    {
        "role": "tool",
        "content": "![image](https://image.pollinations.ai/prompt/A futuristic cyberpunk cityscape at dusk with towering skyscrapers covered in holographic advertisements and neon lights in blue, purple, and cyan colors. The sky has a gradient from deep violet at the top to warm orange and pink near the horizon where the sun is setting. In the center of the image, a glowing neon sign reads \"Tech Summit 2024\" in bright electric blue and cyan with a subtle glitch effect and light trails. The city below has sleek modern architecture with glass and metal surfaces reflecting the neon glow. Digital billboards and tech company logos are visible on building facades. The overall atmosphere is high-tech and professional with a dark futuristic aesthetic perfect for a technology conference banner. Sharp focus, ultra high resolution, HDR lighting, cinematic quality.?width=2000&height=1000&enhance=true&nologo=true&model=flux)"
    },
    {
        "role": "assistant",
        "content": "Perfect! I've generated your promotional banner for Tech Summit 2024. The image features:\n\n- **Futuristic cityscape at dusk** with towering skyscrapers and sleek modern architecture\n- **Vibrant neon lighting** in blue, purple, and cyan colors with holographic advertisements\n- **Glowing \"Tech Summit 2024\" neon logo** centered prominently in electric blue and cyan with glitch effects\n- **2000x1000 pixel resolution** for high-quality print use\n- **Cinematic quality** with professional HDR lighting and sharp focus\n- **Cyberpunk aesthetic** perfect for a technology conference\n\nThe image is ready for download and can be used directly for your conference promotional materials. The dark, high-tech atmosphere combined with the professional neon branding should effectively represent your Tech Summit 2024 event."
    }
]

# Output: plain text messages
[
    {
        "role": "system",
        "content": "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{\"type\": \"function\", \"function\": {\"name\": \"text2image-image_generation\", \"description\": \"\\n    Image generation assistant, please imagine and describe a complete picture in detail based on my simple description. Then translate your detailed description into English\\n\\n    :param width: \u56fe\u7247\u5bbd\u5ea6\uff0c\u9ed8\u8ba41024\\n    :param height: \u56fe\u7247\u9ad8\u5ea6\uff0c\u9ed8\u8ba41024\\n    :param image_prompt:\u56fe\u7247\u63cf\u8ff0\uff0c\u9700\u8981\u82f1\u6587\\n    :return:\u56fe\u7247\u7684url\u5730\u5740\\n    \", \"parameters\": {\"properties\": {\"image_prompt\": {\"title\": \"Image Prompt\", \"type\": \"string\"}, \"width\": {\"default\": 1024, \"title\": \"Width\", \"type\": \"integer\"}, \"height\": {\"default\": 1024, \"title\": \"Height\", \"type\": \"integer\"}}, \"required\": [\"image_prompt\"], \"title\": \"image_generationArguments\", \"type\": \"object\"}}}\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>"
    },
    {
        "role": "user",
        "content": "I need to create a promotional banner for my upcoming conference. The banner should feature a futuristic cityscape at dusk with a glowing neon conference logo that reads \"Tech Summit 2024\" overlaid in the center. The image needs to be high resolution (at least 2000 by 1000 pixels) for print quality. Can you help me generate this visual from my description?"
    },
    {
        "role": "assistant",
        "content": "I'll create your promotional banner for Tech Summit 2024 with the futuristic cityscape at dusk and glowing neon logo. Based on your requirements for print quality, I'll use the higher resolution you specified.\n<tool_call>\n{\"name\": \"text2image-image_generation\", \"arguments\": {\"width\": 2000, \"height\": 1000, \"image_prompt\": \"A futuristic cyberpunk cityscape at dusk with towering skyscrapers covered in holographic advertisements and neon lights in blue, purple, and cyan colors. The sky has a gradient from deep violet at the top to warm orange and pink near the horizon where the sun is setting. In the center of the image, a glowing neon sign reads \\\"Tech Summit 2024\\\" in bright electric blue and cyan with a subtle glitch effect and light trails. The city below has sleek modern architecture with glass and metal surfaces reflecting the neon glow. Digital billboards and tech company logos are visible on building facades. The overall atmosphere is high-tech and professional with a dark futuristic aesthetic perfect for a technology conference banner. Sharp focus, ultra high resolution, HDR lighting, cinematic quality.\"}}\n</tool_call>"
    },
    {
        "role": "user",
        "content": "<tool_response>\n![image](https://image.pollinations.ai/prompt/A futuristic cyberpunk cityscape at dusk with towering skyscrapers covered in holographic advertisements and neon lights in blue, purple, and cyan colors. The sky has a gradient from deep violet at the top to warm orange and pink near the horizon where the sun is setting. In the center of the image, a glowing neon sign reads \"Tech Summit 2024\" in bright electric blue and cyan with a subtle glitch effect and light trails. The city below has sleek modern architecture with glass and metal surfaces reflecting the neon glow. Digital billboards and tech company logos are visible on building facades. The overall atmosphere is high-tech and professional with a dark futuristic aesthetic perfect for a technology conference banner. Sharp focus, ultra high resolution, HDR lighting, cinematic quality.?width=2000&height=1000&enhance=true&nologo=true&model=flux)\n</tool_response>"
    },
    {
        "role": "assistant",
        "content": "<think>\n\n</think>\n\nPerfect! I've generated your promotional banner for Tech Summit 2024. The image features:\n\n- **Futuristic cityscape at dusk** with towering skyscrapers and sleek modern architecture\n- **Vibrant neon lighting** in blue, purple, and cyan colors with holographic advertisements\n- **Glowing \"Tech Summit 2024\" neon logo** centered prominently in electric blue and cyan with glitch effects\n- **2000x1000 pixel resolution** for high-quality print use\n- **Cinematic quality** with professional HDR lighting and sharp focus\n- **Cyberpunk aesthetic** perfect for a technology conference\n\nThe image is ready for download and can be used directly for your conference promotional materials. The dark, high-tech atmosphere combined with the professional neon branding should effectively represent your Tech Summit 2024 event."
    }
]
"""
import json
import re
from copy import deepcopy
from tqdm import tqdm
from transformers import AutoTokenizer, AutoProcessor

tokenizer_path = "Qwen/Qwen3-4B"
print("tokenizer_path: ", tokenizer_path)


def read_json(file_path):
    """Read a JSON file and return the parsed data."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, file_path):
    """Save data to a formatted JSON file."""
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

def process_hf_data(raw_data):
    new_data = []
    for item in raw_data:
        new_item = deepcopy({
            "traj_type": item["traj_type"],
            "task_info": item["task_info"],
            "tools": json.loads(item["tools"]),
            "messages": json.loads(item["messages"]),
            "user_messages": json.loads(item["user_messages"]),
            "steps": item["steps"],
        })
        new_data.append(new_item)
    return new_data

def messages_to_text(tools, messages):
    """
    Convert messages (and optionally tools) to formatted text using the tokenizer's chat template.
    Args:
        tools (list or None): List of tool definitions (function schemas)
        messages (list): List of message dictionaries with roles and content
        
    Returns:
        str: Formatted text string using the chat template
    """
    # TODO: modify the tokenizer path by yourself
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tools is not None:
        text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False,
            add_generation_prompt=True,
            tools=tools)
    else:
        text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False,
            add_generation_prompt=True)
    return text


def text_to_messages(text):
    """Based on the special markers <|im_start|>role\ncontent<|im_end|>, the text is parsed back into messages containing only user, assistant, and content."""
    messages = []
    
    # Use regex to extract all role and content pairs from the template format
    # Pattern matches <|im_start|>role\ncontent<|im_end|>
    pattern = r'<\|im_start\|>(\w+)\n(.*?)<\|im_end\|>'
    matches = re.findall(pattern, text, re.DOTALL)
    
    for role, content in matches:
        content = content.strip()
        if role in ['system', 'user', 'assistant']:
            messages.append({
                'role': role,
                'content': content
            })
    return messages


def keep_think_history(messages_raw, messages_convert):
    """
    Preserve the thinking content for each assistant content in the messages.
    
    Most thinking models automatically strip out reasoning_content when applying chat templates.
    This function restores the reasoning_content by combining it with the response_content
    in the Qwen3 format: <think>\n{reasoning_content}\n</think>\n\n{response_content}
    """
    assert len(messages_raw) == len(messages_convert)
    messages_raw = deepcopy(messages_raw)
    messages_convert = deepcopy(messages_convert)
    new_messages = []
    for j in tqdm(range(len(messages_convert))):
        role = messages_convert[j]["role"]
        if role == "user" or role == "system":
            new_messages.append(deepcopy(messages_convert[j]))
        elif role == "assistant":
            # Restore reasoning_content from raw messages and combine with response_content
            reasoning_content = messages_raw[j]["reasoning_content"].strip()
            response_content = messages_convert[j]["content"].strip()
            content = f"<think>\n{reasoning_content}\n</think>\n\n{response_content}"
            new_messages.append({
                "role": "assistant",
                "content": content
            })
        else:
            print(f"role: {role} is not supported")
            raise ValueError(f"role: {role} is not supported")
    return new_messages
    

def main(read_file_path, save_file_path, is_keep_think_history=True):
    raw_data = read_json(read_file_path)
    data = process_hf_data(raw_data)
    new_data = []
    for item in tqdm(data):
        # Keep a copy of raw messages for restoring reasoning_content if needed
        messages_raw = deepcopy(item["messages"])
        new_item = deepcopy(item)
        
        # Convert messages with tools to formatted text, then parse back to extract plain messages
        raw_text = messages_to_text(item["tools"], item["messages"])
        new_messages = text_to_messages(raw_text)
        
        # Optional: Verify that conversion is consistent
        # new_text = messages_to_text(None, new_messages)
        # if not raw_text == new_text:
        #     print("warning: raw_text != new_text")
        #     continue
        
        # Optionally preserve reasoning_content for thinking models
        if is_keep_think_history:
            new_messages = keep_think_history(messages_raw=messages_raw, messages_convert=new_messages)
        
        new_item["messages"] = new_messages
        new_data.append(new_item)
    print("save_file_path: ", save_file_path)
    save_json(new_data, save_file_path)


if __name__ == "__main__":
    # Setting
    read_file_path = "your_path/envscaler_sft_traj_9k_metadata.json"
    is_keep_think_history = True
    save_file_path = read_file_path.replace(".json", f"_apply_qwen3_template.json")
    # Run
    main(read_file_path, save_file_path, is_keep_think_history)


        
