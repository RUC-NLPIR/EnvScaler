"""
step2: process messages to alpaca format with masked history.

In thinking mode, Qwen3 automatically removes all rounds' reasoning process when applying the chat_template.
To let the model learn the reasoning in each round, we split an n-round sample into n sub-samples (corresponding rounds are 1,2,...,n). 
Only the final round's reasoning and action in each sub-sample are supervised and optimized (implemented via the mask\_history hyperparameter in LlamaFactory).

For example, a 3-turn conversation becomes 3 samples:
- Sample 1: instruction=Turn1, output=Response1, history=[]
- Sample 2: instruction=Turn2, output=Response2, history=[Turn1, Response1]
- Sample 3: instruction=Turn3, output=Response3, history=[Turn1, Response1, Turn2, Response2]
"""
import random
import json
from copy import deepcopy

def read_json(file_path):
    """Read JSON data from a file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def save_json(file_path, data):
    """Save data to a JSON file with formatted output."""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    

def process_alpaca_mask_history(traj_data, remove_think_history=True, remove_think=False):
    """
    Convert multi-turn conversations to multiple Alpaca format samples with masked history.
    
    This function splits a single multi-turn conversation into multiple training samples.
    Each sample trains only on one turn, with previous turns as history.
    """
    new_data = []
    for traj_item in traj_data:
        msgs = deepcopy(traj_item["messages"])

        # Basic length consistency check
        max_turn = len(msgs)
        assert len(msgs) >= 3
        
        # Role validation: first message must be system, then alternating user/assistant
        assert msgs[0]["role"] == "system"
        for i in range(1, max_turn):
            if i % 2 == 1:  # Odd indices => user
                assert msgs[i]["role"] == "user" 
            else:  # Even indices => assistant
                assert msgs[i]["role"] == "assistant"
                
        # If last message is from user, remove it (incomplete conversation)
        if msgs[-1]["role"] == "user":
            msgs = msgs[:-1]

        system = msgs[0]["content"]
        cur_history = []
        
        i = 1
        # Process each user-assistant pair as a separate sample
        while i < len(msgs) :
            if msgs[i]["role"] == "user":
                cur_instruction = msgs[i]["content"]
                cur_output = msgs[i+1]["content"]
                
                # Remove reasoning content from current output if requested
                if remove_think:
                    if '</think>' in cur_output:
                        cur_output = cur_output.split('</think>')[-1].strip()
                
                # Create a sample for this turn with current history
                new_data.append(
                    deepcopy({
                        "instruction": cur_instruction,
                        "input": "",
                        "output": cur_output,
                        "system": system,
                        "history": cur_history,
                    })
                )
                
                # Remove reasoning content from output when adding to history (if requested)
                if remove_think_history:
                    if '</think>' in cur_output:
                        cur_output = cur_output.split('</think>')[-1].strip()
                
                # Add current turn to history for next sample
                cur_history.append(
                    [
                        cur_instruction,
                        cur_output,
                    ]
                )
            i += 2
    return new_data


if __name__ == "__main__":
    # Configuration: input file path
    file_path = "your_path/envscaler_sft_traj_9k_metadata_apply_qwen3_template.json"
    # Read input data
    data = read_json(file_path)
    print(len(data))
    # Convert to Alpaca format with masked history
    new_data = process_alpaca_mask_history(data)
    print(len(new_data))
    # Shuffle data for training
    random.shuffle(new_data)
    # Save output
    save_json("your_path/alpaca_mask_history_envscaler_sft_traj_9k.json", new_data)
    save_json("your_path/alpaca_mask_history_envscaler_sft_traj_9k_top100.json", new_data[:100])