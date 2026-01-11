"""
step2: Infer the most likely stateful and domain-specific environment for a given task.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import re
from tqdm import tqdm
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.call_llm import llm_inference
from utils.process_file import read_file, save_file


# System prompt for environment inference
system_prompt = \
f"""You are a Task Analyst.  
Given a raw task description, your objective is to identify the most plausible stateful and domain-specific environment in which this task would naturally occur.  

The chosen environment should strike a balance: not so broad as to be meaningless, and not so narrow as to apply only to a single, highly specific case. It should be scoped such that this task, along with similar related tasks, can be executed meaningfully.  

Guidelines:  
- If multiple environments seem equally plausible, select one at random rather than listing all possibilities.  
- Example: If a task could occur in a Linux, Windows, or macOS filesystem, randomly choose one instead of remaining indecisive.  

Your response must include the following sections:  

1. # Analysis  
   - Explain the reasoning process used to connect the task to the chosen environment.  
   - Note any relevant entities, constraints, relationships, or dynamics implied by the task.  

2. # Environment Summary  
   - Provide a concise label for the environment type.  
   - Examples: *Linux filesystem*, *E-commerce order management system*, *Airline booking system*.  

3. # Environment Introduction  
   - Introduce the environment itself, without referring to the current task.  
   - Focus on its inherent structure, the nature of the state it maintains, typical operations it supports, and its general scope in real-world usage.  
   - Limit to approximately three sentences.  

4. # Metrics  
   - Usefulness: 1-10  
     Reflects how broadly applicable and valuable this environment is in real-world scenarios. Higher scores indicate environments relevant to many contexts and industries.  
   - Modelability: 1-10  
     Indicates how straightforward it would be to represent this environment using a single Python class, with attributes holding state and methods performing reading, writing, and querying operations. Higher scores indicate simpler, more self-contained structures.  

Your response must follow exactly this format, with no additional text or commentary:

# Analysis
[Your analysis]
# Environment Summary
[Your environment summary]
# Environment Introduction
[Your environment introduction]
# Metrics
Usefulness: [1-10]
Modelability: [1-10]
"""

# Example cases for few-shot learning
input_case_1 = "Analyze the following task and infer the most plausible stateful environment in which this task would naturally take place: \nDelete all files under the directory `/home/test` that have names containing the substring `wayland`."
output_case_1 = "# Analysis\nThe task involves identifying files within a specific directory path that match a given name pattern, then performing deletion operations on them. The reference to `/home/test` indicates a typical home directory in a Unix-like operating system. This points to direct interaction with a hierarchical filesystem, where directories contain files and metadata such as permissions and timestamps, and support common manipulation actions.  \nThe Usefulness score is high (9) because Linux filesystems are extremely common in servers, desktops, embedded systems, and cloud environments, making the environment relevant to a vast range of real-world tasks.  \nThe Modelability score is set to 8 because the core aspects of a filesystem—such as directory structures, file metadata, and basic operations like search and delete—can be effectively represented in a single Python class for simulation purposes. While fully replicating a production filesystem would be complex, capturing its essential structure and common operations is relatively straightforward.\n\n# Environment Summary\nLinux filesystem\n\n# Environment Introduction\nA Linux filesystem is organized as a hierarchical tree structure starting from the root directory. It maintains metadata such as file names, paths, permissions, ownership, timestamps, and file types. This environment supports a wide range of operations including creation, modification, deletion, and traversal of files and directories, making it versatile for managing persistent data on Unix-like systems.\n\n# Metrics\nUsefulness: 9\nModelability: 8"

input_case_2 = "Analyze the following task and infer the most plausible stateful and domain-specific environment in which this task would naturally take place: \nI need to check the antifungal susceptibility of a few fungal species for different antifungal agents. Can you help with that? Here are the details:\n1. Candida albicans for Fluconazole\n2. Aspergillus fumigatus for Amphotericin B\n3. Trichophyton rubrum for Terbinafine"
output_case_2 = "# Analysis\nThe task requires assessing fungal species' responses to various antifungal agents. This involves precise microbiological procedures, structured experimental records, and storage of laboratory results. Such operations fit well within a microbiology Laboratory Information Management System (LIMS), which can manage sample metadata, test protocols, and results. However, the environment's real-world applicability is narrower compared to general-purpose systems like databases or filesystems—it is relevant mainly to specialized labs in clinical or research settings, which limits its broader usefulness. Additionally, a full LIMS typically contains multiple interconnected modules, complex workflows, and external integrations, making it challenging to model comprehensively in a single Python class without heavily simplifying its functionality.\n\n# Environment Summary\nMicrobiology Laboratory Information Management System (LIMS)\n\n# Environment Introduction\nA microbiology LIMS is a specialized software environment designed to track and organize laboratory data related to microbial cultures and tests. It maintains structured records of species profiles, reagents, testing conditions, and observed results. The system supports data entry, retrieval, and reporting functions, enabling research and clinical laboratories to manage experiments efficiently.\n\n# Metrics\nUsefulness: 4\nModelability: 3\n"

# Input template for task inference
input_template = "Analyze the following task and infer the most plausible stateful and domain-specific environment in which this task would naturally take place: \n{task}"


def parse_response(response):
    """Parse LLM response to extract environment inference results."""
    result = {
        "analysis": None,
        "environment_summary": None,
        "environment_introduction": None,
        "metrics": {
            "usefulness": None,
            "modelability": None
        }
    }

    # Check if required sections exist
    required_sections = ["# Analysis", "# Environment Summary", "# Environment Introduction", "# Metrics"]
    if all(section in response for section in required_sections):
        try:
            # Parse each field
            result["analysis"] = response.split("# Analysis")[1].split("# Environment Summary")[0].strip()
            result["environment_summary"] = response.split("# Environment Summary")[1].split("# Environment Introduction")[0].strip()
            result["environment_introduction"] = response.split("# Environment Introduction")[1].split("# Metrics")[0].strip()

            # Extract scores
            metrics_text = response.split("# Metrics")[1].strip()
            usefulness_match = re.search(r"Usefulness:\s*(\d+)", metrics_text)
            modelability_match = re.search(r"Modelability:\s*(\d+)", metrics_text)

            if usefulness_match:
                result["metrics"]["usefulness"] = int(usefulness_match.group(1))
            if modelability_match:
                result["metrics"]["modelability"] = int(modelability_match.group(1))

            return True, result

        except Exception as e:
            print(f"Error parsing response: {e}")
            return False, result
    else:
        print("Error: Missing required section headers in response.")
        return False, result




def process_item(item, model):
    """Process a single item to infer its environment."""
    new_item = deepcopy(item)
    task = item["task"]
    cur_try = 0
    max_try = 3
    while cur_try < max_try:
        cur_try += 1
        response = llm_inference(
            provider="openai",
            model=model, 
            messages=[
                {"role": "system", "content": system_prompt}, 
                {"role": "user", "content": input_case_1},
                {"role": "assistant", "content": output_case_1},
                {"role": "user", "content": input_case_2},
                {"role": "assistant", "content": output_case_2},
                {"role": "user", "content": input_template.format(task=task)}
            ]
        )
        success, result = parse_response(response)
        if success:
            break
    new_item.update(result)
    return new_item


def main(read_file_path, save_file_path, model, num_workers=1):
    """Main function: process tasks in parallel and save results periodically."""
    raw_data = read_file(read_file_path)
    # Keep only items with judge_result=True
    raw_data = [item for item in raw_data if item["judge_result"]]
    new_data = []

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_item, item, model): item for item in raw_data}
        for i, future in enumerate(tqdm(as_completed(futures), total=len(futures))):
            try:
                new_item = future.result()
                new_data.append(new_item)
            except Exception as e:
                print(f"Error processing item '{futures[future]}': {e}")

            # Save every 10 items
            if len(new_data) % 10 == 0:
                save_file(save_file_path, new_data)

    # Final save
    save_file(save_file_path, new_data)
    
if __name__ == "__main__":
    read_file_path = "stage1_collect_env_from_task/temp_result/step1_stateful_task_judge.json"
    model = "gpt-4.1"
    num_workers = 3
    save_file_path = "stage1_collect_env_from_task/temp_result/step2_infered_env_description.json"
    main(read_file_path, save_file_path, model, num_workers=num_workers)