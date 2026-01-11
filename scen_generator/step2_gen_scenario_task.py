"""
Step 2: Generate tasks for each scenario. Different models produce tasks with varying complexity and style.

We found clear differences in task complexity and style across models; try different ones to find the best fit for your needs.  
e.g., gpt-oss-120b produces more complex tasks than K2 or Qwen3-235B-Thinking, while GPT-5 delivers significantly higher quality and solvability.
"""
from tqdm import tqdm
from copy import deepcopy

from utils.process_file import read_file, save_file
from utils.call_llm import llm_inference
from utils.util import generate_timestamp


# Prompt template for generating tasks
input_template = """You are a task design expert, responsible for creating realistic, clear, and challenging tasks for a specific interactive environment.

You do not need to consider how the task will be executed; another execution expert will be responsible for completing it.

# Environment Introduction:  
{env_introduction}

# Environment State Definition:  
{env_state_definition}

# Supported Operation Interfaces:
{env_modify_operation}

# Environment Rules / Constraints:  
{env_rule}

# Current Environment Initial State / Database:  
{env_init_state}

# Task Design Requirements:
- **Realism**: The task must closely align with the current environment, reflect a plausible real-world scenario, and have a well-defined objective with business relevance.
- **Feasibility**: 
    i. The task must be based on and constrained by the current initial state of the environment. For example, you cannot delete a file that does not exist.
    ii. The task should be achievable through a combination of multiple operational interfaces supported by the environment. On one hand, the task description must include nessary information to enable successful completion (for example, if all tools only support actions by ID and can not get the ID anyway, the task should at least provide the ID). On the other hand, it should not include excessive information, so that the Agent is encouraged to use information query tools to obtain the necessary data rather than relying solely on the information provided in the task description.
    iii. Avoid tasks that require more than just the user interface. For example, timestamps might be automatically generated rather than modified through the user interface.
- **State Modification**: The goal of the task must involve modifying the current environment state/database; it cannot be limited to information queries only.
- **Challenge**: The task should be difficult enough that the agent needs to make multiple calls to information query tools and state modification tools to complete it. For example, the modification should have sufficient complexity (e.g., involving multiple objects, multiple attributes, or multi-condition combinations), and should not be achievable in a single simple step.
- **Compositionality**: The task must be composed of several distinct sub-tasks (No less than 50 words). Each sub-task must require multiple tool invocations to complete, ensuring layered complexity and interdependent steps across the task flow.
- **Clarity**: Use concise natural language to describe each sub-task. The description must be easy to understand and unambiguous. Since the ultimate goal is to modify the state, query tools are only part of the task-solving process and serve to provide information to the Agent; therefore, there is no need to include query requirements or intermediate steps in the task description.

Strictly follow the output format below:
# Analysis
[Your detailed step-by step reasoning and thought process, including environment description, current environment state, and supporting operation interfaces and so on.]

# Task
[Only the Task description, without any other analysis or explanation.]
"""

def parse_response(response):
    """Parse LLM response to extract task description."""
    parsed_success = False
    task = None
    if "</think>" in response:
        response = response.split("</think>")[1]
    if "# Analysis" in response:
        response = response.split("# Analysis")[1]
        if "# Task" in response:
            task = response.split("# Task")[1].strip("\n* ")
            parsed_success = True
    return parsed_success, task


def construct_prompt(env_item, init_state):
    """Construct prompt for task generation based on environment info and initial state."""
    operation_list = env_item["operation_list"]
    operation_str = ""
    for i, operation in enumerate(operation_list):
        operation_str += f"{i}.\nOperation Name: "+operation["operation_name"]+"\n"
        operation_str += "Operation Type: "+operation["operation_type"]+"\n"
        operation_str += "Description: "+operation["operation_description"]+"\n"

    env_rule_str = "- "+"\n- ".join(env_item["constraints_rules"])
    input_content = input_template.format(
        env_introduction=env_item["environment_introduction"], 
        env_state_definition=env_item["state_space_definition"], 
        env_modify_operation=operation_str,
        env_rule=env_rule_str, 
        env_init_state=init_state)
    return input_content


class GenTaskAgent:
    """Agent for generating tasks using LLM."""
    
    def __init__(self, env_item, model, temperature=0.5):
        self.env_item = deepcopy(env_item)
        self.model = model
        self.temperature = temperature
        
    def gen_task(self, init_config):
        """Generate a task for the given initial config."""
        prompt = construct_prompt(self.env_item, init_config)
        input_messages = [{"role": "user", "content": prompt}]
        cur_try = 0
        max_try = 3
        while cur_try < max_try:
            cur_try += 1
            response = llm_inference(
                provider="openai",
                model=self.model,
                messages=input_messages,
                temperature=self.temperature
            )
            parsed_success, task = parse_response(response)
            if parsed_success:
                break
        return task


def process_env_item(env_id, env_item, env_all_configs, gen_num, model, temperature):
    """Generate multiple tasks for a single environment."""
    task_info_list = []
    for gen_id in tqdm(range(gen_num)):
        init_config = env_all_configs[gen_id]
        init_config_copy = deepcopy(init_config)
        agent = GenTaskAgent(env_item, model, temperature)
        task = agent.gen_task(init_config)
        if task == None or task == "":
            continue
        task_info_list.append(deepcopy({
            "env_id": env_id,
            "env_class_name": env_item["env_class_name"],
            "task_id": generate_timestamp(),
            "init_config": init_config_copy,
            "task": task
        }))
    return task_info_list


def main(env_data_dict, env_config_dict, save_file_path, gen_num, model, env_ids, temperature):
    """Main function: generate tasks for multiple environments."""
    new_data = []
    for env_id in env_ids:
        env_item = env_data_dict[env_id]
        env_all_configs = env_config_dict[env_id]
        assert len(env_all_configs) >= gen_num
        print(f"env_id: {env_id}, env_class_name: {env_item['env_class_name']},  all configs num: ", len(env_all_configs))
        process_result = process_env_item(env_id=env_id, env_item=env_item, env_all_configs=env_all_configs, gen_num=gen_num, model=model, temperature=temperature)
        new_data.extend(process_result)
        save_file(save_file_path, new_data)
    print(f"save_file_path: {save_file_path}")
    save_file(save_file_path, new_data)


if __name__ == "__main__":
    env_data = read_file("your_path/filtered_env_metadata.json")
    env_config_data = read_file("temp_result/step1_init_env_config.json")
    env_config_dict = {item["env_id"]: item["init_config_list"] for item in env_config_data}
    env_ids = [item["env_id"] for item in env_config_data]
    gen_num = 3 # Number of tasks to generate per environment
    model = "gpt-4.1"
    # model = "gpt-oss-120b" 
    temperature = 0.7
    save_file_path = "temp_result/step2_gen_task.json"
    main(env_data, env_config_dict, save_file_path, gen_num, model, env_ids, temperature)

