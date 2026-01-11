"""
step1: Infer state variables (state space) maintained by the environment from environment description and example task.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm
from copy import deepcopy

from utils.call_llm import llm_inference
from utils.process_file import read_file, save_file


# System prompt for state space inference
system_prompt = \
"""You are an expert task and environment analyst.  
Given an environment description and a example task in this environment, infer the set of state variables (state space) that the environment maintained.  

The state should not be too broad (e.g. "all possible data in an e-commerce system"), nor too narrow (only for this single task).  Instead, reasonably design it to support this task and similar tasks in the same environment.  

The input format is:

# Environment Summary
[Environment summary]

# Environment Introduction
[Environment introduction]

# A Example Task in This Environment
[Example task]


Your output must follow the format below (do not include any other text):

# Analysis
[Your thought process: What states are involved in the environment? What entities/attributes need to be tracked? What constraints or rules exist in the environment? ……]

# State Space Definition
- Entity: EntityName1  
  - Attributes: Attribute1, Attribute2, ...
  - Description: The role of this entity in the environment

- Entity: EntityName2
  - Attributes: ...
  - Description: ...

# Constraints & Rules
- Constraint 1
- Constraint 2
...
"""


# Example cases for few-shot learning
input_case_1 = \
"""Analyze the following task and environment, and infer the set of state variables (state space) that the environment maintained.  

# Environment Summary
E-commerce order management system

# Environment Introduction
This environment consists of a stateful backend for an e-commerce platform, managing products, orders, inventory, and order statuses.  
It keeps records of which products have been purchased in each order, tracks real-time stock quantities for all products, and stores fulfillment information for each order.  
These features make it the natural setting for inventory adjustments and order status updates in response to customer purchases.

# A Example Task in This Environment
Reduce stock quantity by 1 for every product purchased in order #58291 and mark the order as fulfilled.
"""

output_case_1 = \
"""# Analysis
This task requires knowing which products belong to a given order, the quantity of each product, and their current stock levels.  
It also requires an order to have a modifiable status.  
Therefore, the environment must maintain entities for orders, products, and inventory, along with attributes like stock quantity and order state.  

# State Space Definition
- Entity: Product  
  - Attributes: product_id, name, category, price, stock_quantity  
  - Description: Represents a product sold on the platform, with inventory tracking via stock_quantity.

- Entity: Order  
  - Attributes: order_id, customer_id, status, order_items  
  - Description: Represents a purchase order placed by a customer. Includes current order status and the list of items purchased.

- Entity: OrderItem 
  - Attributes: order_id, product_id, quantity  
  - Description: Represents the many-to-many relationship between orders and products, with quantities ordered.

- Entity: Customer 
  - Attributes: customer_id, name, account_status  
  - Description: Represents the user placing the order, useful for related tasks.

# Constraints & Rules
- Stock quantity cannot drop below 0.  
- Only orders with status = "pending" can be marked as "fulfilled".  
- Each product in an order must exist in the product inventory."""


# Input template for state space inference
input_template = \
"""Analyze the following task and environment, and infer the set of state variables (state space) that the environment maintained.  

# Environment Summary
{env_summary}

# Environment Introduction
{env_introduction}

# A Example Task in This Environment
{task}
"""


def parse_state_space_definition(state_space_definition):
    """Parse state space definition into list of dictionaries with entity, attributes, description."""
    if "- Entity: " not in state_space_definition:
        print(f"Error parsing state space definition: {state_space_definition}")
        return []
    state_space_definition = state_space_definition.split("- Entity: ")[1:]
    state_space_definition = [entity.split("\n") for entity in state_space_definition]
    if len(state_space_definition) == 0:
        print(f"Error parsing state space definition: {state_space_definition}")
        return []
    state_space_definition = [{
        "entity": entity[0].strip().strip("- Entity: "),
        "attributes": entity[1].strip().strip("- Attributes: "),
        "description": entity[2].strip().strip("- Description: ")
    } for entity in state_space_definition]
    return state_space_definition


def parse_constraints_rules(constraints_rules):
    """Parse constraints and rules by splitting on '- ' pattern."""
    if "- " not in constraints_rules:
        print(f"Error parsing constraints rules: {constraints_rules}")
        return []
    constraints_rules = constraints_rules.split("- ")[1:]
    constraints_rules = [constraint.strip() for constraint in constraints_rules]
    return constraints_rules


def parse_response(response):
    """Parse LLM response to extract analysis, state space definition, and constraints."""
    if "# Analysis" in response and "# State Space Definition" in response and "# Constraints & Rules" in response:
        try:
            analysis = response.split("# Analysis")[1].split("# State Space Definition")[0].strip()
            state_space_definition = response.split("# State Space Definition")[1].split("# Constraints & Rules")[0].strip()
            constraints_rules = response.split("# Constraints & Rules")[1].strip()
            return analysis, parse_state_space_definition(state_space_definition), parse_constraints_rules(constraints_rules)
        except Exception as e:
            print(f"Error parsing response: {e}")
            return response, None, None
    else:
        print(f"Error parsing response: {response}")
        return response, None, None

def process_env_item(env_item, model):
    """Process a single environment item to infer state space."""
    new_env_item = deepcopy(env_item)
    input_content = input_template.format(
        env_summary=env_item["environment_summary"],
        env_introduction=env_item["environment_introduction"],
        task=env_item["task"]
    )
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
                {"role": "user", "content": input_content}
            ]
        )
        analysis, state_space_definition, constraints_rules = parse_response(response)
        if analysis and state_space_definition and constraints_rules:
            break
    new_env_item["state_space_definition"] = state_space_definition
    new_env_item["constraints_rules"] = constraints_rules
    return new_env_item



if __name__ == "__main__":
    model = "gpt-4.1"
    read_file_path = "stage1_collect_env_from_task/final_result/env_description.json"
    save_file_path = "stage2_syn_env/temp_result/step1_infer_state.json"
    raw_data = read_file(read_file_path)
    new_data = []
    for env_item in tqdm(raw_data):
        new_env_item = process_env_item(env_item, model)
        new_data.append(new_env_item)
        # Save every 10 items
        if len(new_data) % 10 == 0:
            save_file(save_file_path, new_data)
    save_file(save_file_path, new_data)
