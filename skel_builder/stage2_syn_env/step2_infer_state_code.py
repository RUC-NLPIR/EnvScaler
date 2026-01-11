"""
step2: Convert environment description and state space definition to Python environment class definition.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
from tqdm import tqdm
from copy import deepcopy

from utils.call_llm import llm_inference
from utils.process_file import read_file, save_file


# System prompt for converting environment spec to Python class
system_prompt = \
"""You are an AI coding assistant.  
Your job is to translate an environment specification into a Python environment class definition.  
The class should simulate the stateful environment structure (without methods yet).  

You should analyze first and then generate code.

You should follow the rules of Analysis and Code to generate the code.

Rules of Analysis
- Determine the environment class name. It should be EnvironmentSummary or an appropriate adaptation (e.g., `LinuxFileSystem`, `EcommerceOrderSystem`).  
- Extract attribute names (comma-separated) from each entity in `state_space_definition`.
- If needed, generate a corresponding `TypedDict` using the extracted attributes, with attribute name → key and attribute value type → inferred from the appropriate Python primitive type (e.g., `id`=str, `name`=str, `category`=str, `price`/`size`=float/int, `quantity`=int, `status`=str, `timestamps`=str/float).
- `constraints_rules` is left as a comment.

Rules of Code
- Generates each `TypedDict` definition if needed.
- Generates the environment class (with only `__init__` and attributes), with attributes of type `Dict[ID, TypedDict]`.
- Add comments mapping each attribute back to the state space entity/attributes.
- Annotates the constraints in the code comments.
- Do not implement any business logic or methods yet.  

The input format is:
# Environment Summary
<short label, e.g. Linux filesystem, E-commerce order system>"

# Environment Introduction
<paragraph intro>

# State Space Definition
[
    {
      "entity": "EntityName",
      "attributes": "attr1, attr2, ...",
      "description": "short description"
    },
    ...
]

# constraints_rules
constraint 1 ...
constraint 2 ...
}

Your output must follow the format below (do not include any other text):

# Analysis
[Explains how to design Python environment classes based on tasks and state spaces (including class name selection, mapping entities to data structures, which fields are stored as dict/list, and how constraints are expressed through annotations)]

# Class Definition
```python
[Python environment class definition]
```"""


# Example cases for few-shot learning
input_case_1 = \
"""Given the following Environment, State Space, and Constraints, generate a Python environment class definition accordingly.

# Environment Summary
E-commerce order management system

# Environment Introduction
This environment represents an e-commerce order management system, where users can place orders, view products, and manage their accounts.

# State Space Definition
[
    {
      "entity": "Product",
      "attributes": "product_id, name, category, price, stock_quantity",
      "description": "Represents a product sold on the platform."
    },
    {
      "entity": "Order",
      "attributes": "order_id, customer_id, status, order_items",
      "description": "Represents a purchase order placed by a customer."
    },
    {
      "entity": "OrderItem",
      "attributes": "order_id, product_id, quantity",
      "description": "Intermediate entity linking products to orders."
    }
]

# constraints_rules
- tock quantity cannot drop below 0.
- Only orders with status = 'pending' can be marked as 'fulfilled'."""

output_case_1 = "# Analysis\nThe task involves updating inventory and order status. The environment is summarized as an \"e-commerce order management system,\" so the class is named `EcommerceOrderManagementSystem`.\n\nBased on the state space:\n- The Product entity needs to store a dict with key = product_id and value = metadata (including stock_quantity).\n- The Order entity needs to store a dict with key = order_id and value = metadata (including customer_id, status, and order_items).\n- The OrderItem entity represents a many-to-many relationship between Order and Product, ideally stored as {order_id: [{product_id, quantity}, ...]}.\n \nExtract Entity：  \n  • Product → {product_id: str, name: str, category: str, price: float, stock_quantity: int}  \n  • Order → {order_id: str, customer_id: str, status: str, order_items: List[OrderItemInfo]}  \n  • OrderItem → {order_id: str, product_id: str, quantity: int}  \n\nUse TypedDict to define these structures\n- In the environment：  \n  self.products: Dict[str, ProductInfo]  \n  self.orders: Dict[str, OrderInfo] \n \nConstraints such as \"stock ≥ 0\" and \"order status can only transition from pending to fulfilled\" are initially documented in the class as comments and later implemented in method implementations.\n\n# Class Definition\n```python\nfrom typing import Dict, List, TypedDict\n\nclass ProductInfo(TypedDict):\n    product_id: str\n    name: str\n    category: str\n    price: float\n    stock_quantity: int\n\nclass OrderItemInfo(TypedDict):\n    order_id: str\n    product_id: str\n    quantity: int\n\nclass OrderInfo(TypedDict):\n    order_id: str\n    customer_id: str\n    status: str\n    order_items: List[OrderItemInfo]\n\nclass EcommerceOrderManagementSystem:\n    def __init__(self):\n        \"\"\"\n        The environment for e-commerce order management.\n        \"\"\"\n\n        # Products: {product_id: ProductInfo}\n        self.products: Dict[str, ProductInfo] = {}\n\n        # Orders: {order_id: OrderInfo}\n        self.orders: Dict[str, OrderInfo] = {}\n\n        # Constraints reminder:\n        # - Stock quantity cannot drop below 0\n        # - Only orders with status = 'pending' can be marked as 'fulfilled'\n\n        self.current_user: dict = {}\n```"


# Input template for environment class generation
input_template = """Given the following Environment, State Space, and Constraints, generate a Python environment class definition accordingly.

# Environment Summary
{env_summary}

# Environment Introduction
{env_introduction}

# State Space Definition
{state_space_definition}

# constraints_rules
{constraints_rules}
"""  


def parse_response(response):
    """Parse LLM response to extract class definition."""
    if "# Analysis" in response and "# Class Definition" in response:
        try:
            analysis = response.split("# Analysis")[1].split("# Class Definition")[0].strip()
            class_definition = response.split("# Class Definition")[1].strip().lstrip("```python").rstrip("```")
            return True, class_definition
        except Exception as e:
            print(f"Error parsing response: {e}")
            return False, response
    else:
        print(f"Error parsing response: {response}")
        return False, response


def construct_messages(env_item):
    """Construct messages for LLM inference based on environment item."""
    state_space_definition_str = json.dumps(env_item["state_space_definition"], indent=4, ensure_ascii=False)
    constraint_str = ""
    for constraint in env_item["constraints_rules"]:
        constraint_str += f"- {constraint}\n"
    input_content = input_template.format(
        env_summary=env_item["environment_summary"], 
        env_introduction=env_item["environment_introduction"], 
        state_space_definition=state_space_definition_str,
        constraints_rules=constraint_str)
    messages = [
        {"role": "system", "content": system_prompt}, 
        {"role": "user", "content": input_case_1},
        {"role": "assistant", "content": output_case_1},
        {"role": "user", "content": input_content}
    ]
    return messages


def llm_infer(messages, model):
    """Generate class definition using LLM with retry mechanism."""
    cur_try = 0
    max_try = 5
    parse_success = False
    class_definition = ""
    while cur_try < max_try:
        response = llm_inference(
            provider="openai",
            model=model,
            messages=messages
        )
        parse_success, class_definition = parse_response(response)
        if parse_success:
            break
        cur_try += 1
    return class_definition


def process_env_item(env_item, model):
    """Process a single environment item to generate class definition."""
    new_env_item = deepcopy(env_item)
    messages = construct_messages(env_item)
    class_definition = llm_infer(messages, model)
    new_env_item["class_definition"] = class_definition
    return new_env_item


def main(read_file_path, save_file_path, model):
    """Main function: generate class definitions for all environments."""
    raw_data = read_file(read_file_path)
    new_data = []
    for env_item in tqdm(raw_data):
        new_env_item = process_env_item(env_item, model)
        new_data.append(new_env_item)
        # Save every 10 items
        if len(new_data) % 10 == 0:
            save_file(save_file_path, new_data)
    print("Save to file: {}".format(save_file_path))
    save_file(save_file_path, new_data)

if __name__ == "__main__":
    model = "gpt-4.1"
    read_file_path = "stage2_syn_env/temp_result/step1_infer_state.json"
    save_file_path = "stage2_syn_env/temp_result/step2_infer_state_code.json"
    main(read_file_path, save_file_path, model)
