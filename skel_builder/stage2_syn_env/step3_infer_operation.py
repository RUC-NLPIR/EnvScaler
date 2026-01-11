"""
step3: Infer the list of operations needed to support tasks in the environment from environment description and example task.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm
from copy import deepcopy

from utils.call_llm import llm_inference
from utils.process_file import read_file, save_file


# System prompt for operation list inference
system_prompt = \
"""You are an expert in building and analyzing agent environments.
Given an environment summary, introduction, state space definition, constraint rules, Python base class definition, and example task, your goal is to analyze the current environment and then generate the list of operations needed to support the task in this environment (including information query class and state modification class).
Each operation will be converted into a class function for the Agent to use in subsequent steps.


Key Points:  
- Operations are divided into 2 categories: **Information Query Class** and **State Change Class**.  
- Each operation includes: operation name + brief description.  
- Before output, you must first write **# Analysis**: explain task logic → which are query operations, which are state change operations → and how constraints are related.  

Input Format:
Based on the following environment specification, produce the operation list.

{
  "environment_summary": "...",
  "environment_introduction": "...",
  "state_space_definition": [...],
  "constraints_rules": [...],
  "environment_class_definition": "...",
  "environment_example_task": "...",
}



Strictly maintain the following Output Format:

# Analysis
[Explain operation requirements + classification logic + how constraints affect + ……]

# Operation List
## Information Query Class
- Operation: OperationName Description: xxxx  
- Operation: OperationName Description: xxxx 
- ……

## State Change Class
- Operation: OperationName Description: xxxx  
- Operation: OperationName Description: xxxx  
- ……"""


# Example cases for few-shot learning
input_case_1 = \
"""Based on the following environment specification, produce the operation list.

  {
    "environment_summary": "E-commerce order management system",
    "environment_introduction": "This environment models the backend of an e-commerce platform, where users can browse, place, and manage orders.  \nIt keeps track of user accounts, their order histories, and order statuses such as pending, shipped, or completed.  \nOperations like cancelling orders are tightly regulated by the system's business rules and state transitions, making it a natural setting for this kind of task.",
    "state_space_definition": [
      {
        "entity": "User",
        "attributes": "user_id, username, email, account_status, registration_date",
        "description": "Represents a registered customer who can place orders, with associated identity and status."
      },
      {
        "entity": "Order",
        "attributes": "order_id, user_id, order_date, status, order_items",
        "description": "Represents an order placed by a user, including time, current status (pending, shipped, completed, cancelled), and the items in the order."
      },
      {
        "entity": "OrderItem",
        "attributes": "order_id, product_id, quantity",
        "description": "Represents individual product items and quantities included in an order."
      }
    ],
    "constraints_rules": [
      "Only the user who placed an order may request its cancellation.",
      "Only orders with status \"pending\" (not \"shipped,\" \"completed,\" or \"cancelled\") are eligible for cancellation.",
      "There must be a clear ordering of orders per user (e.g., via order_date) to identify the most recent order.",
      "Changing an order's status to \"cancelled\" must follow the status transition rules (e.g., not possible if already shipped or completed)."
    ],
    "environment_class_definition": "\nfrom typing import Dict, List, TypedDict\n\nclass UserInfo(TypedDict):\n    user_id: str\n    username: str\n    email: str\n    account_status: str\n    registration_date: str\n\nclass OrderItemInfo(TypedDict):\n    order_id: str\n    product_id: str\n    quantity: int\n\nclass OrderInfo(TypedDict):\n    order_id: str\n    user_id: str\n    order_date: str  # ISO date or timestamp as string\n    status: str      # e.g., \"pending\", \"shipped\", \"completed\", \"cancelled\"\n    order_items: List[OrderItemInfo]\n\nclass EcommerceOrderManagementSystem:\n    def __init__(self, init_config: dict):\n        \"\"\"\n        Backend environment for an e-commerce platform.\n        \"\"\"\n        # Users: {user_id: UserInfo}\n        self.users: Dict[str, UserInfo] = {}\n\n        # Orders: {order_id: OrderInfo}\n        self.orders: Dict[str, OrderInfo] = {}\n\n        # Constraints/rules:\n        # - Only the user who placed an order may request its cancellation.\n        # - Only orders with status \"pending\" (not \"shipped\", \"completed\", \"cancelled\") are eligible for cancellation.\n        # - There must be a clear ordering of orders per user (e.g., via order_date) to identify the most recent order.\n        # - Changing an order's status to \"cancelled\" must follow status transition rules (not possible if already shipped/completed).\n\n        self.init_config = init_config\n",
    "environment_example_task": "Cancel the most recent order placed by the user `alice123` if it has not yet been shipped.",
  }"""

output_case_1 = \
"""# Analysis
An example task in an environment is: find user alice123, confirm their recent order, and cancel it when the order status is pending.  
Required operations include: get user → get order set → sort to confirm most recent → check status → update to cancelled.  

However, considering that the environment is an "e-commerce order management system" involving multi-dimensional information about users, orders, and order items, the system needs to support more operations, such as:  
- For user layer: besides querying users by username, it can also list all users, query users by user_id, and check user account status.  
- For order layer: besides querying order lists, it may also need to filter by status, get product lists for orders, view order history status changes, and count orders.  
- For order item layer: besides listing order items, it may also need to update product quantities or remove specific entries.  
- For constraints: additional validation operations may be defined, such as verifying whether an order belongs to the user, verifying whether cancellation complies with rules.  
- Environment redundant operations: such as restoring cancelled orders, admin deleting orders, bulk cancelling multiple orders, etc. - these operations may not be needed for the current task, but can improve the richness of the training environment.  

Therefore, the operation set should not only cover the current task path, but also reflect the diverse operational aspects of the environment.

# Operation List
## Information Query Class
- Operation: get_user_by_username  
  Description: Retrieve user info (id, email, status) by username.  

- Operation: get_user_by_id  
  Description: Retrieve user info by unique user_id.  

- Operation: list_all_users  
  Description: Retrieve the list of all registered users in the system.  

- Operation: check_user_account_status  
  Description: Query the account status of a user (active, suspended, etc.).  

- Operation: list_user_orders  
  Description: Retrieve all orders placed by a specific user.  

- Operation: list_orders_by_status  
  Description: Retrieve all orders of a user filtered by status (pending, shipped, etc.).  

- Operation: get_most_recent_order  
  Description: Identify the most recent order for a user based on order_date.  

- Operation: get_order_by_id  
  Description: Retrieve full details of an order given its order_id.  

- Operation: get_order_status  
  Description: Check the current status of an order.  

- Operation: get_order_items  
  Description: List all line items (products and quantities) in an order.  

- Operation: get_order_history  
  Description: Show the chronological status change history of a given order.  

## State Change Class
- Operation: cancel_order  
  Description: Update the status of an eligible order to "cancelled".  

- Operation: update_order_status  
  Description: Change the status of an order to any valid value (pending, shipped, completed, cancelled).  

- Operation: bulk_cancel_orders  
  Description: Cancel multiple pending orders from the same user.  

- Operation: reopen_cancelled_order  
  Description: Revert a cancelled order back to pending, if allowed.  

- Operation: modify_order_items  
  Description: Update or remove specific items in an order.  

- Operation: delete_order  
  Description: Remove an order permanently (admin-level action).  

- Operation: restore_order  
  Description: Restore a previously deleted order if metadata is retained."""

input_template ="Based on the following environment specification, produce the operation list.\n\n{env_info}"

def parse_operation_list(operation_list_str, operation_type):
    """Parse operation list string into list of dictionaries with operation name, description, type."""
    raw_operations = operation_list_str.split("- Operation: ")
    raw_operations = [operation for operation in raw_operations if operation.strip()]
    operations = []
    for operation in raw_operations:
        operation_name = operation.split("Description: ")[0].strip()
        operation_description = operation.split("Description: ")[1].strip()
        operations.append({
            "operation_name": operation_name,
            "operation_description": operation_description,
            "operation_type": operation_type
        })
    return operations


def parse_response(response):
    """Parse LLM response to extract analysis and operation list."""
    if "# Analysis" in response and "# Operation List" in response and "## Information Query Class" in response and "## State Change Class" in response:
        try:
            analysis = response.split("# Analysis")[1].split("# Operation List")[0].strip()
            query_operation_str = response.split("## Information Query Class")[1].split("## State Change Class")[0].strip()
            state_change_operation_str = response.split("## State Change Class")[1].strip()
            query_operation_list = parse_operation_list(query_operation_str, "query")
            state_change_operation_list = parse_operation_list(state_change_operation_str, "state_change")
            operation_list = query_operation_list + state_change_operation_list
            return analysis, operation_list
        except Exception as e:
            print(f"Error parsing response: {e}")
            return "Error parsing response:" + response, []
    else:
        print(f"Error parsing response: {response}")
        return "Error parsing response:" + response, []


def process_env_item(env_item, model):
    """Process a single environment item to infer operation list."""
    new_env_item = deepcopy(env_item)
    env_info = {
        "environment_summary": env_item["environment_summary"],
        "environment_introduction": env_item["environment_introduction"],
        "state_space_definition": env_item["state_space_definition"],
        "constraints_rules": env_item["constraints_rules"],
        "environment_class_definition": env_item["class_definition"],
        "environment_example_task": env_item["task"]
    }
    input_content = input_template.format(env_info=env_info)
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
    analysis, operation_list = parse_response(response)
    new_env_item["operation_list"] = operation_list
    return new_env_item


def main(read_file_path, save_file_path, model):
    """Main function: generate operation lists for all environments."""
    raw_data = read_file(read_file_path)
    new_data = []
    for env_item in tqdm(raw_data):
        # Skip if operation_list already exists
        if "operation_list" in env_item and env_item["operation_list"]:
            new_data.append(env_item)
            continue
        new_env_item = process_env_item(env_item, model)
        new_data.append(new_env_item)
        # Save every 10 items
        if len(new_data) % 10 == 0:
            save_file(save_file_path, new_data)
    print("Save to file: {}".format(save_file_path))
    save_file(save_file_path, new_data)


if __name__ == "__main__":
    model = "gpt-4.1"
    read_file_path = "stage2_syn_env/temp_result/step2_infer_state_code.json"
    save_file_path = "stage2_syn_env/temp_result/step3_infer_operation.json"
    main(read_file_path, save_file_path, model)
