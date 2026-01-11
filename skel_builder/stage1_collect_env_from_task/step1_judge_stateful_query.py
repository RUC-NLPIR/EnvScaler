"""
step1: Judge whether a task depends on a stateful and domain-specific environment.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.call_llm import llm_inference
from utils.process_file import read_file, save_file


# System prompt for LLM to judge stateful tasks
system_prompt = """You are a system that filters natural language tasks to determine if they are **state-dependent, actionable requests** within a **persistent, domain-specific environment**.

### Core Definition
We are ONLY looking for tasks that meet **all** of the following criteria:

1. **Persistent Environment** — The query is about a domain where:
   - There is a live, ongoing state that can be read or changed
   - The environment supports both:
     a) Information queries about current state (read operations)  
     b) Explicit state-changing actions (create, update, delete, move, cancel, etc.)

2. **State Dependency** — The task cannot be answered correctly without:
   - Inspecting the actual current data or configuration in the environment, and/or
   - Executing an operation that modifies that data.

3. **Domain Specificity** — The environment is not general-purpose knowledge; it is a structured system such as:
   - File management system with stored files/folders
   - Order/logistics tracking system
   - Calendar/scheduling system
   - CRM, inventory, ticketing, project management tools
   - Other specialized platforms with records that persist over time

4. **Actionability in Context** — The query must correspond to an actionable operation or status check **within the actual environment** (not hypothetical).

### Eligible Task Types
- **State queries**: "Is invoice #1024 paid?" / "What meetings are scheduled for Wednesday?"
- **State modification operations**: "Upload the proposal.pdf to the project folder" / "Cancel order #4512" / "Move meeting to 3 PM"

### Explicit Exclusions
A request is **NOT eligible** if it is:
- Open-domain factual Q&A unrelated to a live state ("Who invented AI?", "What’s the capital of France?")
- Casual conversation ("How are you?", "What's the weather?")
- Content creation ("Write me a story", "Make a poem")
- Pure hypothetical without actual environment interaction
- Isolated reasoning or calculations without accessing persisted state

### Judgment Rule — Be strict:
Choose **YES** only if:
- The query **cannot** be answered from general knowledge alone
- AND it **requires real-time access** to persistent state in a domain-specific environment
- AND it **targets an actionable operation** (either a read or a write to that environment)
- AND the environment has the capability for both queries and modifications

If any criterion is missing → **NO**.

---

### Task
Given a query, first **analyze** whether it implies or requires:
- A domain-specific environment with both query and modification capabilities
- Accessing or updating persistent state
- Performing a concrete, actionable operation

Then give your final judgment.

### Output Format (Strictly enforce)
# Analysis
<Detailed reasoning whether this query depends on persistent state, involves a stateful operation, and needs a capable environment as defined>

# Answer
YES   --> Only if all strict criteria are met  
NO    --> Otherwise
"""

# Input template for task judgment
input_template = \
"""Analyze the following task and determine if it is a task that depends on a stateful and domain-specific environment.

{query}
"""


def parse_response(response: str):
    """Parse LLM response to extract analysis and judgment result."""
    if "# Analysis" not in response or "# Answer" not in response:
        return "parsed_failed", False
    analysis = response.split("# Analysis")[1].split("# Answer")[0].strip()
    answer = response.split("# Answer")[1].strip()  
    if "YES" in answer:
        return analysis, True
    elif "NO" in answer:
        return analysis, False
    else:
        return "parsed_failed", False


def process_query(query, model):
    """Process a single query to judge if it's stateful."""
    query = query.strip()
    response = llm_inference(
        provider="openai",
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": input_template.format(query=query)}
        ],
    )
    analysis, answer = parse_response(response)
    return {
        "task": query,
        "judge_analysis": analysis,
        "judge_result": answer
    }

def main(tasks, save_file_path, model, max_workers=5):
    """Main function: process tasks in parallel and save results periodically."""
    tasks = [q.strip() for q in tasks]
    new_data = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_query, query, model): query for query in tasks}
        for i, future in enumerate(tqdm(as_completed(futures), total=len(futures))):
            try:
                result = future.result()
                new_data.append(result)
            except Exception as e:
                print(f"Error processing query '{futures[future]}': {e}")

            # Save every 100 items
            if len(new_data) % 100 == 0:
                save_file(save_file_path, new_data)

    # Final save
    save_file(save_file_path, new_data)

if __name__ == "__main__":
    source_tasks = read_file("stage1_collect_env_from_task/temp_result/step0_source_tasks.json")
    source_tasks = [task["task"] for task in source_tasks]
    model = "gpt-4.1"
    save_file_path = "stage1_collect_env_from_task/temp_result/step1_stateful_task_judge.json"
    print(f"save_file_path: {save_file_path}")
    main(
        tasks=source_tasks,
        save_file_path= save_file_path, 
        model = model,
        max_workers=3
    )