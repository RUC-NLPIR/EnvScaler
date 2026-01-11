"""
Environment builder demo - builds environment skeleton from task description through multiple stages.
"""
import json
import time
from rich.console import Console
from rich.json import JSON
from rich.syntax import Syntax
from dotenv import load_dotenv
from stage1_collect_env_from_task.step1_judge_stateful_query import process_query as stage1_step1_juge_task
from stage1_collect_env_from_task.step2_infer_env_topic import process_item as stage1_step2_infer_env_topic
from stage2_syn_env.step1_infer_state import process_env_item as stage2_step1_process_env_item
from stage2_syn_env.step2_infer_state_code import process_env_item as stage2_step2_process_env_item
from stage2_syn_env.step3_infer_operation import process_env_item as stage2_step3_process_env_item
from stage2_syn_env.step4_infer_func_code import process_env_item_for_demo as stage2_step4_process_env_item
from stage2_syn_env.step5_concat import process_env_item as stage2_step5_process_env_item
from stage2_syn_env.step6_analysis_env_class_code import process_env_item as stage2_step6_process_env_item

def read_json(file_path):
    """Read JSON file and return parsed data."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def save_json(file_path, data):
    """Save data to JSON file."""
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
        

def pretty_print(content, style='str', bold=False, color=None, width=None):
    """
    Pretty print content with syntax highlighting and formatting.
    
    Args:
        content: Content to print (string or object)
        style: Style mode - 'json', 'python', or 'str'
        bold: Whether to use bold text (str mode only)
        color: Text color (str mode only, default: cyan)
        width: Display width (None for auto)
    """
    
    console = Console(width=width, soft_wrap=True)
    
    if style.lower() == 'json':
        try:
            # Parse JSON content
            if isinstance(content, str):
                data = json.loads(content)
            else:
                data = content
            
            json_obj = JSON.from_data(data, indent=4, ensure_ascii=False)
            console.print(json_obj)
            
        except json.JSONDecodeError as e:
            console.print(f"[yellow]JSON parse failed, showing raw content:[/yellow]")
            console.print(content)
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            console.print(content)
    
    elif style.lower() == 'python':
        try:
            # Convert to string if needed and apply syntax highlighting
            code_str = str(content) if not isinstance(content, str) else content
            
            syntax = Syntax(
                code_str, 
                'python',
                theme="monokai",
                line_numbers=True,
                word_wrap=True,
                indent_guides=True
            )
            console.print(syntax)
        except Exception as e:
            console.print(f"[red]Code rendering error:[/red] {e}")
            console.print(content)

    elif style.lower() == 'str':
        # Set text color
        txt_color = color if color is not None else "cyan"
        
        # Apply bold formatting if requested
        if bold:
            console.print(f"[bold {txt_color}]{content}[/bold {txt_color}]")
        else:
            console.print(f"[{txt_color}]{content}[/{txt_color}]")
    else:
        raise ValueError(f"Unsupported style: {style}, use 'json', 'python' or 'str'")



def main(env_item_raw, model, save_file_path):
    """Main pipeline: build environment skeleton from task through multiple processing stages."""
    assert save_file_path.endswith(".json")
    pretty_print(f"A Existing Task from {env_item_raw['task_from']}:\n{env_item_raw['task']}", style='str', color='magenta')
    
    # Stage 1 - Step 1: Task analysis and filtering
    print("-" * 80)
    print("[Stage 1 - Step 1] Task Analysis")
    judge_result = stage1_step1_juge_task(query=env_item_raw['task'], model=model)
    pretty_print({"analysis": judge_result["judge_analysis"], "keep_task": judge_result["judge_result"]}, style='json')
    if not judge_result["judge_result"]:
        pretty_print("The task does not meet the requirements; the process ends.", style='str', color='red')
        return
    
    # Stage 1 - Step 2: Infer environment topic and description
    print("-" * 80)
    print("[Stage 1 - Step 2] Infer Environment Topic")
    env_item_with_env_description = stage1_step2_infer_env_topic(item=env_item_raw, model=model)
    pretty_print({"environment_summary": env_item_with_env_description["environment_summary"], "environment_introduction": env_item_with_env_description["environment_introduction"]}, style='json')
    
    # Stage 2 - Step 1: Plan environment state space and constraints
    print("-" * 80)
    print("[Stage 2 - Step 1] Env State & Rules Planning")
    env_item_with_state_planing = stage2_step1_process_env_item(env_item = env_item_with_env_description, model=model)
    pretty_print({"state_space_definition": env_item_with_state_planing["state_space_definition"], "constraints_rules": env_item_with_state_planing["constraints_rules"]}, style='json')
    save_json(save_file_path, env_item_with_state_planing)
    
    # Stage 2 - Step 2: Convert state definition to Python class code
    print("-" * 80)
    print("[Stage 2 - Step 2] Convert State to Class Definition")
    env_item_with_state_code = stage2_step2_process_env_item(env_item = env_item_with_state_planing, model=model)
    pretty_print(env_item_with_state_code["class_definition"], style='python')
    save_json(save_file_path, env_item_with_state_code)
    
    # Stage 2 - Step 3: Plan environment operations/tools
    print("-" * 80)
    print("[Stage 2 - Step 3] Env Tool Operation Planning")
    env_item_with_operation_planing = stage2_step3_process_env_item(env_item = env_item_with_state_code, model=model)
    pretty_print(env_item_with_operation_planing["operation_list"], style='json')
    save_json(save_file_path, env_item_with_operation_planing)
    
    # Stage 2 - Step 4: Convert operations to function code
    print("-" * 80)
    print("[Stage 2 - Step 4] Convert Operation to Function Definition")
    env_item_with_operation_code = stage2_step4_process_env_item(env_item = env_item_with_operation_planing, model=model)
    save_json(save_file_path, env_item_with_operation_code)
    
    # Stage 2 - Step 5: Concatenate state and operations into complete environment class
    time.sleep(5)
    print("-" * 80)
    print("[Stage 2 - Step 5] Concat State & Operation to Complete Environment File")
    success, env_item_with_concat = stage2_step5_process_env_item(env_item = env_item_with_operation_code)
    print("[Full Environment Class Code Showing]")
    time.sleep(3)
    pretty_print(env_item_with_concat["env_class_code"], style='python')
    if not success:
        pretty_print("AST Check Failed, the process ends.", style='str', color='red')
    else:
        print("[AST Check Passed]")
    save_json(save_file_path, env_item_with_concat)
    time.sleep(5)
    
    # Stage 2 - Step 6: Analyze environment class code and extract tools interface
    print("-" * 80)
    print("[Stage 2 - Step 6] Analysis Env Class Code")
    env_item_final = stage2_step6_process_env_item(env_item = env_item_with_concat)
    print("[Tools Interface Showing]")
    time.sleep(3)
    pretty_print(env_item_final["tools"], style='json')
    save_json(save_file_path, env_item_final)
    
    
if __name__ == "__main__":
    # Demo configuration
    load_dotenv()
    env_item_raw = {
        "task": "I need to convert a PDF document named 'Annual_Report_2022.pdf' stored in the 'Reports' folder on 'GoogleDrive' into an Excel file for analysis. Can you assist with this?",
        "task_from": "ToolAce",
    }
    model = "gpt-4.1"
    save_file_path = "env_temp.json"
    main(env_item_raw, model, save_file_path)