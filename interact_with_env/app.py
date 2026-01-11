import gradio as gr
import json
import ast
import os
import re
import time
import html
from datetime import datetime
from dotenv import load_dotenv
from agent.task_solve_agent import TaskSolveAgent
from envscaler_env import EnvScalerConvRLEnv, EnvScalerNonConvRLEnv, EnvScalerConvSFTEnv, EnvScalerNonConvSFTEnv

load_dotenv()

HIGHLIGHT_CSS = """
.missing-field input, .missing-field textarea, .missing-field select, .missing-field .wrap {
    border-color: #ef4444 !important;
    box-shadow: none !important;  /* single red border, no double outline */
}
.missing-field label {
    color: #b91c1c !important;
}

.scroll-box {
    max-height: 320px;
    overflow: auto;
}

.fancy-table table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    border-radius: 10px;
    overflow: hidden;
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
}
.fancy-table th, .fancy-table td {
    padding: 10px 14px;
    border: 0;
}
.fancy-table tr:hover { background: rgba(59, 130, 246, 0.08); }
.fancy-table th {
    text-transform: uppercase;
    letter-spacing: 0.04em;
    font-size: 12px;
    font-weight: 600;
    background: transparent;
}

.info-overlay {
    position: fixed;
    inset: 0;
    background: rgba(15, 23, 42, 0.55);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 9999;
    padding: 24px;
    backdrop-filter: blur(4px);
}
.info-overlay__panel {
    background: #fff;
    color: #0f172a;
    max-width: 960px;
    width: min(960px, 100%);
    max-height: 90vh;
    border-radius: 16px;
    box-shadow: 0 20px 60px rgba(15, 23, 42, 0.25);
    overflow: hidden;
    display: flex;
    flex-direction: column;
}
.info-overlay__close {
    background: none;
    border: none;
    font-size: 18px;
    cursor: pointer;
    padding: 12px 14px;
    align-self: flex-end;
}
.info-overlay__body {
    padding: 0 24px 24px 24px;
    overflow: auto;
}
.info-overlay__section {
    margin-bottom: 16px;
}
.info-overlay__section h3 {
    margin-bottom: 8px;
}
.info-overlay__code {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 12px;
    font-family: ui-monospace, SFMono-Regular, SFMono, Menlo, Monaco, Consolas, monospace;
    font-size: 13px;
    white-space: pre-wrap;
    word-break: break-word;
    overflow: auto;
}
"""

ENV_DATA_PATH = "envscaler_env/data/191_env_metadata.json"
TASK_DATA_RL_PATH = "envscaler_env/data/rl_scenario_metadata.json"
TASK_DATA_SFT_PATH = "envscaler_env/data/sft_scenario_metadata.json"

with open(ENV_DATA_PATH, "r", encoding="utf-8") as f:
    env_raw = json.load(f)
    if isinstance(env_raw, dict):
        ENV_DATA = list(env_raw.values())
    else:
        ENV_DATA = env_raw

with open(TASK_DATA_RL_PATH, "r", encoding="utf-8") as f:
    TASK_DATA = json.load(f)

with open(TASK_DATA_SFT_PATH, "r", encoding="utf-8") as f:
    sft_tasks = json.load(f)
    for t in sft_tasks:
        t["is_sft"] = True
    TASK_DATA.extend(sft_tasks)

ENV_MAP = {env["env_id"]: env for env in ENV_DATA}

TASKS_BY_ENV = {}
ENVS_WITH_RL = set()
ENVS_WITH_SFT = set()

for task in TASK_DATA:
    env_id = task["env_id"]
    if env_id not in TASKS_BY_ENV:
        TASKS_BY_ENV[env_id] = []
    TASKS_BY_ENV[env_id].append(task)
    
    if task.get("is_sft"):
        ENVS_WITH_SFT.add(env_id)
    else:
        ENVS_WITH_RL.add(env_id)


def get_env_choices(task_type="RL"):
    """Get list of environment choices for dropdown, filtered by task type"""
    choices = []
    
    target_envs = ENVS_WITH_RL if task_type == "RL" else ENVS_WITH_SFT
    
    for env in ENV_DATA:
        env_id = env["env_id"]
        if env_id not in target_envs:
            continue
            
        summary = env.get("environment_summary", "")
        choices.append(f"{env_id}: {summary}")
    return choices


def parse_env_choice(choice_str):
    """Parse env_id from choice string"""
    if not choice_str:
        return None
    return choice_str.split(":")[0]


def sanitize_for_filename(value, fallback="value"):
    """Restrict user-provided strings to a safe subset for filenames"""
    if not value:
        return fallback
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))
    if not safe.strip("._-"):
        return fallback
    return safe[:120]


def rows_to_markdown(headers, rows):
    """Render a markdown table from headers and list of dict rows (or list of lists)."""
    if not rows:
        return ""
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        if isinstance(row, dict):
            vals = [str(row.get(h, "")) for h in headers]
        else:
            vals = [str(v) for v in row]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def build_section_overlay(env_choice, section):
    """Create HTML overlay content for a specific env section."""
    env_id = parse_env_choice(env_choice)
    if env_id is None:
        body = "<div class='info-overlay__section'><p>Please select an environment first.</p></div>"
        title = "Info"
    else:
        env = ENV_MAP[env_id]
        title_map = {
            "state": "State Schema",
            "tools": "Tools Schema",
            "program": "Program Implementation",
        }
        title = title_map.get(section, "Info")

        if section == "state":
            code = json.dumps(env.get("state_space_definition", []), ensure_ascii=False, indent=2)
        elif section == "tools":
            code = json.dumps(env.get("tools", []), ensure_ascii=False, indent=2)
        elif section == "program":
            code = env.get("env_class_code", "")
        else:
            code = "Unsupported section"

        code_html = html.escape(code)
        body = f"<div class='info-overlay__section'><div class='info-overlay__code'>{code_html}</div></div>"

    return f"""
    <div class="info-overlay" onclick="this.style.display='none'">
        <div class="info-overlay__panel" onclick="event.stopPropagation()">
            <div style="display:flex; align-items:center; justify-content: space-between; padding: 8px 12px 0 12px;">
                <h3 style="margin: 0;">{title}</h3>
                <button class="info-overlay__close" onclick="this.closest('.info-overlay').style.display='none'">✕</button>
            </div>
            <div class="info-overlay__body">
                {body}
            </div>
        </div>
    </div>
    """


def build_env_info_overlay(env_choice):
    """Fullscreen view for environment information."""
    env_id = parse_env_choice(env_choice)
    if env_id is None:
        body = "<div class='info-overlay__section'><p>Please select an environment first.</p></div>"
        title = "Environment Information"
    else:
        env = ENV_MAP.get(env_id, {})
        rules = "\n".join([f"- {r}" for r in env.get("constraints_rules", [])])
        info_text = (
            f"Env ID: {env.get('env_id', 'N/A')}\n"
            f"Name: {env.get('env_class_name', 'N/A')}\n\n"
            f"Description:\n{env.get('environment_introduction', 'N/A')}\n\n"
            f"Environment Rules:\n{rules or 'N/A'}"
        )
        info_html = html.escape(info_text)
        body = f"<div class='info-overlay__section'><div class='info-overlay__code'>{info_html}</div></div>"
        title = "Environment Information"

    return f"""
    <div class="info-overlay" onclick="this.style.display='none'">
        <div class="info-overlay__panel" onclick="event.stopPropagation()">
            <div style="display:flex; align-items:center; justify-content: space-between; padding: 8px 12px 0 12px;">
                <h3 style="margin: 0;">{title}</h3>
                <button class="info-overlay__close" onclick="this.closest('.info-overlay').style.display='none'">✕</button>
            </div>
            <div class="info-overlay__body">
                {body}
            </div>
        </div>
    </div>
    """


def build_task_overlay(env_choice, scenario_choice):
    """Fullscreen view for task description."""
    env_id = parse_env_choice(env_choice)
    if env_id is None or not scenario_choice:
        body = "<div class='info-overlay__section'><p>Select environment and scenario first.</p></div>"
    else:
        task_id = scenario_choice.split(": ", 1)[1].strip() if ": " in scenario_choice else scenario_choice
        selected_task = None
        for task in TASKS_BY_ENV.get(env_id, []):
            if task.get("task_id") == task_id:
                selected_task = task
                break
        if not selected_task:
            body = "<div class='info-overlay__section'><p>Task not found.</p></div>"
        else:
            task_desc = selected_task.get("task", "N/A")
            task_html = html.escape(task_desc)
            body = f"<div class='info-overlay__section'><div class='info-overlay__code'>{task_html}</div></div>"

    return f"""
    <div class="info-overlay" onclick="this.style.display='none'">
        <div class="info-overlay__panel" onclick="event.stopPropagation()">
            <div style="display:flex; align-items:center; justify-content: space-between; padding: 8px 12px 0 12px;">
                <h3 style="margin: 0;">Task Description</h3>
                <button class="info-overlay__close" onclick="this.closest('.info-overlay').style.display='none'">✕</button>
            </div>
            <div class="info-overlay__body">
                {body}
            </div>
        </div>
    </div>
    """


def build_init_overlay(env_choice, scenario_choice):
    """Fullscreen view for task Initial Configuration."""
    env_id = parse_env_choice(env_choice)
    if env_id is None or not scenario_choice:
        body = "<div class='info-overlay__section'><h3>Initial Configuration</h3><p>Select environment and scenario first.</p></div>"
    else:
        task_id = scenario_choice.split(": ", 1)[1].strip() if ": " in scenario_choice else scenario_choice
        selected_task = None
        for task in TASKS_BY_ENV.get(env_id, []):
            if task.get("task_id") == task_id:
                selected_task = task
                break
        if not selected_task:
            body = "<div class='info-overlay__section'><p>Task not found.</p></div>"
        else:
            init_cfg = html.escape(json.dumps(selected_task.get("init_config", {}), ensure_ascii=False, indent=2))
            body = f"<div class='info-overlay__section'><div class='info-overlay__code'>{init_cfg}</div></div>"

    return f"""
    <div class="info-overlay" onclick="this.style.display='none'">
        <div class="info-overlay__panel" onclick="event.stopPropagation()">
            <div style="display:flex; align-items:center; justify-content: space-between; padding: 8px 12px 0 12px;">
                <h3 style="margin: 0;">Initial Configuration</h3>
                <button class="info-overlay__close" onclick="this.closest('.info-overlay').style.display='none'">✕</button>
            </div>
            <div class="info-overlay__body">
                {body}
            </div>
        </div>
    </div>
    """


def build_reward_overlay(env_choice, scenario_choice):
    """Fullscreen view for Reward Functions."""
    env_id = parse_env_choice(env_choice)
    if env_id is None or not scenario_choice:
        body = "<div class='info-overlay__section'><h3>Reward Functions</h3><p>Select environment and scenario first.</p></div>"
    else:
        task_id = scenario_choice.split(": ", 1)[1].strip() if ": " in scenario_choice else scenario_choice
        selected_task = None
        for task in TASKS_BY_ENV.get(env_id, []):
            if task.get("task_id") == task_id:
                selected_task = task
                break
        if not selected_task:
            body = "<div class='info-overlay__section'><p>Task not found.</p></div>"
        elif selected_task.get("is_sft"):
            body = "<div class='info-overlay__section'><p>Due to cost considerations, validation functions were not generated for the SFT environment.</p></div>"
        else:
            checklist = selected_task.get("checklist_with_func", [])
            
            table_html = "<table class='fancy-table' style='width:100%; text-align:left; table-layout: fixed;'>"
            table_html += "<thead><tr><th style='width:60px'>#</th><th style='width:35%'>Checkpoint Description</th><th>Function</th></tr></thead><tbody>"
            for idx, item in enumerate(checklist):
                func_code = item.get('check_func', '')
                desc = html.escape(item.get('check_item', ''))
                # show full func code
                func_display = html.escape(func_code)
                table_html += f"<tr><td>{idx+1}</td><td>{desc}</td><td><pre style='margin:0; font-size:12px; font-family: ui-monospace, SFMono-Regular, SFMono, Menlo, Monaco, Consolas, monospace; white-space: pre-wrap; word-break: break-word;'>{func_display}</pre></td></tr>"
            table_html += "</tbody></table>"
            
            body = f"<div class='info-overlay__section'>{table_html}</div>"

    return f"""
    <div class="info-overlay" onclick="this.style.display='none'">
        <div class="info-overlay__panel" onclick="event.stopPropagation()">
            <div style="display:flex; align-items:center; justify-content: space-between; padding: 8px 12px 0 12px;">
                <h3 style="margin: 0;">Reward Functions</h3>
                <button class="info-overlay__close" onclick="this.closest('.info-overlay').style.display='none'">✕</button>
            </div>
            <div class="info-overlay__body">
                {body}
            </div>
        </div>
    </div>
    """




def on_env_select(env_choice, task_type="RL"):
    """When environment is selected, display env info and update scenario choices"""
    env_id = parse_env_choice(env_choice)
    if env_id is None:
        return (
            "Please choose an environment from the dropdown to view details.",
            "No environment selected.",
            "No environment selected.",
            "No environment selected.",
            gr.update(choices=[], value=None, label="2️⃣ Select Scenario (choose environment first)"),
            "",  # init_config_display
            "",  # task_display
            ""   # reward_display
        )

    env = ENV_MAP[env_id]
    
    env_info = f"""### Environment Information
**Env ID:** {env['env_id']}

**Name:** {env.get('env_class_name', 'N/A')}

**Description:**
{env.get('environment_introduction', 'N/A')}

**Environment Rules:**
"""
    for rule in env.get('constraints_rules', []):
        env_info += f"- {rule}\n"
    
    state_schema_json = json.dumps(env.get('state_space_definition', []), indent=2, ensure_ascii=False)
    state_schema = f"```json\n{state_schema_json}"
    
    tools_schema_json = json.dumps(env.get('tools', []), indent=2, ensure_ascii=False)
    tools_schema = f"```json\n{tools_schema_json}"

    program_impl = f"```python\n{env.get('env_class_code', 'N/A')}\n"
    
    all_tasks = TASKS_BY_ENV.get(env_id, [])
    tasks = []
    is_sft_mode = (task_type == "SFT")
    for t in all_tasks:
        if t.get("is_sft", False) == is_sft_mode:
            tasks.append(t)

    scenario_choices = []
    for task in tasks:
        task_id = task.get('task_id', 'N/A')
        scenario_choices.append(f"Task ID: {task_id}")

    if not scenario_choices:
        scenario_label = f"2️⃣ Select Scenario (no {task_type} tasks for this environment)"
        default_scenario = None
        init_config_text = ""
        task_text = ""
        reward_text = ""
    else:
        scenario_label = "2️⃣ Select Scenario"
        default_scenario = scenario_choices[0]
        
        init_config_text = f"### Initial Configuration\n\n```json\n{json.dumps(tasks[0].get('init_config', {}), indent=2, ensure_ascii=False)}\n```"
        task_text = f"### Task Description\n\n{tasks[0].get('task', 'N/A')}"
        
        if tasks[0].get("is_sft"):
             reward_text = "Due to cost considerations, validation functions were not generated for the SFT environment."
        else:
            checklist = tasks[0].get('checklist_with_func', [])
            headers = ["#", "Checkpoint Description"]
            rows = []
            for idx, item in enumerate(checklist):
                rows.append({"#": idx + 1, "Checkpoint Description": item.get('check_item', '')})
            reward_text = rows_to_markdown(headers, rows)

    return (
        env_info,
        state_schema,
        tools_schema,
        program_impl,
        gr.update(choices=scenario_choices, value=default_scenario, label=scenario_label),
        init_config_text,
        task_text,
        reward_text
    )


def on_scenario_select(env_choice, scenario_choice):
    """When scenario is selected, display init_config and task"""
    env_id = parse_env_choice(env_choice)
    if env_id is None or not scenario_choice:
        return "", "", ""
    
    task_id = scenario_choice.split(": ")[1].strip()
    tasks = TASKS_BY_ENV.get(env_id, [])
    selected_task = None
    for task in tasks:
        if task.get('task_id') == task_id:
            selected_task = task
            break
    
    if not selected_task:
        return "Task not found", "", ""
    
    init_config_display = f"```json\n{json.dumps(selected_task.get('init_config', {}), indent=2, ensure_ascii=False)}\n"
    
    task_display = f"### Task Description\n\n{selected_task.get('task', 'N/A')}"
    
    if selected_task.get("is_sft"):
        reward_text = "Due to cost considerations, validation functions were not generated for the SFT environment."
    else:
        checklist = selected_task.get('checklist_with_func', [])
        headers = ["#", "Checkpoint Description"]
        rows = []
        for idx, item in enumerate(checklist):
            rows.append({"#": idx + 1, "Checkpoint Description": item.get('check_item', '')})
        reward_text = rows_to_markdown(headers, rows)
    
    return init_config_display, task_display, reward_text


def clear_all():
    """Clear all selections and reset to initial state"""
    return (
        gr.update(value=None, elem_classes=[]),  # env_dropdown
        "Please choose an environment from the dropdown to view details.",  # env_info_display
        "No environment selected.",  # state_schema_display
        "No environment selected.",  # tools_schema_display
        "No environment selected.",  # program_impl_display
        gr.update(choices=[], value=None, label="2️⃣ Select Scenario (choose environment first)", elem_classes=[]),  # scenario_dropdown
        "",  # init_config_display
        "",  # task_display
        "",  # reward_display
        [],  # trajectory_output
        "Waiting for execution...",  # checkpoint_output
        "Waiting for execution...",  # reward_output
        gr.update(value=None, visible=True, interactive=False),  # download_file - keep visible but disabled
        gr.update(value="", visible=False),  # error_display
        gr.update(value="gpt-4.1", elem_classes=[]),  # agent_model
        gr.update(value="gpt-4.1", elem_classes=[]),  # user_model
        gr.update(value="", elem_classes=[]),  # shared_api_key
        gr.update(value="https://api.openai.com/v1", elem_classes=[]),  # shared_base_url
        False,  # allow_run_state
        "",  # progress_output
        gr.update(value="", visible=False)  # info_overlay
    )


def format_action_message(step_num, action):
    """Format action as a chat message content"""
    if not action:
        return ""
    
    if isinstance(action, dict):
        tool_name = action.get('name', 'unknown')
        args = action.get('arguments', {})

        if isinstance(args, dict):
            if tool_name == "chat_with_user" and "content" in args:
                call_str = str(args.get("content", ""))
            else:
                arg_parts = []
                for k, v in args.items():
                    v_str = json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else v
                    arg_parts.append(f'{k}={v_str}')
                arg_str = ", ".join(arg_parts)
                call_str = f"{tool_name}({arg_str})"
        else:
            arg_str = str(args)
            call_str = f"{tool_name}({arg_str})"
    else:
        call_str = str(action)

    return f'Agent Action\n\n<pre style="white-space: pre-wrap; word-break: break-word; margin: 0; background-color: transparent;">{call_str}</pre>'


def format_observation_message(step_num, observation):
    """Format observation as a chat message content with smart parsing"""
    if not observation:
        return ""
    
    obs_type = 'user'
    if isinstance(observation, dict):
        obs_type = observation.get('type', 'user')
        obs_content = observation.get('content', observation)
    else:
        obs_content = observation

    if isinstance(obs_content, str):
        content_str = obs_content.strip()
        if content_str.startswith("{") or content_str.startswith("["):
            try:
                parsed = json.loads(content_str)
                content_str = json.dumps(parsed, ensure_ascii=False, indent=2)
            except (json.JSONDecodeError, ValueError):
                pass
    else:
        content_str = json.dumps(obs_content, ensure_ascii=False, indent=2)

    label = "Tool" if obs_type == "tool" else "User" if obs_type == "user" else f"{obs_type}"

    return f'{label}\n\n<pre style="white-space: pre-wrap; word-break: break-word; margin: 0; background-color: transparent;">{content_str}</pre>'

def create_progress_html(pct, desc):
    pct_val = pct * 100
    return f"""
    <div style="margin-bottom: 10px;">
        <div style="width: 100%; background-color: #e5e7eb; border-radius: 9999px; height: 8px;">
            <div style="background-color: #F97316; height: 8px; border-radius: 9999px; width: {pct_val}%;"></div>
        </div>
    <div style="margin-top: 4px; font-size: 12px; color: #6b7280; text-align: right;">{desc}</div>
    </div>
    """


def open_overlay_section(env_choice, section):
    overlay_html = build_section_overlay(env_choice, section) + f"<!-- {time.time()} -->"
    return gr.update(value=overlay_html, visible=True)

def open_overlay_env_info(env_choice):
    overlay_html = build_env_info_overlay(env_choice) + f"<!-- {time.time()} -->"
    return gr.update(value=overlay_html, visible=True)

def open_overlay_task(env_choice, scenario_choice):
    overlay_html = build_task_overlay(env_choice, scenario_choice) + f"<!-- {time.time()} -->"
    return gr.update(value=overlay_html, visible=True)

def open_overlay_state(env_choice):
    return open_overlay_section(env_choice, "state")

def open_overlay_tools(env_choice):
    return open_overlay_section(env_choice, "tools")

def open_overlay_program(env_choice):
    return open_overlay_section(env_choice, "program")

def open_overlay_init(env_choice, scenario_choice):
    overlay_html = build_init_overlay(env_choice, scenario_choice) + f"<!-- {time.time()} -->"
    return gr.update(value=overlay_html, visible=True)


def open_overlay_reward(env_choice, scenario_choice):
    overlay_html = build_reward_overlay(env_choice, scenario_choice) + f"<!-- {time.time()} -->"
    return gr.update(value=overlay_html, visible=True)


def run_trajectory(env_choice, scenario_choice, env_mode, agent_model, 
                   shared_api_key, shared_base_url, user_model, 
                   temperature, max_steps, allow_run):
    """Run the agent on the selected environment and scenario with streaming updates"""
    download_reset = gr.update(value=None, interactive=False, visible=True)

    infer_mode = "fc"
    
    env_id = parse_env_choice(env_choice)
    if not allow_run or env_id is None or not scenario_choice:
        warn = "### ⚠️ Missing required inputs\n\nPlease fill the required fields highlighted in red."
        yield [], "", warn, download_reset, ""
        return
    
    task_id = scenario_choice.split(": ")[1].strip()
    tasks = TASKS_BY_ENV.get(env_id, [])
    task_index = None
    for idx, task in enumerate(tasks):
        if task.get('task_id') == task_id:
            for global_idx, global_task in enumerate(TASK_DATA):
                if global_task.get('task_id') == task_id and global_task.get('env_id') == env_id:
                    task_index = global_idx
                    break
            break

    if task_index is None:
        yield [], "", "### ⚠️ Task not found", download_reset, ""
        return
    
    history = []
    checkpoint_display = ""
    reward_display = ""
    progress_html = create_progress_html(0, "Starting...")
    
    yield history, checkpoint_display, reward_display, download_reset, progress_html
    
    try:
        progress_html = create_progress_html(0.1, "Initializing environment...")
        yield history, checkpoint_display, reward_display, download_reset, progress_html
        
        env_config = {"mode": "train"}
        
        task_item = TASK_DATA[task_index]
        is_sft = task_item.get("is_sft", False)
        task_file_path = TASK_DATA_SFT_PATH if is_sft else TASK_DATA_RL_PATH

        if env_mode == "conversation":
            if is_sft:
                env = EnvScalerConvSFTEnv(
                    mode=env_config["mode"],
                    user_model=user_model,
                    provider="openai",
                    env_items_path=ENV_DATA_PATH,
                    task_items_path=task_file_path,
                    api_key=shared_api_key,
                    base_url=shared_base_url,
                )
                agent_env_name = "envscaler_conversation_sft"
            else:
                env = EnvScalerConvRLEnv(
                    mode=env_config["mode"],
                    user_model=user_model,
                    provider="openai",
                    env_items_path=ENV_DATA_PATH,
                    task_items_path=task_file_path,
                    api_key=shared_api_key,
                    base_url=shared_base_url,
                )
                agent_env_name = "envscaler_conversation_rl"
        else:
            if is_sft:
                env = EnvScalerNonConvSFTEnv(
                    mode=env_config["mode"],
                    env_items_path=ENV_DATA_PATH,
                    task_items_path=task_file_path
                )
                agent_env_name = "envscaler_non_conversation_sft"
            else:
                env = EnvScalerNonConvRLEnv(
                    mode=env_config["mode"],
                    env_items_path=ENV_DATA_PATH,
                    task_items_path=task_file_path
                )
                agent_env_name = "envscaler_non_conversation_rl"
        
        progress_html = create_progress_html(0.2, "Creating agent...")
        yield history, checkpoint_display, reward_display, download_reset, progress_html
        
        agent = TaskSolveAgent(
            env_name=agent_env_name,
            env=env,
            model=agent_model,
            provider="openai",
            infer_mode=infer_mode,
            temperature=float(temperature),
            max_steps=int(max_steps),
            enable_thinking=False,
            api_key=shared_api_key,
            base_url=shared_base_url,
        )
        
        agent.reset(task_index=task_index)
        
        initial_obs_msg = format_observation_message(0, agent.current_observation)
        history.append({"role": "user", "content": initial_obs_msg})
        yield history, checkpoint_display, reward_display, download_reset, progress_html
        
        step_count = 0
        max_steps_int = int(max_steps)
        
        while (not agent.terminated) and (not agent.truncated) and (step_count < max_steps_int):
            progress_html = create_progress_html((0.3 + 0.6 * step_count / max_steps_int), f"Step {step_count + 1}/{max_steps_int}...")
            yield history, checkpoint_display, reward_display, download_reset, progress_html
            
            observation, reward, terminated, truncated, info, action = agent.step()
            step_count += 1
            
            action_msg = format_action_message(step_count, action)
            history.append({"role": "assistant", "content": action_msg})
            yield history, checkpoint_display, reward_display, download_reset, progress_html
            
            obs_msg = format_observation_message(step_count, observation)
            history.append({"role": "user", "content": obs_msg})
            yield history, checkpoint_display, reward_display, download_reset, progress_html
        
        if step_count >= max_steps_int and not agent.terminated and not agent.truncated:
            agent.truncated = True
        
        progress_html = create_progress_html(0.95, "Calculating rewards...")
        yield history, checkpoint_display, reward_display, download_reset, progress_html
        
        checkpoint_rows = []
        task_item = TASK_DATA[task_index]
        is_sft = task_item.get("is_sft", False)
        
        reward_rows = []
        reward_rows.append({"Metric": "Total Steps", "Value": agent.step_count})
        reward_rows.append({"Metric": "Terminated", "Value": "Yes" if agent.terminated else "No"})
        reward_rows.append({"Metric": "Truncated", "Value": "Yes" if agent.truncated else "No"})

        if is_sft:
            reward_rows.append({"Metric": "Evaluation", "Value": "Skipped"})
            reward_rows.append({"Metric": "Final Reward", "Value": "N/A"})
        else:
            checklist = task_item.get('checklist_with_func', [])
            
            init_state = None
            final_state = None
            
            if hasattr(env, 'init_state') and hasattr(env, 'pred_final_state'):
                init_state = env.init_state
                final_state = env.pred_final_state
            elif hasattr(env, 'trajectory') and len(env.trajectory) > 0:
                if 'state_snapshot' in env.trajectory[0]:
                    init_state = env.trajectory[0]['state_snapshot']
                if 'state_snapshot' in env.trajectory[-1]:
                    final_state = env.trajectory[-1]['state_snapshot']
            
            from envscaler_env.utils.env_util import run_check_function
            
            total_score = 0.0
            passed = 0
            
            for idx, check_item in enumerate(checklist):
                check_desc = check_item.get('check_item', f'Checkpoint {idx+1}')
                check_func = check_item.get('check_func', '')
                
                if check_func and init_state is not None and final_state is not None:
                    success, score, error = run_check_function(check_func, init_state, final_state)
                    if success and score is not None:
                        status = "Pass" if score > 0.5 else "Fail"
                        checkpoint_rows.append({
                            "#": idx + 1,
                            "Checkpoint Description": check_desc,
                            "Status": status,
                            "Score": round(score, 2)
                        })
                        total_score += score
                        if score > 0.5:
                            passed += 1
                    else:
                        checkpoint_rows.append({
                            "#": idx + 1,
                            "Checkpoint Description": check_desc,
                            "Status": "Error",
                            "Score": "-"
                        })
                else:
                    checkpoint_rows.append({
                        "#": idx + 1,
                        "Checkpoint Description": check_desc,
                        "Status": "Not evaluated",
                        "Score": "-"
                    })
            
            if len(checklist) > 0 and total_score > 0:
                avg_score = total_score / len(checklist)
                reward_rows.append({"Metric": "Checkpoints Passed", "Value": f"{passed}/{len(checklist)}"})
                reward_rows.append({"Metric": "Average Checkpoint Score", "Value": f"{avg_score:.4f}"})
            
            reward_rows.append({"Metric": "Final Reward", "Value": f"{agent.total_reward:.4f}"})
        
        progress_html = create_progress_html(1.0, "Complete!")
        
        result_data = {
            "task_info": agent.task_info,
            "tools": agent.tools,
            "user_tools": agent.user_tools,
            "messages": agent.messages,
            "user_messages": agent.user_messages,
            "trajectory": agent.trajectory,
            "total_reward": agent.total_reward,
            "terminated": agent.terminated,
            "truncated": agent.truncated,
            "final_observation": agent.current_observation,
            "final_info": agent.current_info,
            "steps": agent.step_count,
        }
        
        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        safe_agent_model = sanitize_for_filename(agent_model, fallback="model")
        safe_user_model = sanitize_for_filename(user_model, fallback="user") if env_mode == "conversation" else None
        if env_mode == "conversation":
            filename = f"{agent_env_name}-{safe_agent_model}-{infer_mode}_{safe_user_model}_{timestamp}.json"
        else:
            filename = f"{agent_env_name}-{safe_agent_model}-{infer_mode}_{timestamp}.json"
        
        save_dir = "trajectory_results"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.abspath(os.path.join(save_dir, filename))
        
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(result_data, f, indent=4, ensure_ascii=False)
        
        if is_sft:
             checkpoint_display = "Due to cost considerations, validation functions were not generated for the SFT environment."
        else:
            checkpoint_display = rows_to_markdown(
                ["#", "Checkpoint Description", "Status", "Score"], checkpoint_rows
            )
        
        reward_display = rows_to_markdown(
            ["Metric", "Value"], reward_rows
        )
        yield history, checkpoint_display, reward_display, gr.update(
            value=save_path,
            visible=True,
            interactive=True
        ), progress_html
        
    except Exception:
        import traceback
        error_msg = f"### ❌ Error\n\n```\n{traceback.format_exc()}\n```"
        history.append({"role": "assistant", "content": error_msg})
        yield history, "### ❌ Error\nExecution failed", "### ❌ Error\nExecution failed", download_reset, ""


with gr.Blocks(title=" Play with EnvScaler Environment") as demo:
    with gr.Row():
        with gr.Column(scale=0, min_width=100):
            gr.Image(value="../Figs/envscaler_logo.png", show_label=False, container=False, interactive=False, buttons=list())
        with gr.Column(scale=20):
            gr.Markdown("# Play with EnvScaler Environment\nGithub: https://github.com/RUC-NLPIR/EnvScaler")
    
    gr.HTML(f"<style>{HIGHLIGHT_CSS}</style>")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## Environment & Scenario")
            
            task_type_radio = gr.Radio(
                choices=["RL", "SFT"],
                value="RL",
                label="1️⃣ Select Environment",   
            )

            env_dropdown = gr.Dropdown(
                choices=get_env_choices("RL"),
                show_label=False,
                interactive=True,
                allow_custom_value=False,
                filterable=False,
                value=None
            )
            
            with gr.Accordion("Environment Information", open=False):
                with gr.Row():
                    with gr.Column(scale=1, min_width=60):
                        env_info_popout = gr.Button("Fullscreen", size="sm", variant="secondary")
                    with gr.Column(scale=8):
                        env_info_display = gr.Markdown("Select an environment to view details", elem_classes=["scroll-box"])
            
            with gr.Accordion("State Schema", open=False):
                with gr.Row():
                    with gr.Column(scale=1, min_width=60):
                        state_schema_popout = gr.Button("Fullscreen", size="sm", variant="secondary")
                    with gr.Column(scale=8):
                        state_schema_display = gr.Markdown("Select an environment to view state schema", elem_classes=["scroll-box"])
                                
            with gr.Accordion("Tools Schema", open=False):
                with gr.Row():
                    with gr.Column(scale=1, min_width=60):
                        tools_schema_popout = gr.Button("Fullscreen", size="sm", variant="secondary")
                    with gr.Column(scale=8):
                        tools_schema_display = gr.Markdown("Select an environment to view tools", elem_classes=["scroll-box"])
                    
            
            with gr.Accordion("Program Implementation", open=False):
                with gr.Row():
                    with gr.Column(scale=1, min_width=60):
                        program_impl_popout = gr.Button("Fullscreen", size="sm", variant="secondary")
                    with gr.Column(scale=8):
                        program_impl_display = gr.Markdown("Select an environment to view implementation", elem_classes=["scroll-box"])
            
            gr.Markdown("---")
            scenario_dropdown = gr.Dropdown(
                label="2️⃣ Select Scenario",
                interactive=True,
                allow_custom_value=False,
                filterable=False
            )

            with gr.Accordion("Task Description", open=False):
                with gr.Row():
                    with gr.Column(scale=1, min_width=60):
                        task_popout = gr.Button("Fullscreen", size="sm", variant="secondary")
                    with gr.Column(scale=8):
                        task_display = gr.Markdown("Select a scenario to view task", elem_classes=["scroll-box"])
        
            with gr.Accordion("Initial Configuration", open=False):
                with gr.Row():
                    with gr.Column(scale=1, min_width=60):
                        init_config_popout = gr.Button("Fullscreen", size="sm", variant="secondary")
                    with gr.Column(scale=8):
                        init_config_display = gr.Markdown("Select a scenario to view configuration", elem_classes=["scroll-box"])
            
            with gr.Accordion("Reward Functions", open=False):
                with gr.Row():
                    with gr.Column(scale=1, min_width=60):
                        reward_popout = gr.Button("Fullscreen", size="sm", variant="secondary")
                    with gr.Column(scale=8):
                        reward_display = gr.Markdown("Select a scenario to view reward functions", elem_classes=["scroll-box"])
                                
        with gr.Column(scale=2):
            gr.Markdown("## Execution")
            
            with gr.Row():
                run_btn = gr.Button("▶️ Start Trajectory", variant="primary", size="lg")
                clear_btn = gr.Button("🔄 Clear", variant="secondary", size="lg")
            error_display = gr.Markdown("", visible=False)

            trajectory_output = gr.Chatbot(
                label="Trajectory",
                height=600,
                show_label=False
            )

            download_file = gr.DownloadButton(
                label="Trajectory JSON Download",
                variant="primary",
                visible=True,
                interactive=False
            )

            gr.Markdown("## Evaluation Checkpoints")
            checkpoint_output = gr.Markdown("Waiting for execution...", elem_classes=["fancy-table"])
                
        with gr.Column(scale=1):
            gr.Markdown("## Config")
            
            env_mode = gr.Radio(
                choices=["conversation", "non_conversation"],
                value="conversation",
                label="Environment Mode",
            )

            shared_api_key = gr.Textbox(
                label="API Key",
                type="password",
            )
            shared_base_url = gr.Textbox(
                label="Base URL",
                value="https://api.openai.com/v1",
            )
            
            infer_mode = "fc"
            
            max_steps = gr.Slider(
                minimum=1,
                maximum=100,
                value=30,
                step=1,
                label="Max Steps"
            )
            
            agent_model = gr.Textbox(
                label="Agent Model",
                value="gpt-4.1",
            )

            user_model = gr.Textbox(
                label="User Model",
                value="gpt-4.1",
                visible=True
            )

            temperature = gr.Slider(
                minimum=0.0,
                maximum=2.0,
                value=0.5,
                step=0.1,
                label="Temperature"
            )

            
            gr.Markdown("## Evaluation Results")
            
            progress_output = gr.HTML(visible=True)
            reward_output = gr.Markdown("Waiting for execution...", elem_classes=["fancy-table"])

    info_overlay = gr.HTML("", visible=False)

            

    
    def toggle_user_model(mode):
        return gr.update(visible=(mode == "conversation"))
        
    def on_task_type_change(task_type):
        new_choices = get_env_choices(task_type)
        return (
            gr.update(choices=new_choices, value=None),
            "Please choose an environment from the dropdown to view details.",
            "No environment selected.",
            "No environment selected.",
            "No environment selected.",
            gr.update(choices=[], value=None, label=f"2️⃣ Select Scenario (choose environment first)"),
            "",
            "",
            ""
        )
    
    task_type_radio.change(
        fn=on_task_type_change,
        inputs=[task_type_radio],
        outputs=[env_dropdown, env_info_display, state_schema_display, tools_schema_display, 
                program_impl_display, scenario_dropdown,
                init_config_display, task_display, reward_display]
    )
    
    env_dropdown.change(
        fn=on_env_select,
        inputs=[env_dropdown, task_type_radio],
        outputs=[env_info_display, state_schema_display, tools_schema_display, 
                program_impl_display, scenario_dropdown,
                init_config_display, task_display, reward_display]
    )

    scenario_dropdown.change(
        fn=on_scenario_select,
        inputs=[env_dropdown, scenario_dropdown],
        outputs=[init_config_display, task_display, reward_display]
    )
    state_schema_popout.click(
        fn=open_overlay_state,
        inputs=[env_dropdown],
        outputs=[info_overlay],
        queue=False
    )
    env_info_popout.click(
        fn=open_overlay_env_info,
        inputs=[env_dropdown],
        outputs=[info_overlay],
        queue=False
    )
    tools_schema_popout.click(
        fn=open_overlay_tools,
        inputs=[env_dropdown],
        outputs=[info_overlay],
        queue=False
    )
    program_impl_popout.click(
        fn=open_overlay_program,
        inputs=[env_dropdown],
        outputs=[info_overlay],
        queue=False
    )
    init_config_popout.click(
        fn=open_overlay_init,
        inputs=[env_dropdown, scenario_dropdown],
        outputs=[info_overlay],
        queue=False
    )
    task_popout.click(
        fn=open_overlay_task,
        inputs=[env_dropdown, scenario_dropdown],
        outputs=[info_overlay],
        queue=False
    )
    reward_popout.click(
        fn=open_overlay_reward,
        inputs=[env_dropdown, scenario_dropdown],
        outputs=[info_overlay],
        queue=False
    )
    
    env_mode.change(
        fn=toggle_user_model,
        inputs=[env_mode],
        outputs=[user_model]
    )
    allow_run_state = gr.State(False)

    def validate_inputs(env_choice, scenario_choice, env_mode, agent_model,
                        shared_api_key, shared_base_url, user_model):
        missing_env = parse_env_choice(env_choice) is None
        missing_scenario = not scenario_choice
        missing_agent_model = not agent_model
        missing_user_model = env_mode == "conversation" and not user_model
        missing_api_key = not shared_api_key
        missing_base_url = not shared_base_url

        messages = []
        if missing_env:
            messages.append("Select environment")
        if missing_scenario:
            messages.append("Select scenario")
        if missing_agent_model:
            messages.append("Enter agent model")
        if env_mode == "conversation" and missing_user_model:
            messages.append("Enter user model")
        if missing_api_key:
            messages.append("Enter shared API Key")
        if missing_base_url:
            messages.append("Enter Base URL")

        error_msg = "、".join(messages)
        allow_run = len(messages) == 0

        return (
            gr.update(elem_classes=["missing-field"] if missing_env else []),
            gr.update(elem_classes=["missing-field"] if missing_scenario else []),
            gr.update(elem_classes=["missing-field"] if missing_agent_model else []),
            gr.update(elem_classes=["missing-field"] if missing_user_model else []),
            gr.update(elem_classes=["missing-field"] if missing_api_key else []),
            gr.update(elem_classes=["missing-field"] if missing_base_url else []),
            gr.update(value=f"⚠️ {error_msg}" if error_msg else "", visible=not allow_run),
            allow_run,
        )

    def disable_buttons():
        return gr.update(interactive=False), gr.update(interactive=False)

    def enable_buttons():
        return gr.update(interactive=True), gr.update(interactive=True)

    run_validation = run_btn.click(
        fn=validate_inputs,
        inputs=[env_dropdown, scenario_dropdown, env_mode, agent_model,
               shared_api_key, shared_base_url, user_model],
        outputs=[env_dropdown, scenario_dropdown, agent_model, user_model,
                 shared_api_key, shared_base_url, error_display, allow_run_state],
        queue=False
    )
    
    run_validation.then(
        fn=disable_buttons,
        inputs=[],
        outputs=[run_btn, clear_btn],
        queue=False
    ).then(
        fn=run_trajectory,
        inputs=[env_dropdown, scenario_dropdown, env_mode, agent_model,
               shared_api_key, shared_base_url, user_model,
               temperature, max_steps, allow_run_state],
        outputs=[trajectory_output, checkpoint_output, reward_output, download_file, progress_output],
        show_progress="hidden"
    ).then(
        fn=enable_buttons,
        inputs=[],
        outputs=[run_btn, clear_btn],
        queue=False
    )
    
    clear_btn.click(
        fn=clear_all,
        inputs=[],
        outputs=[env_dropdown, env_info_display, state_schema_display, tools_schema_display,
                program_impl_display, scenario_dropdown, init_config_display, task_display,
                reward_display,
                trajectory_output, checkpoint_output, reward_output, download_file,
                error_display, agent_model, user_model,
                shared_api_key, shared_base_url, allow_run_state, progress_output, info_overlay]
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
