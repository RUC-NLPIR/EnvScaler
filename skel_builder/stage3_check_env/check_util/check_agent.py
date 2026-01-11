"""
Check agent for validating environment class method calls using LLM analysis.
Analyzes method behavior, state changes, and return values to detect issues.
"""
from utils.call_llm import llm_inference

# Prompt template for LLM-based method call validation
input_template =\
"""You are an experienced "Interactive Simulation Environment Testing Specialist", with extensive background in validating simulated systems and environments (such as game simulations, business system sandboxes, etc.).

Your task is to fully analyze whether a given method call in an environment class meets the expected behavior, based on the provided environment class structure, method source code, call parameters, the relevant internal state before and after the call, the differences between these states, and the method's returned observation.  

You should pay special attention to:  
- Whether the method causes the relevant internal state to change correctly before and after the call  
- Whether the code logic, conditional checks, state changes, and return value are consistent  
- Whether there are any unexpected exceptions or logical errors  

---

Below are the specific details of the environment class and method:

[Environment Introduction]  
{env_introduction}

[Environment Rules/Constraints]  
{env_rules}

[Environment Class Definition (from file start to '__init__' method)]  
{env_class_def}

[Name of the Method Called]  
{func_name}

[Source Code of the Method Called]  
{func_source}

[Method Call Parameters]  
{func_params}

[Relevant State Before Call]  
{state_before_call}

[Method's Return Value]  
{func_return}

[Relevant State After Call]  
{state_after_call}

[Difference Between States Before and After Call]  
{state_diff}

---

Strictly output your answer in the following format:
[Analysis]  
Your Step-by-step analysis.

[Result]  
Answer only one of the three words: 'Pass', 'Warning', or 'Fail', without any other words.  
- 'Pass' — The method fully meets expectations, implementation is correct, and no issues are found.  
- 'Warning' — The method works and meets functional expectations, but there are potential issues such as missing parameter validation, lack of boundary checks, absence of fallback mechanisms, or minor style/robustness problems.  
- 'Fail' — The method does not meet functional expectations, contains major logic errors, incorrect state changes, unhandled exceptions, or behaviors that violate environment rules.

[Error Reason]  
If the answer to 'Result' is 'Fail' or 'Warning', provide the reason you believe the error occurred and accordiing solutions.  
If the answer is 'Pass',  just output 'No error'."""


class CheckAgent:
    """Agent for checking environment method calls using LLM validation."""
    
    def __init__(self, model, temperature, env_item):
        """Initialize check agent with LLM model and environment item."""
        self.model = model
        self.temperature = temperature
        self.env_info = {
            "env_introduction": env_item["environment_introduction"],
            "env_rules": "\n".join([f"- {rule}" for rule in env_item["constraints_rules"]]),
            "env_class_def": env_item["env_class_def"],
        }
        # Map function names to their source code
        self.func_source_map = {func_name:func_detail["source_code"] for func_name,func_detail in env_item["env_func_details"].items()}
        self.input_template = input_template
        
    def format_input(self, func_name, state_before_call, func_params, func_return, state_after_call, state_diff):
        """Format method call information into LLM prompt."""
        input_content = self.input_template.format(
            env_introduction=self.env_info["env_introduction"],
            env_rules=self.env_info["env_rules"],
            env_class_def=self.env_info["env_class_def"],
            func_name=func_name,
            func_source=self.func_source_map[func_name],
            state_before_call=state_before_call,
            func_params=func_params,
            func_return=func_return,
            state_after_call=state_after_call,
            state_diff=state_diff,
        )
        return input_content
    
    def parse_response(self, response):
        """Parse LLM response into structured format (analysis, result, error_reason)."""
        try:
            # Extract analysis section
            analysis = response.split("[Analysis]")[1].split("[Meets Expectation]")[0].strip()
            # Extract result (Pass/Warning/Fail)
            result = response.split("[Result]")[1].split("[Error Reason]")[0].strip()
            valid_results = ["pass", "warning", "fail"]
            assert result.lower() in valid_results
            # Extract error reason
            error_reason = response.split("[Error Reason]")[1].strip()
            parsed_content = {
                "analysis": analysis,
                "result": result,
                "error_reason": error_reason
            }
            return True, parsed_content
        except Exception as e:
            print("parse response error:", e, "response:", response)
            return False, {"analysis": f"parse response error {e}", "result": "", "error_reason": ""}
        
        
        
    def check_func_call(self, func_name, state_before_call, func_params, func_return, state_after_call, state_diff):
        """Check method call behavior using LLM, retry up to max_check_try times if parsing fails."""
        input_content = self.format_input(func_name, state_before_call, func_params, func_return, state_after_call, state_diff)
        input_message = [{"role": "user", "content": input_content}]
        cur_check_try = 0
        max_check_try = 5
        # Retry if response parsing fails
        while cur_check_try < max_check_try:
            response = llm_inference(provider="openai", model=self.model, messages=input_message, temperature=self.temperature)
            parsed_success, parsed_content = self.parse_response(response)
            if parsed_success:
                break
        return parsed_content
    
    

