"""
Reward calculation utilities for AceBench evaluation.
"""
from acebench_env.utils.eval_util import attribute_checker, calculate_process_accuracy


def calc_end_to_end_score(predicted_classes, reference_classes):
    """Calculate end-to-end score by comparing predicted and reference class states."""
    # Convert reference to list if not already
    if type(reference_classes) != list:
        reference_classes = [reference_classes]

    # Initialize evaluation result dictionary
    eval_info = {"valid": True, "error": [], "error_type": ""}
    
    is_valid = True
    checker_result = {}
    checker_result["valid"] = True

    # Check if number of classes matches
    if len(reference_classes) != len(predicted_classes):
        eval_info["valid"] = False
        eval_info["error_type"] = "wrong number of class"
        is_valid = False
    else:
        # Compare attribute values class by class
        for index in range(len(reference_classes)):
            possible_keys = set(reference_classes[index].keys())  
            matched_dict = None 
            
            # Find predicted class with matching key set
            for model_dict in predicted_classes:
                model_keys = set(model_dict.keys()) 
                if possible_keys == model_keys:  
                    matched_dict = model_dict
                    break  
            
            # If matched, use attribute checker to compare values
            if matched_dict:
                checker_result = attribute_checker(model_output=matched_dict, possible_answer=reference_classes[index])

            # If checker determines incorrect
            if checker_result["valid"] == False:
                eval_info["valid"] = False
                eval_info["error"].append(checker_result["error"])
                eval_info["error_type"] = checker_result["error_type"]
                is_valid = False

    return is_valid, eval_info

def calc_process_score(predicted_steps, expected_steps):
    """Calculate process score by comparing predicted and expected action steps."""
    process_eval_detail = {}

    if isinstance(expected_steps[0], list):
        # Multiple candidate steps, find best match
        best_process_score = -1
        for candidate_steps in expected_steps:
            rounded_score, raw_score, _ = calculate_process_accuracy(candidate_steps, predicted_steps)
            if rounded_score > best_process_score:
                best_process_score = rounded_score

        if best_process_score != 1.0:
            process_eval_detail = {
                "process_accuracy": best_process_score,
                "model_output": predicted_steps,
                "call_process": expected_steps
            }
        return best_process_score, process_eval_detail
    else:
        # Single expected steps
        rounded_score, raw_score, _ = calculate_process_accuracy(expected_steps, predicted_steps)
        if raw_score != 1.0:
            process_eval_detail = {
                "process_accuracy": rounded_score,
                "model_output": predicted_steps,
                "call_process": expected_steps
            }
        return raw_score, process_eval_detail

    
    
def calc_score(pred_env_states,pred_actions,target_env_states,target_actions):
    """Calculate overall score including end-to-end and process scores."""
    is_result_valid, check_end_to_end_info = calc_end_to_end_score(predicted_classes=pred_env_states, reference_classes=target_env_states)
    end_to_end_score = 1.0 if is_result_valid else 0.0
    # Note: Follow original logic - if end-to-end is correct, process score is 1.0; otherwise calculate from milestone
    if not is_result_valid:
        process_score, check_process_info = calc_process_score(predicted_steps=pred_actions, expected_steps=target_actions)
    else:
        process_score, check_process_info = 1.0, {}
    # process_score, check_process_info = calc_process_score(predicted_steps=pred_actions, expected_steps=target_actions)
    return end_to_end_score, process_score, check_end_to_end_info, check_process_info