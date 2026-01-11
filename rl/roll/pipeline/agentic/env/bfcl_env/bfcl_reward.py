import ast
from .xml_parser import XMLParser


class BfclRewrard:
    """Reward calculation for BFCL environment evaluation."""

    def __init__(self,
                 parser: XMLParser = XMLParser(fields=["reasoning", "tool"]),
                 env_parser: XMLParser = XMLParser(fields=["tool_result"])):
        self.parser = parser
        self.env_parser = env_parser
        self.reward_funcs = [
            self.unified_reward_func,
        ]

    @staticmethod
    def _parse_function_call(func_call_str: str):
        """
        Parse a function call string into structured dictionary.
        Example: 'foo(a=1, b="x")' -> {'name': 'foo', 'args': {'a': 1, 'b': 'x'}}
        """
        try:
            # Parse string into AST safely
            tree = ast.parse(func_call_str, mode='eval')
            # Verify it's a function call
            if not isinstance(tree.body, ast.Call):
                raise ValueError("Input is not a valid function call.")
            # Extract function name
            func_name = tree.body.func.id if isinstance(tree.body.func, ast.Name) else None
            if not func_name:
                raise ValueError("Could not determine function name.")
            # Parse all arguments
            args_dict = {}
            # Parse keyword arguments
            for kw in tree.body.keywords:
                args_dict[kw.arg] = ast.literal_eval(kw.value)  # Convert AST to actual Python value
            # Parse positional arguments
            for i, arg in enumerate(tree.body.args):
                args_dict[f"arg{i+1}"] = ast.literal_eval(arg)
            # Return structured object
            json_obj = {
                "name": func_name,
                "args": args_dict
            }

            return json_obj

        except Exception:
            raise Exception(f"Error in Parsing Ground Truth Function Call is Not Expected!!")

    @staticmethod
    def _is_subsequence_unordered(list1, list2) -> tuple[bool, list]:
        """
        Check if all elements of list1 appear in list2 (unordered, respects duplicates).
        """
        # Empty lists cannot be contained
        if list1 == [] or list2 == []:
            return False, []
        # Copy list2 to avoid modifying original
        list2_copy = list2[:]
        # Track missing elements by trying to remove each item
        missing_elements = []
        for item in list1:
            try:
                # Remove one occurrence to handle duplicates correctly
                list2_copy.remove(item)
            except ValueError:
                # Item not found, add to missing list
                missing_elements.append(item)
        
        # All items found if no missing elements
        is_subsequence = len(missing_elements) == 0
        return is_subsequence, missing_elements


    @staticmethod
    def compare_instances(model_obect, ground_truth_object):
        """
        Compare all public attributes of two Python objects of the same type.
        
        Args:
            model_obect: Model output object (e.g., API instance after execution)
            ground_truth_object: Ground truth object (e.g., API instance from GT answer)
        Returns:
            Tuple of (valid, differences) where valid is bool and differences is dict
        """
        # Ensure both objects are of the same type
        assert type(model_obect) == type(
            ground_truth_object
        ), "Objects are not of the same type."

        differences = {}
        valid = True

        # Iterate through all attributes of ground truth object
        for attr_name in vars(ground_truth_object):
            # Skip private attributes (starting with underscore)
            if attr_name.startswith("_"):
                continue
            # Get attribute values from both objects
            model_attr = getattr(model_obect, attr_name)
            ground_truth_attr = getattr(ground_truth_object, attr_name)
            # Record differences if attributes don't match
            if model_attr != ground_truth_attr:
                valid = False
                differences[attr_name] = {"model": model_attr, "ground_truth": ground_truth_attr}

        return valid, differences

    
    def unified_reward_func(self,
                      state,
                      func_match_max_score: float = 0.5, 
                      state_match_max_score: float = 0.5):
        """Calculate reward based on state and function call matching."""
        # State matching: compare environment objects
        num_state_matches = 0
        num_state_total = 0
        for key in state["ground_truth_environment"]:
            # Compare attributes using compare_instances
            valid, diffs = self.compare_instances(state["ground_truth_environment"][key], state["environment"][key])

            num_state_matches += int(valid)
            num_state_total += 1

        # State score = match ratio * max score
        state_score = state_match_max_score * (num_state_matches / num_state_total)

        # Function call matching
        num_func_matches = 0
        num_func_total = 0
        model_func_calls = state["successful_func_calls"]
        ground_truth_func_calls = state["sample"]['answers']
        assert len(model_func_calls) == len(ground_truth_func_calls)

        for model_calls, gt_calls_str in zip(model_func_calls, ground_truth_func_calls):
            gt_calls = [self._parse_function_call(call_str) for call_str in gt_calls_str]
            
            def make_hashable(value):
                """Convert nested structures to hashable form for comparison."""
                if isinstance(value, dict):
                    return frozenset((k, make_hashable(v)) for k, v in value.items())
                elif isinstance(value, list):
                    return tuple(make_hashable(item) for item in value)
                elif isinstance(value, set):
                    return frozenset(make_hashable(item) for item in value)
                return value

            # Convert model calls to hashable format for comparison
            comparable_model_calls = []
            for call in model_calls:
                try:
                    comparable_model_calls.append((call["name"], frozenset((k, make_hashable(v)) for k, v in call["args"].items())))
                except Exception:
                    raise Exception("Error in Parsing Model Function Call is Not Expected!!")
            
            # Convert lists in gt parameters to tuples for comparison
            for call in gt_calls:
                if "args" in call:
                    for key, value in call["args"].items():
                        if isinstance(value, list):
                            call["args"][key] = tuple(value)
                else:
                    raise Exception("Error in Parsing Ground Truth Function Call is Not Expected!!")

            comparable_gt_calls = [
                (call["name"], frozenset((k, make_hashable(v)) for k, v in call["args"].items()))
                for call in gt_calls
            ]

            # Check if GT is an unordered subsequence of output
            is_match, _ = self._is_subsequence_unordered(comparable_gt_calls, comparable_model_calls)
            num_func_matches += int(is_match)
            num_func_total += 1
        func_score = func_match_max_score * (num_func_matches / num_func_total)

        # Base score = state score + function score
        base_score = state_score + func_score

        # If not all correct, no score
        if base_score != state_match_max_score + func_match_max_score:
            final_score = 0
        else:
            final_score = base_score

        score_info = {
            "state_score": state_score,
            "func_score": func_score,
            "base_score": base_score,
        }
        return final_score, score_info
        