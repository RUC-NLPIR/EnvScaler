"""
Build environment class structure from Python code using AST analysis.
Extracts state types, method dependencies, and attribute usage patterns.
"""
import ast
from collections import defaultdict
from pathlib import Path

def ann_to_str(node):
    """Convert AST annotation node to string representation."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Subscript):
        base = ann_to_str(node.value)
        sl = node.slice
        if isinstance(sl, ast.Index):
            sub = ann_to_str(sl.value)
        elif isinstance(sl, ast.Tuple):
            sub = ", ".join(ann_to_str(elt) for elt in sl.elts)
        else:
            sub = ann_to_str(sl)
        return f"{base}[{sub}]"
    if isinstance(node, ast.Constant):
        return type(node.value).__name__
    if isinstance(node, ast.Str):
        return "str"
    return "unknown"

class ClassAnalyzer(ast.NodeVisitor):
    """AST visitor to analyze class structure, state types, and method dependencies."""
    
    def __init__(self, class_name):
        self.class_name = class_name
        self.states = defaultdict(lambda: {
            "type": None,
            "used_by": set(),
            "modified_by": set(),
        })
        self.methods = defaultdict(lambda: {
            "calls": set(),
            "reads": set(),
            "writes": set(),
        })
        self.current_method = None

    def visit_ClassDef(self, node):
        """Visit class definition and analyze its structure."""
        if node.name != self.class_name:
            return
        self._collect_state_types(node)
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                # Skip __init__ method
                if item.name == "__init__":
                    continue
                self.current_method = item.name
                _ = self.methods[self.current_method]  # Ensure method appears in output
                self._scan_method(item)

    def _base_attr_from_node(self, node):
        """Extract top-level attribute name from self.attr or self.attr[...] -> 'attr'."""
        curr = node
        while True:
            if isinstance(curr, ast.Attribute):
                if isinstance(curr.value, ast.Name) and curr.value.id == "self":
                    return curr.attr
                curr = curr.value
            elif isinstance(curr, ast.Subscript):
                v = curr.value
                if isinstance(v, ast.Attribute) and isinstance(v.value, ast.Name) and v.value.id == "self":
                    return v.attr
                curr = v
            elif isinstance(curr, ast.Call):
                curr = curr.func
            else:
                break
        return None

    def _scan_method(self, func_node):
        """Scan method body to detect method calls, attribute reads, and writes."""
        # Methods that modify containers in-place
        mutators = {"append", "extend", "insert", "update", "clear", "pop", "remove", "discard", "setdefault"}

        for stmt in ast.walk(func_node):
            # Track self.method() calls
            if isinstance(stmt, ast.Call) and isinstance(stmt.func, ast.Attribute):
                if isinstance(stmt.func.value, ast.Name) and stmt.func.value.id == "self":
                    self.methods[self.current_method]["calls"].add(stmt.func.attr)

            # Read: self.attr or self.attr[...] with Load context
            if isinstance(stmt, (ast.Attribute, ast.Subscript)) and isinstance(getattr(stmt, "ctx", None), ast.Load):
                base = self._base_attr_from_node(stmt)
                if base:
                    self.states[base]["used_by"].add(self.current_method)
                    self.methods[self.current_method]["reads"].add(base)

            # Write: assignment target is self.attr[...] or self.attr
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    base = self._base_attr_from_node(target)
                    if base:
                        self.states[base]["modified_by"].add(self.current_method)
                        self.methods[self.current_method]["writes"].add(base)

            # Write: augmented assignment (e.g., +=, -=)
            if isinstance(stmt, ast.AugAssign):
                base = self._base_attr_from_node(stmt.target)
                if base:
                    self.states[base]["modified_by"].add(self.current_method)
                    self.methods[self.current_method]["writes"].add(base)

            # Write: in-place container mutations (e.g., list.append, dict.update)
            if isinstance(stmt, ast.Call) and isinstance(stmt.func, ast.Attribute):
                base = self._base_attr_from_node(stmt.func.value)
                if base and stmt.func.attr in mutators:
                    self.states[base]["modified_by"].add(self.current_method)
                    self.methods[self.current_method]["writes"].add(base)

            # Write: deletion (del self.attr[...] or del self.attr)
            if isinstance(stmt, ast.Delete):
                for target in stmt.targets:
                    base = self._base_attr_from_node(target)
                    if base:
                        self.states[base]["modified_by"].add(self.current_method)
                        self.methods[self.current_method]["writes"].add(base)

    def _collect_state_types(self, class_node):
        """Collect attribute types from class-level and __init__ assignments."""
        # Class-level annotated assignments and assignments
        for stmt in class_node.body:
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                self.states[stmt.target.id]["type"] = ann_to_str(stmt.annotation)
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Name) and self.states[target.id]["type"] is None:
                        self.states[target.id]["type"] = ann_to_str(stmt.value)
        # self.xxx assignments in __init__
        for stmt in class_node.body:
            if isinstance(stmt, ast.FunctionDef) and stmt.name == "__init__":
                for inner in ast.walk(stmt):
                    if isinstance(inner, ast.AnnAssign):
                        if isinstance(inner.target, ast.Attribute) and isinstance(inner.target.value, ast.Name) and inner.target.value.id == "self":
                            self.states[inner.target.attr]["type"] = ann_to_str(inner.annotation)
                    if isinstance(inner, ast.Assign):
                        for tgt in inner.targets:
                            if isinstance(tgt, ast.Attribute) and isinstance(tgt.value, ast.Name) and tgt.value.id == "self":
                                if self.states[tgt.attr]["type"] is None:
                                    self.states[tgt.attr]["type"] = ann_to_str(inner.value)

    def get_tree(self):
        """Return structured analysis result with sorted lists."""
        return {
            "states": {
                k: {
                    "type": v["type"],
                    "used_by": sorted(v["used_by"]),
                    "modified_by": sorted(v["modified_by"]),
                }
                for k, v in self.states.items()
            },
            "methods": {
                k: {
                    "calls": sorted(v["calls"]),
                    "reads": sorted(v["reads"]),
                    "writes": sorted(v["writes"]),
                }
                for k, v in self.methods.items()
            }
        }

def build_class_tree_form_file(py_file, class_name):
    """Build class structure tree from Python file path."""
    src = Path(py_file).read_text(encoding="utf-8")
    tree = ast.parse(src)
    analyzer = ClassAnalyzer(class_name)
    analyzer.visit(tree)
    return analyzer.get_tree()

def build_class_tree_form_str(src, class_name):
    """Build class structure tree from Python source code string."""
    tree = ast.parse(src)
    analyzer = ClassAnalyzer(class_name)
    analyzer.visit(tree)
    return analyzer.get_tree()
    