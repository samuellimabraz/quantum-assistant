"""Utility for extracting functions from code for HumanEval-style evaluation."""

import ast
import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class FunctionInfo:
    """Information about an extracted function."""

    name: str
    signature: str
    docstring: Optional[str]
    body: str
    full_definition: str
    imports: list[str]
    start_line: int
    end_line: int


class FunctionExtractor:
    """
    Extract function definitions from Python code.

    This utility enables converting full code samples into function completion format
    for evaluation on HumanEval-style benchmarks.
    """

    @staticmethod
    def extract_functions(code: str) -> list[FunctionInfo]:
        """
        Extract all function definitions from code.

        Args:
            code: Python source code

        Returns:
            List of FunctionInfo objects
        """
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return []

        # Extract imports
        imports = FunctionExtractor._extract_imports(tree)

        # Extract functions
        functions = []
        code_lines = code.split("\n")

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_info = FunctionExtractor._extract_function_info(node, code_lines, imports)
                if func_info:
                    functions.append(func_info)

        return functions

    @staticmethod
    def _extract_imports(tree: ast.AST) -> list[str]:
        """Extract all import statements from AST."""
        imports = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.asname:
                        imports.append(f"import {alias.name} as {alias.asname}")
                    else:
                        imports.append(f"import {alias.name}")

            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    if alias.asname:
                        imports.append(f"from {module} import {alias.name} as {alias.asname}")
                    else:
                        imports.append(f"from {module} import {alias.name}")

        return imports

    @staticmethod
    def _extract_function_info(
        node: ast.FunctionDef, code_lines: list[str], imports: list[str]
    ) -> Optional[FunctionInfo]:
        """Extract function information from AST node."""
        # Get function name
        name = node.name

        # Get function signature
        args = FunctionExtractor._format_arguments(node.args)
        signature = f"def {name}({args}):"

        # Get docstring
        docstring = ast.get_docstring(node)

        # Get function body (line range)
        start_line = node.lineno - 1  # 0-indexed
        end_line = node.end_lineno if node.end_lineno else start_line + 1

        # Extract full function definition
        full_definition = "\n".join(code_lines[start_line:end_line])

        # Extract body (without signature and docstring)
        body_lines = code_lines[start_line + 1 : end_line]

        # Remove docstring if present
        if docstring:
            # Find and remove docstring lines
            body_lines = FunctionExtractor._remove_docstring_lines(body_lines, docstring)

        body = "\n".join(body_lines).strip()

        return FunctionInfo(
            name=name,
            signature=signature,
            docstring=docstring,
            body=body,
            full_definition=full_definition,
            imports=imports,
            start_line=start_line,
            end_line=end_line,
        )

    @staticmethod
    def _format_arguments(args: ast.arguments) -> str:
        """Format function arguments as string."""
        arg_parts = []

        # Regular arguments
        for i, arg in enumerate(args.args):
            arg_str = arg.arg
            # Add type annotation if present
            if arg.annotation:
                arg_str += f": {ast.unparse(arg.annotation)}"
            # Add default value if present
            defaults_offset = len(args.args) - len(args.defaults)
            if i >= defaults_offset:
                default_idx = i - defaults_offset
                arg_str += f" = {ast.unparse(args.defaults[default_idx])}"
            arg_parts.append(arg_str)

        # *args
        if args.vararg:
            vararg_str = f"*{args.vararg.arg}"
            if args.vararg.annotation:
                vararg_str += f": {ast.unparse(args.vararg.annotation)}"
            arg_parts.append(vararg_str)

        # **kwargs
        if args.kwarg:
            kwarg_str = f"**{args.kwarg.arg}"
            if args.kwarg.annotation:
                kwarg_str += f": {ast.unparse(args.kwarg.annotation)}"
            arg_parts.append(kwarg_str)

        return ", ".join(arg_parts)

    @staticmethod
    def _remove_docstring_lines(body_lines: list[str], docstring: str) -> list[str]:
        """Remove docstring lines from function body."""
        # Simple approach: find consecutive lines matching docstring content
        docstring_lines = docstring.split("\n")
        result = []
        skip_mode = False
        skip_count = 0

        for line in body_lines:
            stripped = line.strip()

            # Check for docstring delimiters
            if stripped.startswith('"""') or stripped.startswith("'''"):
                if skip_mode:
                    skip_mode = False
                    continue
                else:
                    skip_mode = True
                    # Check if single-line docstring
                    if stripped.count('"""') == 2 or stripped.count("'''") == 2:
                        skip_mode = False
                    continue

            if skip_mode:
                continue

            result.append(line)

        return result

    @staticmethod
    def create_function_stub(func_info: FunctionInfo) -> str:
        """
        Create a function stub for completion (signature + docstring only).

        Args:
            func_info: Function information

        Returns:
            Function stub code
        """
        parts = []

        # Add imports
        if func_info.imports:
            parts.append("\n".join(func_info.imports))
            parts.append("")

        # Add signature
        parts.append(func_info.signature)

        # Add docstring if present
        if func_info.docstring:
            parts.append(f'    """{func_info.docstring}"""')
            parts.append("    pass")
        else:
            parts.append("    pass")

        return "\n".join(parts)

    @staticmethod
    def extract_primary_function(code: str) -> Optional[FunctionInfo]:
        """
        Extract the primary (longest) function from code.

        Args:
            code: Python source code

        Returns:
            FunctionInfo for the primary function or None
        """
        functions = FunctionExtractor.extract_functions(code)

        if not functions:
            return None

        # Return longest function (by body length)
        return max(functions, key=lambda f: len(f.body))

    @staticmethod
    def code_to_humaneval_format(
        code: str, task_description: str, function_name: Optional[str] = None
    ) -> dict:
        """
        Convert full code to HumanEval format.

        Args:
            code: Complete Python code
            task_description: Description of what the function should do
            function_name: Optional specific function to extract (otherwise uses primary)

        Returns:
            Dict with 'prompt', 'canonical_solution', 'entry_point', 'imports'
        """
        functions = FunctionExtractor.extract_functions(code)

        if not functions:
            return {
                "prompt": task_description,
                "canonical_solution": code,
                "entry_point": None,
                "imports": [],
            }

        # Select target function
        if function_name:
            target_func = next((f for f in functions if f.name == function_name), None)
            if not target_func:
                target_func = functions[0]
        else:
            target_func = FunctionExtractor.extract_primary_function(code)

        if not target_func:
            return {
                "prompt": task_description,
                "canonical_solution": code,
                "entry_point": None,
                "imports": [],
            }

        # Build HumanEval-style prompt
        args_desc = FunctionExtractor._describe_arguments(target_func.signature)
        prompt = f"{task_description}\nYou must implement this using a function named `{target_func.name}` {args_desc}."

        return {
            "prompt": prompt,
            "canonical_solution": target_func.full_definition,
            "entry_point": target_func.name,
            "imports": target_func.imports,
        }

    @staticmethod
    def _describe_arguments(signature: str) -> str:
        """Generate argument description for HumanEval prompt."""
        match = re.search(r"def \w+\((.*?)\):", signature)
        if not match:
            return "with no arguments"

        args_str = match.group(1).strip()

        if not args_str:
            return "with no arguments"

        # Parse argument names (ignore defaults and annotations)
        arg_names = []
        for arg in args_str.split(","):
            arg = arg.strip()
            # Remove type annotations and defaults
            arg_name = arg.split(":")[0].split("=")[0].strip()
            if arg_name and not arg_name.startswith("*"):
                arg_names.append(arg_name)

        if not arg_names:
            return "with no arguments"

        return f"with the following arguments: {', '.join(arg_names)}"
