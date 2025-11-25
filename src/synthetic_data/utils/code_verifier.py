"""Code verification and correction system for generated samples."""

import ast
import asyncio
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from synthetic_data.models import LLMClient, Message


@dataclass
class CodeVerificationResult:
    """Result of code verification."""

    is_valid: bool
    error_message: Optional[str] = None
    error_type: Optional[str] = None
    corrected_code: Optional[str] = None
    iterations_used: int = 0


class CodeVerifier:
    """
    Verifies and corrects Python code in generated samples.

    This verifier:
    1. Extracts code blocks from text
    2. Validates code (syntax and execution)
    3. If errors found, asks the model to fix them using conversation context
    4. Repeats with configurable max iterations
    """

    def __init__(
        self,
        llm_client: LLMClient,
        max_iterations: int = 3,
        timeout_seconds: int = 30,
    ):
        """
        Initialize code verifier.

        Args:
            llm_client: LLM client for code correction
            max_iterations: Maximum correction attempts
            timeout_seconds: Timeout for code execution
        """
        self.llm_client = llm_client
        self.max_iterations = max_iterations
        self.timeout_seconds = timeout_seconds

    async def verify_and_correct_sample_async(
        self, text: str, question: str = ""
    ) -> CodeVerificationResult:
        """
        Verify and correct code in a sample's answer (async version).

        Args:
            text: The text containing code blocks
            question: The original question (provides context for correction)

        Returns:
            CodeVerificationResult with validation status and corrected code if needed
        """
        code_blocks = self._extract_code_blocks(text)

        if not code_blocks:
            return CodeVerificationResult(is_valid=True)

        main_code = self._select_main_code_block(code_blocks)

        if not main_code:
            return CodeVerificationResult(is_valid=True)

        error_info = await self._verify_code_async(main_code)

        if not error_info:
            return CodeVerificationResult(is_valid=True)

        corrected_code = main_code
        iterations = 0
        conversation_history = []

        for iteration in range(self.max_iterations):
            iterations = iteration + 1

            corrected_code = await self._request_correction_async(
                code=corrected_code,
                error_message=error_info["message"],
                error_type=error_info["type"],
                question=question,
                original_answer=text,
                conversation_history=conversation_history,
            )

            if not corrected_code:
                break

            error_info = await self._verify_code_async(corrected_code)

            if not error_info:
                corrected_text = self._replace_code_in_text(text, main_code, corrected_code)
                return CodeVerificationResult(
                    is_valid=True,
                    corrected_code=corrected_text,
                    iterations_used=iterations,
                )

            conversation_history.append(
                {
                    "attempt": iteration,
                    "code": corrected_code,
                    "error": error_info["message"],
                }
            )

        return CodeVerificationResult(
            is_valid=False,
            error_message=error_info["message"] if error_info else "Unknown error",
            error_type=error_info["type"] if error_info else "unknown",
            iterations_used=iterations,
        )

    def _extract_code_blocks(self, text: str) -> list[str]:
        """
        Extract code blocks from markdown text.

        Args:
            text: Text containing code blocks

        Returns:
            List of code strings
        """
        pattern = r"```(?:python)?\s*\n(.*?)```"
        matches = re.findall(pattern, text, re.DOTALL)
        return [match.strip() for match in matches if match.strip()]

    def _select_main_code_block(self, code_blocks: list[str]) -> Optional[str]:
        """
        Select the main code block to verify (typically the longest).

        Args:
            code_blocks: List of code blocks

        Returns:
            Main code block or None
        """
        if not code_blocks:
            return None

        return max(code_blocks, key=len)

    def _check_syntax(self, code: str) -> Optional[str]:
        """
        Check Python syntax.

        Args:
            code: Python code string

        Returns:
            Error message if invalid syntax, None if valid
        """
        try:
            ast.parse(code)
            return None
        except SyntaxError as e:
            return f"Syntax error at line {e.lineno}: {e.msg}"
        except Exception as e:
            return f"Parsing error: {str(e)}"

    async def _verify_code_async(self, code: str) -> Optional[dict]:
        """
        Verify Python code for errors (async version).

        Args:
            code: Python code string

        Returns:
            Dict with error info if invalid, None if valid
        """
        # Step 1: Check syntax (synchronous, fast)
        syntax_error = self._check_syntax(code)
        if syntax_error:
            return {"type": "syntax", "message": syntax_error}

        # Step 2: Try to execute (async)
        execution_error = await self._check_execution_async(code)
        if execution_error:
            return {"type": "execution", "message": execution_error}

        return None

    async def _check_execution_async(self, code: str) -> Optional[str]:
        """
        Check code execution asynchronously.

        Args:
            code: Python code string

        Returns:
            Error message if execution fails, None if succeeds
        """
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            temp_path = f.name

        try:
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                temp_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                _, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.timeout_seconds,
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return f"Execution timeout (>{self.timeout_seconds}s)"

            if process.returncode != 0:
                stderr_text = stderr.decode("utf-8").strip()
                error_lines = stderr_text.split("\n")
                relevant_error = next(
                    (line for line in reversed(error_lines) if "Error:" in line),
                    stderr_text,
                )
                return f"Execution error: {relevant_error}"

            return None

        except (OSError, IOError) as e:
            return f"Execution error: {str(e)}"
        finally:
            Path(temp_path).unlink(missing_ok=True)

    async def _request_correction_async(
        self,
        code: str,
        error_message: str,
        error_type: str,
        question: str,
        original_answer: str,
        conversation_history: list,
    ) -> Optional[str]:
        """
        Request code correction from LLM using conversation context.

        Args:
            code: The code with errors
            error_message: Error message from verification
            error_type: Type of error (syntax, execution)
            question: The original question being answered
            original_answer: The original answer text
            conversation_history: Previous correction attempts

        Returns:
            Corrected code or None if correction failed
        """
        messages = []

        system_prompt = """Reasoning: high. You are a Python code correction expert.
Fix code errors while preserving the original logic and intent.
Return ONLY the corrected Python code in a single code block, nothing else.

Requirements:
- Include ALL necessary imports
- Keep code clean and functional
- Make minimal changes to fix the error
- Do NOT add comments or explanations outside the code block"""

        messages.append(Message(role="system", content=system_prompt))

        if question and not conversation_history:
            context_prompt = f"""Question: {question}

Original answer:
{original_answer}

The code in this answer has a {error_type} error:
{error_message}

Please fix the code."""
            messages.append(Message(role="user", content=context_prompt))

        elif conversation_history:
            if question:
                messages.append(
                    Message(
                        role="user",
                        content=f"Question: {question}\n\nOriginal answer:\n{original_answer}",
                    )
                )

            for attempt in conversation_history[:-1]:
                messages.append(
                    Message(
                        role="user",
                        content=f"Error: {attempt['error']}\n\nPlease fix:\n```python\n{attempt['code']}\n```",
                    )
                )
                messages.append(
                    Message(role="assistant", content=f"```python\n{attempt['code']}\n```")
                )

            # Current attempt
            last_attempt = conversation_history[-1]
            messages.append(
                Message(
                    role="user",
                    content=f"Still has error: {last_attempt['error']}\n\nFix the code:\n```python\n{code}\n```",
                )
            )
        else:
            messages.append(
                Message(
                    role="user",
                    content=f"{error_type} error: {error_message}\n\nCode:\n```python\n{code}\n```",
                )
            )

        try:
            response = await self.llm_client.generate_async(messages, temperature=0.1)
            corrected_blocks = self._extract_code_blocks(response)

            if corrected_blocks:
                return corrected_blocks[0]

            # Fallback: response might be just code without markers
            if "import" in response or "def " in response:
                return response.strip()

            return None

        except (ValueError, ConnectionError, TimeoutError):
            return None

    def _replace_code_in_text(self, original_text: str, old_code: str, new_code: str) -> str:
        """
        Replace code block in original text.

        Args:
            original_text: Original text with code blocks
            old_code: Old code to replace
            new_code: New corrected code

        Returns:
            Text with corrected code
        """
        old_block = f"```python\n{old_code}\n```"
        new_block = f"```python\n{new_code}\n```"

        if old_block in original_text:
            return original_text.replace(old_block, new_block, 1)

        old_block_no_lang = f"```\n{old_code}\n```"
        new_block_no_lang = f"```python\n{new_code}\n```"

        if old_block_no_lang in original_text:
            return original_text.replace(old_block_no_lang, new_block_no_lang, 1)

        # Fallback: just replace the code itself
        return original_text.replace(old_code, new_code, 1)
