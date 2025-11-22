"""Code verification and correction system for generated samples."""

import ast
import re
import subprocess
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

    def verify_and_correct_sample(self, text: str, question: str = "") -> CodeVerificationResult:
        """
        Verify and correct code in a sample's answer.

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

        error_info = self._verify_code(main_code)

        if not error_info:
            return CodeVerificationResult(is_valid=True)

        corrected_code = main_code
        iterations = 0
        conversation_history = []

        for iteration in range(self.max_iterations):
            iterations = iteration + 1

            corrected_code = self._request_correction(
                code=corrected_code,
                error_message=error_info["message"],
                error_type=error_info["type"],
                question=question,
                original_answer=text,
                conversation_history=conversation_history,
            )

            if not corrected_code:
                break

            error_info = self._verify_code(corrected_code)

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

    def _verify_code(self, code: str) -> Optional[dict]:
        """
        Verify Python code for errors.

        Args:
            code: Python code string

        Returns:
            Dict with error info if invalid, None if valid
        """
        # Step 1: Check syntax
        syntax_error = self._check_syntax(code)
        if syntax_error:
            return {"type": "syntax", "message": syntax_error}

        # Step 2: Try to execute (catches import errors, undefined names, etc.)
        execution_error = self._check_execution(code)
        if execution_error:
            return {"type": "execution", "message": execution_error}

        return None

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

    def _check_execution(self, code: str) -> Optional[str]:
        """
        Check code execution (with safety constraints).

        Args:
            code: Python code string

        Returns:
            Error message if execution fails, None if succeeds
        """
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            temp_path = f.name

        try:
            result = subprocess.run(
                [sys.executable, temp_path],
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )

            if result.returncode != 0:
                stderr = result.stderr.strip()
                error_lines = stderr.split("\n")
                relevant_error = next(
                    (line for line in reversed(error_lines) if "Error:" in line), stderr
                )
                return f"Execution error: {relevant_error}"

            return None

        except subprocess.TimeoutExpired:
            return f"Execution timeout (>{self.timeout_seconds}s)"
        except (OSError, IOError) as e:
            return f"Execution error: {str(e)}"
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def _request_correction(
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
            response = self.llm_client.generate(messages, temperature=0.1)
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


async def verify_and_correct_batch(
    samples: list,
    llm_client: LLMClient,
    max_iterations: int = 3,
    timeout_seconds: int = 30,
) -> tuple[list, list]:
    """
    Verify and correct code in a batch of samples.

    Args:
        samples: List of Sample objects with code
        llm_client: LLM client for corrections
        max_iterations: Max correction iterations per sample
        timeout_seconds: Execution timeout

    Returns:
        Tuple of (verified_samples, failed_samples)
    """
    verifier = CodeVerifier(
        llm_client=llm_client,
        max_iterations=max_iterations,
        timeout_seconds=timeout_seconds,
    )

    verified_samples = []
    failed_samples = []

    for sample in samples:
        # Only verify samples that likely contain code
        if sample.question_type in ["code", "qa"]:
            result = verifier.verify_and_correct_sample(sample.answer, question=sample.question)

            if result.is_valid:
                # Update sample if code was corrected
                if result.corrected_code:
                    sample.answer = result.corrected_code
                verified_samples.append(sample)
            else:
                failed_samples.append(
                    {
                        "sample": sample,
                        "error": result.error_message,
                        "error_type": result.error_type,
                    }
                )
        else:
            # Non-code samples pass through
            verified_samples.append(sample)

    return verified_samples, failed_samples
