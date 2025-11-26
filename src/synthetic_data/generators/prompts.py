from dataclasses import dataclass
from typing import Optional

from synthetic_data.config.schema import QuestionType


# Image-specific instructions appended to prompts when multimodal
IMAGE_CONTEXT_INSTRUCTIONS = """
MULTIMODAL INPUT REQUIREMENTS:
- An image description is provided with specific visual information
- The image content MUST be essential to understanding and answering
- Reference SPECIFIC visual elements in your question/answer:
  * For circuits: gate types (H, X, CNOT), qubit indices (q_0, q_1), angles (π/2, θ)
  * For histograms: measurement outcomes (|00⟩, |11⟩), probabilities, shot counts
  * For Bloch spheres: state vectors, rotation angles, axis directions
  * For diagrams: labeled components, connections, mathematical expressions
- The task MUST require examining the image to answer correctly
- DO NOT make the image optional or decorative

MULTIMODAL EXAMPLES:
- function_completion: "Implement the circuit shown in the image" (docstring describes gates from image)
- code_generation: "Create a circuit that produces the measurement distribution shown in the histogram"
- qa: "Analyze the quantum circuit in the image. What state does it prepare?"
"""


@dataclass
class PromptSet:
    # Question type prompts
    function_completion_prompt: str
    code_generation_prompt: str
    qa_prompt: str

    # Answer prompts
    answer_with_test_prompt: str
    answer_without_test_prompt: str

    # System prompts
    question_generation_system: str = ""
    answer_generation_system: str = ""
    test_generation_system: str = ""

    # Quality control prompts
    content_quality_check: str = ""
    content_filter_system: str = ""
    image_quality_check: str = ""
    image_filter_system: str = ""
    image_transcription: str = ""
    image_transcription_system: str = ""

    # Classification and curation
    category_classification: str = ""
    category_classification_system: str = ""
    sample_curation: str = ""
    sample_curation_system: str = ""

    def get_question_prompt(self, question_type: QuestionType) -> str:
        """Get the user prompt template for a question type.

        Args:
            question_type: Type of question to generate

        Returns:
            Prompt template string with {context} placeholder
        """
        mapping = {
            QuestionType.FUNCTION_COMPLETION: self.function_completion_prompt,
            QuestionType.CODE_GENERATION: self.code_generation_prompt,
            QuestionType.QA: self.qa_prompt,
        }
        return mapping.get(question_type, self.qa_prompt)

    def get_question_system_prompt(self, use_image: bool = False) -> str:
        """Get system prompt for question generation.

        Args:
            use_image: Whether to include image-specific instructions

        Returns:
            System prompt string
        """
        base = self.question_generation_system
        if use_image and base:
            return f"{base}\n{IMAGE_CONTEXT_INSTRUCTIONS}"
        return base

    def get_answer_prompt(
        self,
        question_type: QuestionType,
        question: str,
        context: str,
        test_code: Optional[str] = None,
    ) -> str:
        """Get the answer generation prompt.

        For code types (function_completion, code_generation), includes the test
        that the code must pass. For qa type, no test is provided.

        Args:
            question_type: Type of question
            question: The generated question
            context: Source context
            test_code: Unit test code (for code types)

        Returns:
            Formatted answer prompt
        """
        if question_type in (QuestionType.FUNCTION_COMPLETION, QuestionType.CODE_GENERATION):
            if test_code:
                return self.answer_with_test_prompt.format(
                    question=question,
                    context=context,
                    test_code=test_code,
                )

        return self.answer_without_test_prompt.format(
            question=question,
            context=context,
        )

    def get_answer_system_prompt(self, use_image: bool = False) -> str:
        """Get system prompt for answer generation.

        Args:
            use_image: Whether to include image-specific instructions

        Returns:
            System prompt string
        """
        base = self.answer_generation_system
        if use_image and base:
            return f"{base}\n{IMAGE_CONTEXT_INSTRUCTIONS}"
        return base


def build_context(
    chunk_text: str,
    previous_text: str = "",
    next_text: str = "",
    code_context: str = "",
    image_description: str = "",
    max_length: int = 3200,
) -> str:
    """Build the context string for generation prompts.

    Organizes context with image description first (if multimodal),
    followed by surrounding text and code context.

    Args:
        chunk_text: Main chunk content
        previous_text: Text from previous chunk
        next_text: Text from next chunk
        code_context: All code from source document
        image_description: VLM transcription of associated image
        max_length: Maximum context length

    Returns:
        Formatted context string
    """
    parts = []

    if image_description:
        parts.append(f"[Image Description]\n{image_description}")

    # Previous context
    if previous_text:
        parts.append(f"[Previous Context]\n{previous_text}")

    # Main content (truncated if needed)
    main_text = chunk_text
    parts.append(f"[Main Content]\n{main_text}")

    # Next context
    if next_text:
        parts.append(f"[Next Context]\n{next_text}")

    # Code context
    if code_context:
        parts.append(f"[Code from Document]\n```python\n{code_context}\n```")

    return "\n\n".join(parts)


def extract_entry_point_from_prompt(prompt: str) -> Optional[str]:
    """Extract function name from a function completion or code generation prompt.

    Looks for patterns like:
    - def function_name(
    - function named `function_name`
    - a function named `function_name`
    
    Avoids false positives like 'named', 'function', 'that', 'name', etc.

    Args:
        prompt: The function completion or code generation prompt

    Returns:
        Function name or None
    """
    import re

    # Priority 1: Qiskit HumanEval Hard pattern - "function named `name`" at end
    # This is the most specific pattern
    match = re.search(
        r"function\s+named\s+[`'\"]([a-zA-Z_][a-zA-Z0-9_]*)[`'\"]",
        prompt,
        re.IGNORECASE
    )
    if match:
        name = match.group(1)
        # Filter out common false positives
        if name.lower() not in ("name", "that", "this", "function", "it", "one"):
            return name

    # Priority 2: def function_name( pattern (for function completion stubs)
    match = re.search(r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", prompt)
    if match:
        name = match.group(1)
        if name.lower() not in ("name", "that", "this", "function", "it", "one"):
            return name

    # Priority 3: Look for backtick-quoted names after "named"
    match = re.search(r"named\s+`([a-zA-Z_][a-zA-Z0-9_]*)`", prompt, re.IGNORECASE)
    if match:
        name = match.group(1)
        if name.lower() not in ("name", "that", "this", "function", "it", "one"):
            return name

    return None
