"""Prompt management for the generation pipeline."""

from dataclasses import dataclass
from typing import Optional

from synthetic_data.config.schema import QuestionType


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
- DO NOT mention the image name, number, use a generic name like "the image" or "the diagram".

MULTIMODAL EXAMPLES:
- function_completion: "Implement the circuit shown in the image" (docstring describes gates from image)
- code_generation: "Create a circuit that produces the measurement distribution shown in the histogram"
- qa: "Analyze the quantum circuit in the image. What state does it prepare?"
If the images are described as a large circuit diagram, with too many elements (gates, qubits, etc.), then focus in some particular part, not the whole diagram.
"""


@dataclass
class PromptSet:
    """Complete set of prompts for the generation pipeline."""

    # === INPUT GENERATION SESSION PROMPTS ===
    input_generation_system: str = ""
    function_completion_prompt: str = ""
    code_generation_prompt: str = ""
    qa_prompt: str = ""
    test_generation_prompt: str = ""

    # === ANSWER GENERATION SESSION PROMPTS ===
    answer_generation_system: str = ""
    function_completion_answer_prompt: str = ""
    code_generation_answer_prompt: str = ""
    qa_answer_prompt: str = ""
    answer_correction_prompt: str = ""

    # === QUALITY CONTROL PROMPTS ===
    content_quality_check: str = ""
    content_filter_system: str = ""
    image_quality_check: str = ""
    image_filter_system: str = ""
    image_transcription: str = ""
    image_transcription_system: str = ""

    # === CANDIDATE FILTERING ===
    candidate_filter_system: str = ""
    candidate_filter_prompt: str = ""

    # === CLASSIFICATION AND CURATION ===
    category_classification: str = ""
    category_classification_system: str = ""
    sample_curation: str = ""
    sample_curation_system: str = ""

    def get_question_prompt(self, question_type: QuestionType) -> str:
        """Get the question prompt template for a question type."""
        mapping = {
            QuestionType.FUNCTION_COMPLETION: self.function_completion_prompt,
            QuestionType.CODE_GENERATION: self.code_generation_prompt,
            QuestionType.QA: self.qa_prompt,
        }
        return mapping.get(question_type, self.qa_prompt)

    def get_input_system_prompt(self, use_image: bool = False) -> str:
        """Get system prompt for input generation session."""
        base = self.input_generation_system
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
        """Get the answer generation prompt."""
        if question_type == QuestionType.FUNCTION_COMPLETION:
            return self.function_completion_answer_prompt.format(
                question=question,
                context=context,
                test_code=test_code or "N/A",
            )
        elif question_type == QuestionType.CODE_GENERATION:
            return self.code_generation_answer_prompt.format(
                question=question,
                context=context,
                test_code=test_code or "N/A",
            )
        else:
            return self.qa_answer_prompt.format(
                question=question,
                context=context,
            )

    def get_answer_system_prompt(self, use_image: bool = False) -> str:
        """Get system prompt for answer generation session."""
        base = self.answer_generation_system
        if use_image and base:
            return f"{base}\n{IMAGE_CONTEXT_INSTRUCTIONS}"
        return base


def build_context(
    chunk_text: str,
    previous_text: str = "",
    next_text: str = "",
    code_context: str = "",
    target_image_description: str = "",
    max_length: int = 4096,
) -> str:
    """Build the context string for generation prompts.

    The chunk_text should already have inline image transcriptions embedded.
    If a target_image_description is provided, it's emphasized at the start.

    Args:
        chunk_text: Main content (with inline image transcriptions)
        previous_text: Context from previous chunk
        next_text: Context from next chunk
        code_context: Additional code from document
        target_image_description: Emphasized image for multimodal
        max_length: Maximum context length

    Returns:
        Formatted context string
    """
    parts = []

    # Target image emphasized first for multimodal
    if target_image_description:
        parts.append(f"[Target Image]\n{target_image_description}")

    # Previous context
    if previous_text:
        parts.append(f"[Previous Context]\n{previous_text}")

    # Main content
    parts.append(f"[Main Content]\n{chunk_text}")

    # Next context
    if next_text:
        parts.append(f"[Next Context]\n{next_text}")

    # Code from document
    if code_context:
        parts.append(f"[Code from Document]\n```python\n{code_context}\n```")

    context = "\n\n".join(parts)

    # Truncate if too long
    if len(context) > max_length:
        # Keep target image and main content, truncate others
        if target_image_description:
            essential = (
                f"[Target Image]\n{target_image_description}\n\n[Main Content]\n{chunk_text}"
            )
        else:
            essential = f"[Main Content]\n{chunk_text}"

        if len(essential) > max_length:
            return essential[:max_length]
        return essential

    return context


def extract_entry_point_from_prompt(prompt: str) -> Optional[str]:
    """Extract function name from a function completion or code generation prompt."""
    import re

    patterns = [
        r"function\s+named\s+[`'\"]([a-zA-Z_][a-zA-Z0-9_]*)[`'\"]",
        r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(",
        r"named\s+`([a-zA-Z_][a-zA-Z0-9_]*)`",
    ]

    for pattern in patterns:
        match = re.search(pattern, prompt, re.IGNORECASE)
        if match:
            name = match.group(1)
            if name.lower() not in ("name", "that", "this", "function", "it", "one"):
                return name

    return None
