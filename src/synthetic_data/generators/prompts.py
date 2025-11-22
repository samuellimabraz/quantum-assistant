"""Prompt management"""

from dataclasses import dataclass

from synthetic_data.config.schema import QuestionType


# Image-specific instructions appended to system prompts
IMAGE_QUESTION_INSTRUCTIONS = """
IMPORTANT - Image Context Available:
- The image description provided contains crucial visual information (circuits, gates, diagrams, formulas, graphs)
- Generate questions that REQUIRE examining these visual elements to answer
- Reference SPECIFIC visual components directly (gate types, qubit indices, circuit structure, diagram elements)
- Make the visual content ESSENTIAL to the question - not just supplementary
- Examples: "How implement the circuit shown in the image?", "What gate sequence is applied to qubits 0-2?", "What rotation angle does the RY gate use?"
"""

IMAGE_ANSWER_INSTRUCTIONS = """
IMPORTANT - Image Description Provided:
- An image description is included in the context showing visual elements
- Use this visual information to inform your answer when relevant
- Reference visual elements naturally when explaining your solution
- The image content is provided to help you give accurate, complete answers
"""


@dataclass
class PromptSet:
    """Set of prompts for generation tasks."""

    # Required prompts (no defaults)
    question_generation: str
    answer_generation: str
    summary_generation: str
    caption_generation: str
    code_generation: str
    content_quality_check: str
    image_quality_check: str
    category_classification: str

    # Optional system prompts (with defaults)
    question_generation_system: str = ""
    answer_generation_system: str = ""
    content_filter_system: str = ""
    image_filter_system: str = ""
    category_classification_system: str = ""
    sample_curation: str = ""
    sample_curation_system: str = ""

    def get_question_system_prompt(self, use_image: bool = False) -> str:
        """
        Get system prompt for question generation with optional image enhancement.

        Args:
            use_image: Whether to append image-specific instructions

        Returns:
            System prompt string
        """
        base_prompt = self.question_generation_system
        if use_image and base_prompt:
            return f"{base_prompt}\n{IMAGE_QUESTION_INSTRUCTIONS}"
        return base_prompt

    def get_answer_system_prompt(self, use_image: bool = False) -> str:
        """
        Get system prompt for answer generation with optional image enhancement.

        Args:
            use_image: Whether to append image-specific instructions

        Returns:
            System prompt string
        """
        base_prompt = self.answer_generation_system
        if use_image and base_prompt:
            return f"{base_prompt}\n{IMAGE_ANSWER_INSTRUCTIONS}"
        return base_prompt

    def get_question_prompt(self, question_type: QuestionType) -> str:
        """
        Get user prompt template for a question type.

        Args:
            question_type: Type of question to generate

        Returns:
            Prompt template string
        """
        mapping = {
            QuestionType.QA: self.question_generation,
            QuestionType.CODE: self.code_generation,
            QuestionType.CAPTION: self.caption_generation,
            QuestionType.SUMMARY: self.summary_generation,
        }
        return mapping.get(question_type, self.question_generation)
