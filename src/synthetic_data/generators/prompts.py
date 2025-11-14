"""Prompt management from configuration."""

from dataclasses import dataclass

from synthetic_data.config.schema import QuestionType


@dataclass
class PromptSet:
    """Set of prompts for generation tasks."""

    question_generation: str
    answer_generation: str
    summary_generation: str
    caption_generation: str
    code_generation: str
    content_quality_check: str
    image_quality_check: str
    category_classification: str

    def get_question_prompt(self, question_type: QuestionType) -> str:
        """Get prompt template for a question type."""
        mapping = {
            QuestionType.QA: self.question_generation,
            QuestionType.CODE: self.code_generation,
            QuestionType.CAPTION: self.caption_generation,
            QuestionType.SUMMARY: self.summary_generation,
        }
        return mapping.get(question_type, self.question_generation)
