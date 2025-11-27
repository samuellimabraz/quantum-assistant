"""Generation tracer for detailed prompt/response inspection."""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Any
import threading


@dataclass
class TraceEntry:
    """A single trace entry for a generation step."""

    timestamp: str
    stage: str  # input_generation, test_generation, answer_generation, correction
    step_type: str  # system_prompt, user_prompt, assistant_response
    question_type: Optional[str] = None
    candidate_id: Optional[str] = None
    iteration: int = 0
    content: str = ""
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "stage": self.stage,
            "step_type": self.step_type,
            "question_type": self.question_type,
            "candidate_id": self.candidate_id,
            "iteration": self.iteration,
            "content": self.content,
            "metadata": self.metadata,
        }


@dataclass
class ConversationTrace:
    """A complete conversation trace with all messages."""

    id: str
    stage: str
    question_type: Optional[str] = None
    started_at: str = ""
    completed_at: str = ""
    success: bool = False
    total_iterations: int = 0
    messages: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def add_message(self, role: str, content: str, iteration: int = 0, **kwargs):
        """Add a message to the conversation."""
        self.messages.append({
            "role": role,
            "content": content,
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            **kwargs,
        })

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "stage": self.stage,
            "question_type": self.question_type,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "success": self.success,
            "total_iterations": self.total_iterations,
            "messages": self.messages,
            "metadata": self.metadata,
        }


class GenerationTracer:
    """Traces generation pipeline for inspection and debugging.
    
    Logs complete prompts and responses for:
    - Input generation (question prompts)
    - Test generation (with correction loop)
    - Answer generation (response + validation)
    - Error correction iterations
    
    All traces are saved to a JSONL file for easy inspection.
    """

    def __init__(
        self,
        output_dir: Path,
        enabled: bool = True,
        log_to_console: bool = False,
    ):
        """Initialize tracer.
        
        Args:
            output_dir: Directory to save trace files
            enabled: Whether tracing is enabled
            log_to_console: Whether to also print traces to console
        """
        self.output_dir = Path(output_dir)
        self.enabled = enabled
        self.log_to_console = log_to_console
        self._lock = threading.Lock()
        self._conversation_counter = 0
        
        if self.enabled:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.trace_file = self.output_dir / "generation_traces.jsonl"
            self.conversations_file = self.output_dir / "conversations.jsonl"
            # Clear previous traces
            self.trace_file.write_text("")
            self.conversations_file.write_text("")

    def _get_conversation_id(self) -> str:
        """Generate unique conversation ID."""
        with self._lock:
            self._conversation_counter += 1
            return f"conv_{self._conversation_counter:06d}"

    def start_conversation(
        self,
        stage: str,
        question_type: Optional[str] = None,
        **metadata,
    ) -> ConversationTrace:
        """Start a new conversation trace.
        
        Args:
            stage: Stage name (input_generation, test_generation, answer_generation)
            question_type: Type of question being generated
            **metadata: Additional metadata
            
        Returns:
            ConversationTrace object to track the conversation
        """
        if not self.enabled:
            return ConversationTrace(id="", stage=stage)
        
        conv = ConversationTrace(
            id=self._get_conversation_id(),
            stage=stage,
            question_type=question_type,
            started_at=datetime.now().isoformat(),
            metadata=metadata,
        )
        return conv

    def log_message(
        self,
        conversation: ConversationTrace,
        role: str,
        content: str,
        iteration: int = 0,
        **kwargs,
    ):
        """Log a message in a conversation.
        
        Args:
            conversation: The conversation trace
            role: Message role (system, user, assistant)
            content: Message content (full, not truncated)
            iteration: Iteration number for correction loops
            **kwargs: Additional metadata for the message
        """
        if not self.enabled:
            return
        
        conversation.add_message(role, content, iteration, **kwargs)
        
        # Also log as individual trace entry for streaming inspection
        entry = TraceEntry(
            timestamp=datetime.now().isoformat(),
            stage=conversation.stage,
            step_type=f"{role}_message",
            question_type=conversation.question_type,
            candidate_id=conversation.id,
            iteration=iteration,
            content=content,
            metadata=kwargs,
        )
        self._write_trace(entry)
        
        if self.log_to_console:
            self._print_trace(entry)

    def complete_conversation(
        self,
        conversation: ConversationTrace,
        success: bool = True,
        total_iterations: int = 0,
        **metadata,
    ):
        """Mark conversation as complete and save to file.
        
        Args:
            conversation: The conversation trace
            success: Whether the generation was successful
            total_iterations: Total iterations used
            **metadata: Additional final metadata
        """
        if not self.enabled:
            return
        
        conversation.completed_at = datetime.now().isoformat()
        conversation.success = success
        conversation.total_iterations = total_iterations
        conversation.metadata.update(metadata)
        
        self._write_conversation(conversation)

    def log_entry(
        self,
        stage: str,
        step_type: str,
        content: str,
        question_type: Optional[str] = None,
        candidate_id: Optional[str] = None,
        iteration: int = 0,
        **metadata,
    ):
        """Log a standalone trace entry.
        
        Args:
            stage: Generation stage
            step_type: Type of step
            content: Content to log (full, not truncated)
            question_type: Question type if applicable
            candidate_id: Candidate ID if applicable
            iteration: Iteration number
            **metadata: Additional metadata
        """
        if not self.enabled:
            return
        
        entry = TraceEntry(
            timestamp=datetime.now().isoformat(),
            stage=stage,
            step_type=step_type,
            question_type=question_type,
            candidate_id=candidate_id,
            iteration=iteration,
            content=content,
            metadata=metadata,
        )
        self._write_trace(entry)
        
        if self.log_to_console:
            self._print_trace(entry)

    def _write_trace(self, entry: TraceEntry):
        """Write trace entry to file."""
        with self._lock:
            with open(self.trace_file, "a", encoding="utf-8") as f:
                json.dump(entry.to_dict(), f, ensure_ascii=False)
                f.write("\n")

    def _write_conversation(self, conversation: ConversationTrace):
        """Write conversation to file."""
        with self._lock:
            with open(self.conversations_file, "a", encoding="utf-8") as f:
                json.dump(conversation.to_dict(), f, ensure_ascii=False)
                f.write("\n")

    def _print_trace(self, entry: TraceEntry):
        """Print trace entry to console."""
        print(f"\n{'='*80}")
        print(f"[{entry.stage}] {entry.step_type} (iteration={entry.iteration})")
        print(f"Question type: {entry.question_type}")
        print(f"{'='*80}")
        print(entry.content[:2000])  # Truncate for console only
        if len(entry.content) > 2000:
            print(f"... (truncated, {len(entry.content)} chars total)")
        print(f"{'='*80}\n")

    def get_summary(self) -> dict:
        """Get summary statistics of traces."""
        if not self.enabled or not self.conversations_file.exists():
            return {}
        
        stages = {}
        total = 0
        successful = 0
        
        with open(self.conversations_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    conv = json.loads(line)
                    total += 1
                    if conv.get("success"):
                        successful += 1
                    
                    stage = conv.get("stage", "unknown")
                    if stage not in stages:
                        stages[stage] = {"total": 0, "successful": 0, "avg_iterations": 0, "iterations_sum": 0}
                    stages[stage]["total"] += 1
                    stages[stage]["iterations_sum"] += conv.get("total_iterations", 0)
                    if conv.get("success"):
                        stages[stage]["successful"] += 1
        
        # Calculate averages
        for stage in stages:
            if stages[stage]["total"] > 0:
                stages[stage]["avg_iterations"] = stages[stage]["iterations_sum"] / stages[stage]["total"]
            del stages[stage]["iterations_sum"]
        
        return {
            "total_conversations": total,
            "successful": successful,
            "success_rate": successful / total if total > 0 else 0,
            "by_stage": stages,
        }

