"""Model clients (wrapper for top-level models module)."""

import sys
from pathlib import Path

# Import from top-level models module
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models import LLMClient, Message, ModelRegistry, VLMClient  # noqa: E402

from synthetic_data.config import ModelConfig

from .types import Sample

__all__ = ["LLMClient", "Message", "ModelRegistry", "Sample", "VLMClient"]


# Adapter for synthetic_data config compatibility
class SyntheticDataModelRegistry(ModelRegistry):
    """Adapter for synthetic_data ModelConfig compatibility."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        # Register endpoints from config
        for endpoint in config.endpoints:
            from models.registry import ModelEndpoint

            self.register_endpoint(
                ModelEndpoint(
                    name=endpoint.name,
                    base_url=endpoint.base_url,
                    api_key=endpoint.api_key,
                    model_name=endpoint.model_name,
                    max_tokens=endpoint.max_tokens,
                    temperature=endpoint.temperature,
                    service_tier=endpoint.service_tier,
                    top_p=endpoint.top_p,
                    min_p=endpoint.min_p,
                    top_k=endpoint.top_k,
                    presence_penalty=endpoint.presence_penalty,
                    frequency_penalty=endpoint.frequency_penalty,
                    repetition_penalty=endpoint.repetition_penalty,
                )
            )


# Replace ModelRegistry with the adapter
ModelRegistry = SyntheticDataModelRegistry
