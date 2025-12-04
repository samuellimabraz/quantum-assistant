"""Model registry for managing clients."""

from models.client import LLMClient, VLMClient


class ModelEndpoint:
    """Model endpoint configuration."""

    def __init__(
        self,
        name: str,
        base_url: str,
        api_key: str = "",
        model_name: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        timeout: float = 300.0,
        service_tier: str | None = None,
        top_p: float | None = None,
        min_p: float | None = None,
        top_k: int | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        repetition_penalty: float | None = None,
    ):
        self.name = name
        self.base_url = base_url
        self.api_key = api_key
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.service_tier = service_tier
        self.top_p = top_p
        self.min_p = min_p
        self.top_k = top_k
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.repetition_penalty = repetition_penalty


class ModelRegistry:
    """Registry for managing model clients."""

    def __init__(self):
        self._clients: dict[str, LLMClient] = {}
        self._endpoints: dict[str, ModelEndpoint] = {}

    def register_endpoint(self, endpoint: ModelEndpoint):
        """Register a model endpoint."""
        self._endpoints[endpoint.name] = endpoint

    def get_llm_client(self, model_name: str) -> LLMClient:
        """Get or create an LLM client."""
        if model_name in self._clients:
            return self._clients[model_name]

        endpoint = self._endpoints.get(model_name)
        if not endpoint:
            raise ValueError(f"Model endpoint not found: {model_name}")

        client = LLMClient(
            base_url=endpoint.base_url,
            api_key=endpoint.api_key,
            model_name=endpoint.model_name or endpoint.name,
            max_tokens=endpoint.max_tokens,
            temperature=endpoint.temperature,
            timeout=endpoint.timeout,
            service_tier=endpoint.service_tier,
            top_p=endpoint.top_p,
            min_p=endpoint.min_p,
            top_k=endpoint.top_k,
            presence_penalty=endpoint.presence_penalty,
            frequency_penalty=endpoint.frequency_penalty,
            repetition_penalty=endpoint.repetition_penalty,
        )
        self._clients[model_name] = client
        return client

    def get_vlm_client(self, model_name: str) -> VLMClient:
        """Get or create a VLM client."""
        if model_name in self._clients:
            client = self._clients[model_name]
            if isinstance(client, VLMClient):
                return client

        endpoint = self._endpoints.get(model_name)
        if not endpoint:
            raise ValueError(f"Model endpoint not found: {model_name}")

        client = VLMClient(
            base_url=endpoint.base_url,
            api_key=endpoint.api_key,
            model_name=endpoint.model_name or endpoint.name,
            max_tokens=endpoint.max_tokens,
            temperature=endpoint.temperature,
            timeout=endpoint.timeout,
            service_tier=endpoint.service_tier,
            top_p=endpoint.top_p,
            min_p=endpoint.min_p,
            top_k=endpoint.top_k,
            presence_penalty=endpoint.presence_penalty,
            frequency_penalty=endpoint.frequency_penalty,
            repetition_penalty=endpoint.repetition_penalty,
        )
        self._clients[model_name] = client
        return client

    def close_all(self):
        """Close all client connections."""
        for client in self._clients.values():
            client.close()
        self._clients.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close_all()
