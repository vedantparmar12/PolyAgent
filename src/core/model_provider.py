"""Multi-model provider system supporting OpenRouter and direct providers"""

from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
from pydantic import BaseModel, Field
import os
from dataclasses import dataclass
import httpx


class ModelProviderEnum(str, Enum):
    """Supported model providers"""
    OPENROUTER = "openrouter"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    MISTRAL = "mistral"
    GROQ = "groq"
    TOGETHER = "together"
    REPLICATE = "replicate"
    HUGGINGFACE = "huggingface"
    LOCAL = "local"


class ModelCapability(str, Enum):
    """Model capabilities"""
    CHAT = "chat"
    COMPLETION = "completion"
    EMBEDDING = "embedding"
    IMAGE_GENERATION = "image_generation"
    VISION = "vision"
    FUNCTION_CALLING = "function_calling"
    JSON_MODE = "json_mode"
    STREAMING = "streaming"


@dataclass
class ModelInfo:
    """Information about a specific model"""
    id: str
    name: str
    provider: ModelProviderEnum
    capabilities: List[ModelCapability]
    context_window: int
    max_output: int
    pricing: Dict[str, float]  # per 1k tokens
    speed_rank: int  # 1-10, 10 being fastest
    quality_rank: int  # 1-10, 10 being best
    description: str
    tags: List[str]


class ModelRegistry:
    """Registry of all available models"""
    
    def __init__(self):
        self._models: Dict[str, ModelInfo] = {}
        self._register_default_models()
    
    def _register_default_models(self):
        """Register default models from various providers"""
        
        # OpenAI Models
        self.register(ModelInfo(
            id="openai/gpt-4",
            name="GPT-4",
            provider=ModelProviderEnum.OPENAI,
            capabilities=[
                ModelCapability.CHAT,
                ModelCapability.FUNCTION_CALLING,
                ModelCapability.JSON_MODE
            ],
            context_window=8192,
            max_output=4096,
            pricing={"input": 0.03, "output": 0.06},
            speed_rank=6,
            quality_rank=9,
            description="OpenAI's most capable model for complex tasks",
            tags=["general", "coding", "reasoning"]
        ))
        
        self.register(ModelInfo(
            id="openai/gpt-4-turbo",
            name="GPT-4 Turbo",
            provider=ModelProviderEnum.OPENAI,
            capabilities=[
                ModelCapability.CHAT,
                ModelCapability.FUNCTION_CALLING,
                ModelCapability.JSON_MODE,
                ModelCapability.VISION
            ],
            context_window=128000,
            max_output=4096,
            pricing={"input": 0.01, "output": 0.03},
            speed_rank=7,
            quality_rank=9,
            description="GPT-4 Turbo with vision, JSON mode, and 128k context",
            tags=["general", "coding", "vision", "long-context"]
        ))
        
        self.register(ModelInfo(
            id="openai/gpt-3.5-turbo",
            name="GPT-3.5 Turbo",
            provider=ModelProviderEnum.OPENAI,
            capabilities=[
                ModelCapability.CHAT,
                ModelCapability.FUNCTION_CALLING,
                ModelCapability.JSON_MODE
            ],
            context_window=16384,
            max_output=4096,
            pricing={"input": 0.0005, "output": 0.0015},
            speed_rank=9,
            quality_rank=7,
            description="Fast and capable model for most tasks",
            tags=["general", "fast", "affordable"]
        ))
        
        # Anthropic Models
        self.register(ModelInfo(
            id="anthropic/claude-3.7-sonnet",
            name="Claude 3.7 Sonnet",
            provider=ModelProviderEnum.ANTHROPIC,
            capabilities=[
                ModelCapability.CHAT,
                ModelCapability.VISION,
                ModelCapability.FUNCTION_CALLING,
                ModelCapability.JSON_MODE
            ],
            context_window=200000,
            max_output=8192,
            pricing={"input": 0.003, "output": 0.015},
            speed_rank=8,
            quality_rank=9,
            description="Latest Claude model with enhanced capabilities and reasoning",
            tags=["general", "coding", "analysis", "long-context", "latest"]
        ))
        
        self.register(ModelInfo(
            id="anthropic/claude-3-opus",
            name="Claude 3 Opus",
            provider=ModelProviderEnum.ANTHROPIC,
            capabilities=[
                ModelCapability.CHAT,
                ModelCapability.VISION,
                ModelCapability.FUNCTION_CALLING
            ],
            context_window=200000,
            max_output=4096,
            pricing={"input": 0.015, "output": 0.075},
            speed_rank=5,
            quality_rank=10,
            description="Most intelligent Claude model, excels at complex analysis",
            tags=["general", "coding", "analysis", "long-context"]
        ))
        
        self.register(ModelInfo(
            id="anthropic/claude-3-sonnet",
            name="Claude 3 Sonnet",
            provider=ModelProviderEnum.ANTHROPIC,
            capabilities=[
                ModelCapability.CHAT,
                ModelCapability.VISION,
                ModelCapability.FUNCTION_CALLING
            ],
            context_window=200000,
            max_output=4096,
            pricing={"input": 0.003, "output": 0.015},
            speed_rank=7,
            quality_rank=8,
            description="Balanced Claude model for most tasks",
            tags=["general", "coding", "balanced"]
        ))
        
        self.register(ModelInfo(
            id="anthropic/claude-3-haiku",
            name="Claude 3 Haiku",
            provider=ModelProviderEnum.ANTHROPIC,
            capabilities=[
                ModelCapability.CHAT,
                ModelCapability.VISION
            ],
            context_window=200000,
            max_output=4096,
            pricing={"input": 0.00025, "output": 0.00125},
            speed_rank=10,
            quality_rank=6,
            description="Fastest Claude model for simple tasks",
            tags=["fast", "affordable", "simple-tasks"]
        ))
        
        # Google Models
        self.register(ModelInfo(
            id="google/gemini-pro",
            name="Gemini Pro",
            provider=ModelProviderEnum.GOOGLE,
            capabilities=[
                ModelCapability.CHAT,
                ModelCapability.FUNCTION_CALLING,
                ModelCapability.VISION
            ],
            context_window=32768,
            max_output=8192,
            pricing={"input": 0.0005, "output": 0.0015},
            speed_rank=8,
            quality_rank=8,
            description="Google's advanced model with multimodal capabilities",
            tags=["general", "vision", "multimodal"]
        ))
        
        # Mistral Models
        self.register(ModelInfo(
            id="mistralai/mixtral-8x7b-instruct",
            name="Mixtral 8x7B",
            provider=ModelProviderEnum.MISTRAL,
            capabilities=[
                ModelCapability.CHAT,
                ModelCapability.FUNCTION_CALLING
            ],
            context_window=32768,
            max_output=4096,
            pricing={"input": 0.0007, "output": 0.0007},
            speed_rank=8,
            quality_rank=7,
            description="Mistral's MoE model with strong performance",
            tags=["open-source", "coding", "multilingual"]
        ))
        
        # Groq Models (via OpenRouter)
        self.register(ModelInfo(
            id="groq/mixtral-8x7b-32768",
            name="Mixtral 8x7B (Groq)",
            provider=ModelProviderEnum.GROQ,
            capabilities=[
                ModelCapability.CHAT,
                ModelCapability.STREAMING
            ],
            context_window=32768,
            max_output=4096,
            pricing={"input": 0.00027, "output": 0.00027},
            speed_rank=10,
            quality_rank=7,
            description="Ultra-fast inference via Groq LPU",
            tags=["ultra-fast", "streaming", "affordable"]
        ))
        
        # Meta Models (via OpenRouter)
        self.register(ModelInfo(
            id="meta-llama/llama-3-70b-instruct",
            name="Llama 3 70B",
            provider=ModelProviderEnum.OPENROUTER,
            capabilities=[
                ModelCapability.CHAT,
                ModelCapability.FUNCTION_CALLING
            ],
            context_window=8192,
            max_output=4096,
            pricing={"input": 0.0008, "output": 0.0008},
            speed_rank=6,
            quality_rank=8,
            description="Meta's latest open model with strong capabilities",
            tags=["open-source", "general", "coding"]
        ))
    
    def register(self, model_info: ModelInfo):
        """Register a new model"""
        self._models[model_info.id] = model_info
    
    def get_model(self, model_id: str) -> Optional[ModelInfo]:
        """Get model information by ID"""
        return self._models.get(model_id)
    
    def list_models(
        self,
        provider: Optional[ModelProviderEnum] = None,
        capability: Optional[ModelCapability] = None,
        max_price: Optional[float] = None,
        min_speed: Optional[int] = None,
        min_quality: Optional[int] = None,
        tags: Optional[List[str]] = None
    ) -> List[ModelInfo]:
        """List models with optional filters"""
        models = list(self._models.values())
        
        if provider:
            models = [m for m in models if m.provider == provider]
        
        if capability:
            models = [m for m in models if capability in m.capabilities]
        
        if max_price is not None:
            models = [m for m in models if m.pricing.get("input", float("inf")) <= max_price]
        
        if min_speed is not None:
            models = [m for m in models if m.speed_rank >= min_speed]
        
        if min_quality is not None:
            models = [m for m in models if m.quality_rank >= min_quality]
        
        if tags:
            models = [m for m in models if any(tag in m.tags for tag in tags)]
        
        return models
    
    def recommend_model(
        self,
        task_type: str,
        context_size: int = 0,
        budget_priority: bool = False,
        speed_priority: bool = False
    ) -> ModelInfo:
        """Recommend a model based on requirements"""
        suitable_models = self.list_models()
        
        # Filter by context size if needed
        if context_size > 0:
            suitable_models = [m for m in suitable_models if m.context_window >= context_size]
        
        # Score models based on priorities
        for model in suitable_models:
            score = 0
            
            # Base quality score
            score += model.quality_rank * 2
            
            # Task-specific scoring
            if task_type == "coding" and "coding" in model.tags:
                score += 5
            elif task_type == "analysis" and "analysis" in model.tags:
                score += 5
            elif task_type == "vision" and ModelCapability.VISION in model.capabilities:
                score += 10
            
            # Priority scoring
            if budget_priority:
                # Lower price is better
                avg_price = (model.pricing.get("input", 0) + model.pricing.get("output", 0)) / 2
                score += (1 / (avg_price + 0.001)) * 5
            
            if speed_priority:
                score += model.speed_rank
            
            model.recommendation_score = score
        
        # Sort by score and return best
        suitable_models.sort(key=lambda m: m.recommendation_score, reverse=True)
        return suitable_models[0] if suitable_models else None


class ModelProvider:
    """Unified interface for multiple model providers"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize model provider
        
        Args:
            config: Provider configuration including API keys
        """
        self.config = config or {}
        self.registry = ModelRegistry()
        self._providers = {}
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize provider connections"""
        # OpenRouter
        if self.config.get("openrouter_api_key") or os.getenv("OPENROUTER_API_KEY"):
            self._providers[ModelProviderEnum.OPENROUTER] = OpenRouterProvider(
                api_key=self.config.get("openrouter_api_key") or os.getenv("OPENROUTER_API_KEY")
            )
        
        # Direct providers
        if self.config.get("openai_api_key") or os.getenv("OPENAI_API_KEY"):
            self._providers[ModelProviderEnum.OPENAI] = DirectProvider(
                provider=ModelProviderEnum.OPENAI,
                api_key=self.config.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
            )
        
        if self.config.get("anthropic_api_key") or os.getenv("ANTHROPIC_API_KEY"):
            self._providers[ModelProviderEnum.ANTHROPIC] = DirectProvider(
                provider=ModelProviderEnum.ANTHROPIC,
                api_key=self.config.get("anthropic_api_key") or os.getenv("ANTHROPIC_API_KEY")
            )
    
    def get_model_string(self, model_id: str) -> str:
        """Get the model string for Pydantic AI
        
        Args:
            model_id: Model identifier (e.g., "openai/gpt-4", "anthropic/claude-3-opus")
            
        Returns:
            Model string for Pydantic AI
        """
        # Check if it's an OpenRouter model
        if "/" in model_id and not model_id.startswith(("openai/", "anthropic/", "google/")):
            # Use OpenRouter for models like meta-llama/*, groq/*, etc.
            return f"openrouter:{model_id}"
        
        # Direct provider models
        if model_id.startswith("openai/"):
            return model_id.replace("openai/", "openai:")
        elif model_id.startswith("anthropic/"):
            return model_id.replace("anthropic/", "anthropic:")
        elif model_id.startswith("google/"):
            return model_id.replace("google/", "google:")
        
        # Default to model ID as-is
        return model_id
    
    def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        """Get information about a model"""
        return self.registry.get_model(model_id)
    
    def list_available_models(self, **filters) -> List[ModelInfo]:
        """List all available models with optional filters"""
        return self.registry.list_models(**filters)
    
    def recommend_model(self, **criteria) -> ModelInfo:
        """Get model recommendation based on criteria"""
        return self.registry.recommend_model(**criteria)
    
    def estimate_cost(
        self,
        model_id: str,
        input_tokens: int,
        output_tokens: int
    ) -> Dict[str, float]:
        """Estimate cost for a model usage"""
        model = self.registry.get_model(model_id)
        if not model:
            return {"error": "Model not found"}
        
        input_cost = (input_tokens / 1000) * model.pricing.get("input", 0)
        output_cost = (output_tokens / 1000) * model.pricing.get("output", 0)
        
        return {
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": input_cost + output_cost,
            "currency": "USD"
        }


class OpenRouterProvider:
    """OpenRouter-specific provider implementation"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1"
        self.http_client = httpx.AsyncClient(
            headers={
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": "https://github.com/enhanced-agentic-workflow",
                "X-Title": "Enhanced Agentic Workflow"
            }
        )
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """List all models available on OpenRouter"""
        response = await self.http_client.get(f"{self.base_url}/models")
        response.raise_for_status()
        return response.json()["data"]
    
    async def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific model"""
        models = await self.list_models()
        for model in models:
            if model["id"] == model_id:
                return model
        return None


class DirectProvider:
    """Direct provider implementation for OpenAI, Anthropic, etc."""
    
    def __init__(self, provider: ModelProviderEnum, api_key: str):
        self.provider = provider
        self.api_key = api_key
        
        # Set up provider-specific configuration
        if provider == ModelProviderEnum.OPENAI:
            self.base_url = "https://api.openai.com/v1"
        elif provider == ModelProviderEnum.ANTHROPIC:
            self.base_url = "https://api.anthropic.com/v1"
        # Add more providers as needed


# Configuration class for the entire system
class ModelConfig(BaseModel):
    """Configuration for model provider system"""
    default_model: str = Field(default="openai/gpt-4", description="Default model to use")
    fallback_models: List[str] = Field(
        default=["openai/gpt-3.5-turbo", "anthropic/claude-3-haiku"],
        description="Fallback models if primary fails"
    )
    enable_streaming: bool = Field(default=True, description="Enable streaming responses")
    enable_caching: bool = Field(default=True, description="Enable response caching")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    timeout_seconds: int = Field(default=300, description="Request timeout")
    
    # Provider API keys (can also be set via environment variables)
    openrouter_api_key: Optional[str] = Field(default=None, description="OpenRouter API key")
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    anthropic_api_key: Optional[str] = Field(default=None, description="Anthropic API key")
    google_api_key: Optional[str] = Field(default=None, description="Google API key")
    
    # Cost management
    max_cost_per_request: Optional[float] = Field(default=None, description="Maximum cost per request in USD")
    daily_cost_limit: Optional[float] = Field(default=None, description="Daily cost limit in USD")
    
    # Model selection preferences
    prefer_open_source: bool = Field(default=False, description="Prefer open source models")
    prefer_fast_models: bool = Field(default=False, description="Prefer faster models")
    prefer_cheap_models: bool = Field(default=False, description="Prefer cheaper models")