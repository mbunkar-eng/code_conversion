"""
Tokenizer Service - Handles tokenization for LLM inference.
"""

import logging
from typing import Optional, Union
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class TokenizationResult:
    """Result of tokenization operation."""
    tokens: list[int]
    token_count: int
    text: Optional[str] = None


@dataclass
class ChatMessage:
    """Chat message structure."""
    role: str
    content: str


class TokenizerService:
    """
    Service for tokenizing text and managing chat templates.

    Provides token counting, text encoding/decoding, and
    chat template application for different models.
    """

    def __init__(
        self,
        model_path: str,
        trust_remote_code: bool = True
    ):
        """
        Initialize tokenizer service.

        Args:
            model_path: Path to model or HuggingFace repo
            trust_remote_code: Trust remote code in tokenizer
        """
        self.model_path = model_path
        self.trust_remote_code = trust_remote_code
        self._tokenizer = None
        self._initialized = False

    def initialize(self) -> None:
        """Initialize the tokenizer."""
        if self._initialized:
            return

        try:
            from transformers import AutoTokenizer

            logger.info(f"Loading tokenizer from: {self.model_path}")

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=self.trust_remote_code
            )

            # Ensure pad token is set
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            self._initialized = True
            logger.info("Tokenizer initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize tokenizer: {e}")
            raise

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        truncation: bool = False
    ) -> TokenizationResult:
        """
        Encode text to token IDs.

        Args:
            text: Input text to tokenize
            add_special_tokens: Add model-specific special tokens
            max_length: Maximum sequence length
            truncation: Truncate to max_length

        Returns:
            TokenizationResult with token IDs and count
        """
        if not self._initialized:
            self.initialize()

        kwargs = {
            "add_special_tokens": add_special_tokens,
            "return_tensors": None
        }

        if max_length is not None:
            kwargs["max_length"] = max_length
            kwargs["truncation"] = truncation

        tokens = self._tokenizer.encode(text, **kwargs)

        return TokenizationResult(
            tokens=tokens,
            token_count=len(tokens),
            text=text
        )

    def decode(
        self,
        tokens: list[int],
        skip_special_tokens: bool = True
    ) -> str:
        """
        Decode token IDs to text.

        Args:
            tokens: Token IDs to decode
            skip_special_tokens: Skip special tokens in output

        Returns:
            Decoded text
        """
        if not self._initialized:
            self.initialize()

        return self._tokenizer.decode(
            tokens,
            skip_special_tokens=skip_special_tokens
        )

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.

        Args:
            text: Input text

        Returns:
            Token count
        """
        result = self.encode(text, add_special_tokens=False)
        return result.token_count

    def apply_chat_template(
        self,
        messages: list[ChatMessage],
        add_generation_prompt: bool = True,
        tokenize: bool = False
    ) -> Union[str, list[int]]:
        """
        Apply chat template to messages.

        Args:
            messages: List of chat messages
            add_generation_prompt: Add prompt for assistant response
            tokenize: Return tokens instead of text

        Returns:
            Formatted prompt string or token IDs
        """
        if not self._initialized:
            self.initialize()

        # Convert to dict format expected by tokenizer
        messages_dict = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]

        if hasattr(self._tokenizer, "apply_chat_template"):
            return self._tokenizer.apply_chat_template(
                messages_dict,
                add_generation_prompt=add_generation_prompt,
                tokenize=tokenize
            )
        else:
            # Fallback for tokenizers without chat template
            return self._format_chat_fallback(messages, add_generation_prompt)

    def _format_chat_fallback(
        self,
        messages: list[ChatMessage],
        add_generation_prompt: bool
    ) -> str:
        """Fallback chat formatting for models without templates."""
        formatted = []

        for msg in messages:
            if msg.role == "system":
                formatted.append(f"System: {msg.content}")
            elif msg.role == "user":
                formatted.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                formatted.append(f"Assistant: {msg.content}")

        result = "\n\n".join(formatted)

        if add_generation_prompt:
            result += "\n\nAssistant:"

        return result

    def truncate_to_max_tokens(
        self,
        text: str,
        max_tokens: int,
        from_end: bool = False
    ) -> str:
        """
        Truncate text to maximum token count.

        Args:
            text: Input text
            max_tokens: Maximum tokens to keep
            from_end: Truncate from end instead of start

        Returns:
            Truncated text
        """
        tokens = self.encode(text, add_special_tokens=False).tokens

        if len(tokens) <= max_tokens:
            return text

        if from_end:
            truncated_tokens = tokens[-max_tokens:]
        else:
            truncated_tokens = tokens[:max_tokens]

        return self.decode(truncated_tokens)

    def get_special_tokens(self) -> dict:
        """Get special token information."""
        if not self._initialized:
            self.initialize()

        return {
            "bos_token": self._tokenizer.bos_token,
            "eos_token": self._tokenizer.eos_token,
            "pad_token": self._tokenizer.pad_token,
            "unk_token": self._tokenizer.unk_token,
            "bos_token_id": self._tokenizer.bos_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
            "pad_token_id": self._tokenizer.pad_token_id,
            "vocab_size": self._tokenizer.vocab_size
        }

    def estimate_cost(
        self,
        text: str,
        cost_per_1k_tokens: float = 0.002
    ) -> dict:
        """
        Estimate processing cost for text.

        Args:
            text: Input text
            cost_per_1k_tokens: Cost per 1000 tokens

        Returns:
            Cost estimation details
        """
        token_count = self.count_tokens(text)
        cost = (token_count / 1000) * cost_per_1k_tokens

        return {
            "token_count": token_count,
            "cost_per_1k": cost_per_1k_tokens,
            "estimated_cost": round(cost, 6)
        }


class MockTokenizerService(TokenizerService):
    """Mock tokenizer service for testing without model files."""

    def initialize(self) -> None:
        """Initialize mock tokenizer."""
        self._initialized = True
        logger.info("Mock tokenizer initialized")

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        truncation: bool = False
    ) -> TokenizationResult:
        """Mock encode - approximate 4 chars per token."""
        tokens = list(range(len(text) // 4 + 1))

        if max_length and truncation:
            tokens = tokens[:max_length]

        return TokenizationResult(
            tokens=tokens,
            token_count=len(tokens),
            text=text
        )

    def decode(
        self,
        tokens: list[int],
        skip_special_tokens: bool = True
    ) -> str:
        """Mock decode."""
        return f"[decoded {len(tokens)} tokens]"

    def apply_chat_template(
        self,
        messages: list[ChatMessage],
        add_generation_prompt: bool = True,
        tokenize: bool = False
    ) -> Union[str, list[int]]:
        """Apply fallback chat template."""
        result = self._format_chat_fallback(messages, add_generation_prompt)

        if tokenize:
            return self.encode(result).tokens
        return result

    def get_special_tokens(self) -> dict:
        """Return mock special tokens."""
        return {
            "bos_token": "<s>",
            "eos_token": "</s>",
            "pad_token": "<pad>",
            "unk_token": "<unk>",
            "bos_token_id": 1,
            "eos_token_id": 2,
            "pad_token_id": 0,
            "vocab_size": 32000
        }
