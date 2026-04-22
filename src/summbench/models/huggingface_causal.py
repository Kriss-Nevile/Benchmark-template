from __future__ import annotations

from typing import Any

from .base import SummarizationModel


class HuggingFaceCausalSummarizer(SummarizationModel):
    """Generic Hugging Face causal LM adapter."""

    def __init__(
        self,
        model_name: str,
        max_input_tokens: int = 4096,
        max_new_tokens: int = 256,
        torch_dtype: str | None = None,
        device_map: str = "auto",
        load_in_4bit: bool = False,
        top_p: float = 0.9,
    ) -> None:
        super().__init__(name=model_name)
        self.model_name = model_name
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens
        self.torch_dtype = torch_dtype
        self.device_map = device_map
        self.load_in_4bit = load_in_4bit
        self.top_p = top_p
        self._model = None
        self._tokenizer = None

    def load(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "transformers and torch are required. Run: pip install -e .[huggingface]"
            ) from exc

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model_kwargs: dict[str, Any] = {"device_map": self.device_map}
        if self.load_in_4bit:
            model_kwargs["load_in_4bit"] = True

        if self.torch_dtype:
            model_kwargs["torch_dtype"] = getattr(torch, self.torch_dtype)

        model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
        self._tokenizer = tokenizer
        self._model = model

    def generate_summary(self, source: str, temperature: float = 0.7) -> str:
        self.load()
        tokenizer = self._tokenizer
        model = self._model
        assert tokenizer is not None and model is not None

        prompt_messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": source},
        ]

        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            prompt = tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = f"{self.system_prompt}\n\nSource article:\n{source}\n\nSummary:"

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        )

        inputs = {key: value.to(model.device) for key, value in inputs.items()}

        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "transformers and torch are required. Run: pip install -e .[huggingface]"
            ) from exc

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                top_p=self.top_p,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        generated_tokens = outputs[0][inputs["input_ids"].shape[1] :]
        return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    def close(self) -> None:
        self._model = None
        self._tokenizer = None

