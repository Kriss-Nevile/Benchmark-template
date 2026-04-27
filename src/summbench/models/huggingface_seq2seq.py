from __future__ import annotations

from typing import Any

from .base import SummarizationModel


class HuggingFaceSeq2SeqSummarizer(SummarizationModel):
    """Generic Hugging Face sequence-to-sequence LM adapter."""

    def __init__(
        self,
        model_name: str,
        max_input_tokens: int = 1024,
        max_new_tokens: int = 1024,
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
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "transformers and torch are required. Run: pip install -e .[huggingface]"
            ) from exc

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        model_kwargs: dict[str, Any] = {"device_map": self.device_map}
        if self.load_in_4bit:
            model_kwargs["load_in_4bit"] = True

        if self.torch_dtype:
            model_kwargs["torch_dtype"] = getattr(torch, self.torch_dtype)

        model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name, **model_kwargs)
        self._tokenizer = tokenizer
        self._model = model

    def generate_summary(self, source: str, temperature: float = 0.7) -> str:
        self.load()
        tokenizer = self._tokenizer
        model = self._model
        assert tokenizer is not None and model is not None

        # Format input based on the model's pre-training/fine-tuning requirements
        if "vit5" in self.model_name.lower():
            # Vit-5 summarization models expect the 'vietnews: ' prefix
            prompt = f"vietnews: {source} </s>"
        elif "bartpho" in self.model_name.lower() and "word" in self.model_name.lower():
            # BARTpho-word expects word-segmented input (using pyvi or VnCoreNLP)
            try:
                from pyvi import ViTokenizer
                prompt = ViTokenizer.tokenize(source)
            except ImportError as exc:
                raise ImportError(
                    "The 'pyvi' package is required for BARTpho word segmentation. "
                    "Run: pip install pyvi"
                ) from exc
        else:
            prompt = source

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
            )

        # Seq2Seq output only contains the generated tokens, unlike causal LM
        generated_tokens = outputs[0]
        return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    def close(self) -> None:
        self._model = None
        self._tokenizer = None
