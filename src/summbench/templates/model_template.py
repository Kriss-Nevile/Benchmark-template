from __future__ import annotations

from summbench.models.base import SummarizationModel


class CustomModelTemplate(SummarizationModel):
    """
    Copy this class when you want to benchmark a new model.

    You only need to edit two parts:
    1. load()
    2. generate_summary()
    """

    def __init__(self) -> None:
        super().__init__(name="my-new-model")
        self.model = None

    def load(self) -> None:
        """
        Put your model loading code here.

        Examples:
        - load a local Hugging Face model
        - connect to an API client
        - load a custom class from another file
        """
        self.model = "replace-this-with-your-real-model"

    def generate_summary(self, source: str, temperature: float = 0.0) -> str:
        """
        Replace this fake summary with a real model call.
        The function must return a single summary string.
        """
        del temperature

        if self.model is None:
            self.load()

        preview = source[:180].replace("\n", " ").strip()
        return f"[Replace this with your real model output] {preview}"

