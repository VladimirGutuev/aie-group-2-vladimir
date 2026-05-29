from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    host: str = "0.0.0.0"
    port: int = 8000

    ocr_langs: str = Field(default="en,ru")
    use_detector: bool = False
    detector_weights: str = "yolov8n.pt"
    min_confidence: float = 0.3
    use_gpu: bool = False

    @property
    def ocr_lang_list(self) -> list[str]:
        return [lang.strip() for lang in self.ocr_langs.split(",") if lang.strip()]


settings = Settings()
