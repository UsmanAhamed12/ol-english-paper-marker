"""Concrete OCR provider adapters."""

from app.ocr.providers.ollama import OllamaOCRProvider
from app.ocr.providers.tesseract import TesseractOCRProvider

__all__ = ["OllamaOCRProvider", "TesseractOCRProvider"]
