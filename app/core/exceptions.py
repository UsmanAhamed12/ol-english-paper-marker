"""Base exceptions shared across the application."""


class ApplicationError(Exception):
    """Base class for expected application failures."""


class ConfigurationError(ApplicationError):
    """Raised when runtime configuration cannot support an operation."""


class InvalidPDFError(ApplicationError):
    """Raised when an input is not a readable, supported PDF."""


class PDFTooLargeError(InvalidPDFError):
    """Raised when a PDF exceeds the configured byte limit."""


class PDFPageLimitError(InvalidPDFError):
    """Raised when a PDF exceeds the configured page-count limit."""


class PDFRenderingError(ApplicationError):
    """Raised when validated PDF pages cannot be rendered safely."""


class OCRProcessingError(ApplicationError):
    """Raised when OCR cannot produce a valid page or document result."""


class OCRProviderError(OCRProcessingError):
    """Raised when an OCR provider fails or violates its contract."""


class OCRBenchmarkPreparationError(ApplicationError):
    """Raised when private OCR benchmark samples cannot be prepared safely."""
