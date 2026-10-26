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


class ImagePreprocessingError(OCRProcessingError):
    """Raised when a derived OCR image cannot be produced safely."""


class StructureDetectionError(ApplicationError):
    """Raised when exam structure cannot be derived or rendered safely."""


class EvidenceSeparationError(ApplicationError):
    """Raised when page evidence cannot be classified or rendered safely."""


class AnswerExtractionError(ApplicationError):
    """Raised when safe evidence cannot be converted into answer provenance."""


class MarkingSchemeIngestionError(ApplicationError):
    """Raised when an official marking scheme cannot be ingested safely."""


class MarkingSchemeParseError(MarkingSchemeIngestionError):
    """Raised when marking-scheme OCR has no reliable Test structure."""


class KnowledgeBaseIngestionError(ApplicationError):
    """Raised when a reference document cannot be indexed safely."""


class KnowledgeBaseRetrievalError(ApplicationError):
    """Raised when local reference material cannot be retrieved safely."""


class MarkingEngineError(ApplicationError):
    """Raised when deterministic marking cannot preserve score safety."""


class SubjectiveMarkingProviderError(MarkingEngineError):
    """Raised when the local subjective-marking provider cannot respond safely."""


class SubjectiveMarkingTimeoutError(SubjectiveMarkingProviderError):
    """Raised when local subjective inference exceeds its configured deadline."""


class OrchestrationError(ApplicationError):
    """Raised when the top-level marking workflow cannot preserve safety."""


class OCRBenchmarkPreparationError(ApplicationError):
    """Raised when private OCR benchmark samples cannot be prepared safely."""
