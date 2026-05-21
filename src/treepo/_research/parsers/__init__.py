"""Parser utilities for modality-specific data ingestion."""

from .pdf_text import ParsedPDFDocument, PDFTextParser
from .router import (
    DEFAULT_ROUTER_ACTIONS,
    PARSER_ROUTER_CONTRACT_VERSION,
    ParserRouter,
    ParserRouterConfig,
    normalize_parser_action_name,
)

__all__ = [
    "ParsedPDFDocument",
    "PDFTextParser",
    "ParserRouter",
    "ParserRouterConfig",
    "DEFAULT_ROUTER_ACTIONS",
    "PARSER_ROUTER_CONTRACT_VERSION",
    "normalize_parser_action_name",
]
