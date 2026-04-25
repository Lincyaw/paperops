"""IR layer exports."""

from paperops.slides.ir.node import Node
from paperops.slides.ir.schema import (
    IR_DOCUMENT_SCHEMA,
    IR_NODE_SCHEMA,
    STYLE_KEY_SCHEMAS,
    validate_document,
    validate_node,
)
from paperops.slides.ir.validator import (
    StructuredValidationError,
    ValidationReport,
    validate_ir_document,
)

__all__ = [
    "IR_DOCUMENT_SCHEMA",
    "IR_NODE_SCHEMA",
    "Node",
    "STYLE_KEY_SCHEMAS",
    "StructuredValidationError",
    "ValidationReport",
    "validate_document",
    "validate_ir_document",
    "validate_node",
]
