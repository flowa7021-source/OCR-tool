"""Post-OCR structured-field parsers (orchestrator layer).

Each submodule wraps a parser implementation (``src.tn_parser`` for
транспортные накладные / УПД; future ones for invoices / contracts)
behind a uniform signature the :class:`~src.application.pipeline.
OCRPipeline` dispatches to based on
:class:`~src.core.models.ExtractConfig.kind`.

Orchestrators live in the application layer — they pull domain
objects from :mod:`src.core.models`, call into parser packages
(``src.tn_parser``), and handle all I/O / error conversion so the
pipeline never sees a parser-specific exception.
"""
