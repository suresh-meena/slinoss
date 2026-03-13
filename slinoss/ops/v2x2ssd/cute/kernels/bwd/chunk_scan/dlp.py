"""CuTe backward ``dlp`` entrypoint for the ``v2x2ssd`` chunk-scan stage.

This class is the dedicated host-dispatch surface for the stable DLP-producing
kernel implementation used by chunk-scan backward.
"""

from __future__ import annotations

from .dcdr import ChunkScanBwdDCDRAmpere


class ChunkScanBwdDLPAmpere(ChunkScanBwdDCDRAmpere):
    """Dedicated DLP-dispatch class for chunk-scan backward on Ampere."""


__all__ = ["ChunkScanBwdDLPAmpere"]
