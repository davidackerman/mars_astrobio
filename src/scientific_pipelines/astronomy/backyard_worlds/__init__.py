"""Backyard Worlds brown dwarf detection pipeline."""

from .brown_dwarf_scorer import BrownDwarfScorer
from .downloader import BackyardWorldsDownloader
from .pipeline import BackyardWorldsPipeline
from .sequence_encoder import FlipbookSequenceEncoder

__all__ = [
    "BackyardWorldsPipeline",
    "BackyardWorldsDownloader",
    "FlipbookSequenceEncoder",
    "BrownDwarfScorer",
]
