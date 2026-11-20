"""``Levenshtein.distance`` from rapidfuzz, the implementation the Levenshtein package wraps.

gepa-artifact's LiveBench-Math olympiad scorer imports ``from Levenshtein import distance``;
once that benchmark package is imported, :mod:`agents_gym_core.gepa_artifact` serves this module
under that name when the Levenshtein package is not installed.
"""

from rapidfuzz.distance.Levenshtein import distance

__all__ = ["distance"]
