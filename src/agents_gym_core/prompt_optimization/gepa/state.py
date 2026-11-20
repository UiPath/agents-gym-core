"""Optimization state — candidate pool + per-example score matrix.

The candidate with the highest mean training score is selected as the parent for the next
iteration. The state stores candidate ancestry, per-example training scores and validation
scores.

Adapted from ``gepa_artifact/gepa/gepa_utils.py:GEPAState``; candidates are plain dictionaries.
"""

from dataclasses import dataclass, field

# A Candidate maps component name -> instruction text.
# For basic loops there is exactly one component: ``system_prompt``.
Candidate = dict[str, str]


@dataclass
class OptimizationState:
    """All candidates seen so far + per-example train scores + val mean."""

    candidates: list[Candidate] = field(default_factory=list)
    parents: list[int | None] = field(default_factory=list)
    score_matrix: list[list[float]] = field(default_factory=list)
    val_scores: list[float | None] = field(default_factory=list)

    def add(
        self,
        candidate: Candidate,
        parent_idx: int | None,
        train_scores: list[float],
        val_score: float | None,
    ) -> int:
        idx = len(self.candidates)
        self.candidates.append(candidate)
        self.parents.append(parent_idx)
        self.score_matrix.append(list(train_scores))
        self.val_scores.append(val_score)
        return idx

    def train_mean(self, idx: int) -> float:
        scores = self.score_matrix[idx]
        return sum(scores) / len(scores) if scores else 0.0

    def best_idx(self) -> int:
        """Index of the candidate with highest mean train score (val tiebreak)."""
        best = 0
        best_key = (self.train_mean(0), self.val_scores[0] or 0.0)
        for i in range(1, len(self.candidates)):
            key = (self.train_mean(i), self.val_scores[i] or 0.0)
            if key > best_key:
                best_key = key
                best = i
        return best
