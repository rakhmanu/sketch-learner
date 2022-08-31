from typing import List
from dataclasses import dataclass


@dataclass
class StatePair:
    source_idx: int
    target_idx: int
    distance: int

    def __eq__(self, other):
        return self.source_idx == other.source_idx and self.target_idx == other.target_idx and self.distance == other.distance

    def __hash__(self):
        return hash((self.source_idx, self.target_idx, self.distance))
