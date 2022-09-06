from enum import Enum
from typing import Dict, Tuple, List, MutableSet

from .state_pair import StatePair


class StatePairClassification(Enum):
    DELTA_OPTIMAL = 0
    NOT_DELTA_OPTIMAL = 1


class StatePairClassifier:
    """ """
    def __init__(self, delta, state_pair_to_classification: Dict[StatePair, StatePairClassification], state_pair_to_distance: Dict[StatePair, int], source_idx_to_state_pairs: Dict[int, MutableSet[StatePair]], expanded_s_idxs: List[int], generated_s_idxs: List[int]):
        self.delta = delta
        self.state_pair_to_classification = state_pair_to_classification
        self.state_pair_to_distance = state_pair_to_distance
        self.source_idx_to_state_pairs = source_idx_to_state_pairs
        self.expanded_s_idxs = expanded_s_idxs  # DELTA_OPTIMAL reachable with outgoing DELTA_OPTIMAL state pair
        self.generated_s_idxs = generated_s_idxs  # DELTA_OPTIMAL reachable

    def classify(self, state_pair: StatePair):
        return self.state_pair_to_classification[state_pair]

    def print(self):
        print("StatePairClassifier:")
        print("    state_pair_to_classification:", self.state_pair_to_classification)
        print("    source_idx_to_state_pairs:", self.source_idx_to_state_pairs)
        print("    expanded_s_idxs:", self.expanded_s_idxs)
        print("    generated_s_idxs:", self.generated_s_idxs)
