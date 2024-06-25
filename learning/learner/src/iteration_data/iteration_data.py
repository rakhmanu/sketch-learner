from dataclasses import dataclass
from typing import List, Dict, Union

import pymimir as mm
import dlplan.policy as dlplan_policy

from ..instance_data.instance_data import InstanceData
from ..iteration_data.feature_pool import Feature
from ..iteration_data.state_pair_equivalence import StatePairEquivalence
from ..iteration_data.tuple_graph_equivalence import TupleGraphEquivalence

@dataclass
class IterationData:
    """ Store data related to a domain. """
    # Changes in each iterations
    instance_datas: List[InstanceData] = None

    gfa_states: List[mm.GlobalFaithfulAbstractState] = None

    feature_pool: List[Feature] = None
    gfa_state_id_to_feature_evaluations: Dict[int, List[Union[bool, int]]] = None

    state_pair_equivalences: List[dlplan_policy.Rule] = None
    gfa_state_id_to_state_pair_equivalence: Dict[int, StatePairEquivalence] = None

    gfa_state_id_to_tuple_graph_equivalence: Dict[int, TupleGraphEquivalence] = None