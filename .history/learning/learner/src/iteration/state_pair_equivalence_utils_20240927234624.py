import math
from collections import defaultdict
from typing import List, Dict, Set, Tuple

import dlplan.core as dlplan_core
import dlplan.policy as dlplan_policy

from .state_pair_equivalence import StatePairEquivalence
from .feature_pool import Feature
from .iteration_data import IterationData
from ..preprocessing import PreprocessingData

def get_state_pairs_from_equivalence(gfa_state_global_idx_to_state_pair_equivalence):
    """ 
    Retrieve and print state pairs from the StatePairEquivalence mapping.
    """
    state_pairs = []

    # Iterate over each state and its equivalence class
    for s_i, equivalence in gfa_state_global_idx_to_state_pair_equivalence.items():
        # For each equivalence class (r_idx), find the subgoal states
        for r_idx, subgoal_gfa_state_global_idxs in equivalence.r_idx_to_subgoal_gfa_state_global_idxs.items():
            # For each state pair (s_i, s_j) where s_j is a subgoal
            for s_j in subgoal_gfa_state_global_idxs:
                # Store the state pair (s_i, s_j)
                state_pairs.append((s_i, s_j))
                print(f"State pair: (s_i = {s_i}, s_j = {s_j})")

    return state_pairs

def create_inverse_mapping(gfa_state_global_idx_to_state_pair_equivalence: Dict[int, StatePairEquivalence]) -> Dict[int, Set[int]]:
    """
    Create a mapping from each state to the set of equivalence classes that include its inverse pairs.
    """
    inverse_mapping = defaultdict(set)

    # Populate the mapping
    for s_i, equivalence in gfa_state_global_idx_to_state_pair_equivalence.items():
        for r_idx, subgoal_gfa_state_global_idxs in equivalence.r_idx_to_subgoal_gfa_state_global_idxs.items():
            for s_j in subgoal_gfa_state_global_idxs:
                inverse_mapping[s_j].add(r_idx)  # Add r_idx to the inverse of s_j

    return inverse_mapping

def make_conditions(policy_builder: dlplan_policy.PolicyFactory,
                    feature_pool: List[Feature],
                    feature_valuations):
    """ Create conditions over all features that are satisfied in source_idx """
    conditions = set()
    for f_idx, (feature, val) in enumerate(zip(feature_pool, feature_valuations)):
        if isinstance(feature.dlplan_feature, dlplan_core.Boolean):
            if val:
                conditions.add(policy_builder.make_pos_condition(policy_builder.make_boolean(f"f{f_idx}", feature.dlplan_feature)))
            else:
                conditions.add(policy_builder.make_neg_condition(policy_builder.make_boolean(f"f{f_idx}", feature.dlplan_feature)))
        elif isinstance(feature.dlplan_feature, dlplan_core.Numerical):
            if val > 0:
                conditions.add(policy_builder.make_gt_condition(policy_builder.make_numerical(f"f{f_idx}", feature.dlplan_feature)))
            else:
                conditions.add(policy_builder.make_eq_condition(policy_builder.make_numerical(f"f{f_idx}", feature.dlplan_feature)))
    return conditions

def make_effects(policy_builder: dlplan_policy.PolicyFactory,
                 feature_pool: List[Feature],
                 source_feature_valuations,
                 target_feature_valuations):
    """ Create effects over all features that are satisfied in (source_idx, target_idx) """
    effects = set()
    for f_idx, (feature, source_val, target_val) in enumerate(zip(feature_pool, source_feature_valuations, target_feature_valuations)):
        if isinstance(feature.dlplan_feature, dlplan_core.Boolean):
            if source_val and not target_val:
                effects.add(policy_builder.make_neg_effect(policy_builder.make_boolean(f"f{f_idx}", feature.dlplan_feature)))
            elif not source_val and target_val:
                effects.add(policy_builder.make_pos_effect(policy_builder.make_boolean(f"f{f_idx}", feature.dlplan_feature)))
            else:
                effects.add(policy_builder.make_bot_effect(policy_builder.make_boolean(f"f{f_idx}", feature.dlplan_feature)))
        elif isinstance(feature.dlplan_feature, dlplan_core.Numerical):
            if source_val > target_val:
                effects.add(policy_builder.make_dec_effect(policy_builder.make_numerical(f"f{f_idx}", feature.dlplan_feature)))
            elif source_val < target_val:
                effects.add(policy_builder.make_inc_effect(policy_builder.make_numerical(f"f{f_idx}", feature.dlplan_feature)))
            else:
                effects.add(policy_builder.make_bot_effect(policy_builder.make_numerical(f"f{f_idx}", feature.dlplan_feature)))
    return effects

def compute_state_pair_equivalences(preprocessing_data: PreprocessingData,
                                     iteration_data: IterationData):
    # Initialize policy builder and data structures
    policy_builder = preprocessing_data.domain_data.policy_builder
    rules = []
    rule_repr_to_idx = dict()
    gfa_state_global_idx_to_state_pair_equivalence: Dict[int, StatePairEquivalence] = dict()

    for gfa_state in iteration_data.gfa_states:
        instance_idx = gfa_state.get_faithful_abstraction_index()
        instance_data = preprocessing_data.instance_datas[instance_idx]
        gfa_state_global_idx = gfa_state.get_global_index()

        if instance_data.gfa.is_deadend_state(gfa_state.get_faithful_abstract_state_index()):
            continue

        tuple_graph = preprocessing_data.gfa_state_global_idx_to_tuple_graph[gfa_state_global_idx]
        tuple_graph_states_by_distance = tuple_graph.get_states_grouped_by_distance()

        r_idx_to_distance = dict()
        r_idx_to_subgoal_gfa_state_global_idxs = defaultdict(set)
        subgoal_gfa_state_global_idx_to_r_idx = dict()

        # Add conditions
        conditions = make_conditions(policy_builder,
                                     iteration_data.feature_pool,
                                     iteration_data.gfa_state_global_idx_to_feature_evaluations[gfa_state_global_idx])

        for s_distance, mimir_ss_states_prime in enumerate(tuple_graph_states_by_distance):
            for mimir_ss_state_prime in mimir_ss_states_prime:
                gfa_state_prime = preprocessing_data.state_finder.get_gfa_state_from_ss_state_idx(instance_idx, instance_data.mimir_ss.get_state_index(mimir_ss_state_prime))
                gfa_state_prime_global_idx = gfa_state_prime.get_global_index()

                # Add effects
                effects = make_effects(policy_builder,
                                       iteration_data.feature_pool,
                                       iteration_data.gfa_state_global_idx_to_feature_evaluations[gfa_state_global_idx],
                                       iteration_data.gfa_state_global_idx_to_feature_evaluations[gfa_state_prime_global_idx])

                # Create rule and add it to the mapping
                rule = policy_builder.make_rule(conditions, effects)
                rule_repr = repr(rule)

                if rule_repr in rule_repr_to_idx:
                    r_idx = rule_repr_to_idx[rule_repr]
                else:
                    r_idx = len(rules)
                    rule_repr_to_idx[rule_repr] = r_idx
                    rules.append(rule)

                r_idx_to_distance[r_idx] = min(r_idx_to_distance.get(r_idx, math.inf), s_distance)
                r_idx_to_subgoal_gfa_state_global_idxs[r_idx].add(gfa_state_prime_global_idx)
                subgoal_gfa_state_global_idx_to_r_idx[gfa_state_prime_global_idx] = r_idx

        # Store equivalence data for this gfa_state
        gfa_state_global_idx_to_state_pair_equivalence[gfa_state_global_idx] = StatePairEquivalence(
            r_idx_to_subgoal_gfa_state_global_idxs,
            r_idx_to_distance,
            subgoal_gfa_state_global_idx_to_r_idx
        )
    get_state_pairs_from_equivalence(gfa_state_global_idx_to_state_pair_equivalence)
    # Create inverse mapping
    create_inverse_mapping(gfa_state_global_idx_to_state_pair_equivalence)

    return rules, gfa_state_global_idx_to_state_pair_equivalence



    # Idea to retrieve inverse state pairs in time that is linear in the number of state pair equivalence classes
    # for r_idx, subgoal_gfa_state_global_idxs in gfa_state_global_idx_to_state_pair_equivalence[s_i].r_idx_to_subgoal_gfa_state_global_idxs:
    #     if s_j in subgoal_gfa_state_global_idxs:
    #         # the state pair equivalence class r_idx contains a state pair [s_i,s_j]
    # (Can be optimized by precomputing a mapping f : S x S -> 2^X where S is the set of abstract states and X is the set of state pair equivalences,
    # such that f(s,s') is the set of state pair equivalence classes Y \subseteq X such that [s',s] is in y for all y in Y.)


