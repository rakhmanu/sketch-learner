import logging
from collections import defaultdict
from pathlib import Path
from termcolor import colored
from typing import List, MutableSet, Dict
import clingo
import pymimir as mm
from dlplan.policy import PolicyMinimizer
from collections import defaultdict
from .src.exit_codes import ExitCode
from .src.iteration import EncodingType, ASPFactory, ClingoExitCode, IterationData, LearningStatistics, Sketch, D2sepDlplanPolicyFactory, ExplicitDlplanPolicyFactory, compute_feature_pool, compute_per_state_feature_valuations, compute_state_pair_equivalences, compute_tuple_graph_equivalences, minimize_tuple_graph_equivalences
from .src.util import Timer, create_experiment_workspace, change_working_directory, write_file, change_dir, memory_usage, add_console_handler, print_separation_line
from .src.preprocessing import InstanceData, PreprocessingData, StateFinder, compute_instance_datas, compute_tuple_graphs

def unsolvable_states_from_solution(symbols):
    unsolvable_states = set()
    for symbol in symbols:
        if isinstance(symbol, clingo.Symbol) and symbol.name == "unsolvable":
            if len(symbol.arguments) >= 2:
                state_id = int(symbol.arguments[0].number)
                instance_id = int(symbol.arguments[1].number)
                unsolvable_states.add((state_id, instance_id))
    return unsolvable_states

def learn_sketch_for_problem_class(
    domain_filepath: Path,
    problems_directory: Path,
    workspace: Path,
    width: int,
    disable_closed_Q: bool = True,
    max_num_states_per_instance: int = 2000,
    max_time_per_instance: int = 10,
    encoding_type: EncodingType = EncodingType.EXPLICIT,
    max_num_rules: int = 1,
    enable_goal_separating_features: bool = True,
    disable_feature_generation: bool = True,
    enable_incomplete_feature_pruning: bool = False,
    concept_complexity_limit: int = 9,
    role_complexity_limit: int = 9,
    boolean_complexity_limit: int = 9,
    count_numerical_complexity_limit: int = 9,
    distance_numerical_complexity_limit: int = 9,
    feature_limit: int = 1000000,
    additional_booleans: List[str] = None,
    additional_numericals: List[str] = None,
    enable_dump_files: bool = False,
):
    # Initialize variables and setup
    if additional_booleans is None:
        additional_booleans = []
    if additional_numericals is None:
        additional_numericals = []
    instance_filepaths = list(problems_directory.iterdir())
    add_console_handler(logging.getLogger(), logging.INFO)
    create_experiment_workspace(workspace)
    change_working_directory(workspace)

    total_timer = Timer()
    preprocessing_timer = Timer()
    asp_timer = Timer(stopped=True)
    verification_timer = Timer(stopped=True)
    iteration_data = IterationData()

    # Generate data
    with change_dir("input"):
        logging.info(colored("Constructing InstanceDatas...", "blue", "on_grey"))
        domain_data, instance_datas, num_ss_states, num_gfa_states = compute_instance_datas(domain_filepath, instance_filepaths, disable_closed_Q, max_num_states_per_instance, max_time_per_instance, enable_dump_files)
        logging.info(colored("..done", "blue", "on_grey"))
        if instance_datas is None:
            raise Exception("Failed to create InstanceDatas.")

        state_finder = StateFinder(domain_data, instance_datas)

        logging.info(colored("Initializing TupleGraphs...", "blue", "on_grey"))
        gfa_state_id_to_tuple_graph: Dict[int, mm.TupleGraph] = compute_tuple_graphs(domain_data, instance_datas, state_finder, width, enable_dump_files)
        logging.info(colored("..done", "blue", "on_grey"))

    preprocessing_data = PreprocessingData(domain_data, instance_datas, state_finder, gfa_state_id_to_tuple_graph)
    preprocessing_timer.stop()

    if not preprocessing_data.instance_datas:
        raise RuntimeError("Data is empty")

    # Learn sketch
    if encoding_type == EncodingType.EXPLICIT:
        create_experiment_workspace(workspace)
        preprocessing_timer.resume()

        sketches = set()

        for instance_data in preprocessing_data.instance_datas:
            # TODO: when is the best time to generate features?
            # Based on current instance currently looks most reasonable.
            # It could be restricted to the states in the tuple graph as well,
            # which reduces the number of rules, but not fully sure yet about the consequences in the bigger picture.
            iteration_data = IterationData()
            iteration_data.instance_datas = [instance_data]
            iteration_data.gfa_states = instance_data.gfa.get_states()

            # Report some progress
            write_file(f"{instance_data.idx}.dot", instance_data.dlplan_ss.to_dot(1))
            print("     ", end="")
            print("id:", instance_data.idx,
                    "problem_filepath:", instance_data.mimir_ss.get_problem().get_filepath(),
                    "num_states:", instance_data.mimir_ss.get_num_states(),
                    "num_state_equivalences:", instance_data.gfa.get_num_states())

            logging.info(colored("Initializing DomainFeatureData...", "blue", "on_grey"))
            iteration_data.feature_pool = compute_feature_pool(
                preprocessing_data,
                iteration_data,
                gfa_state_id_to_tuple_graph,
                state_finder,
                disable_feature_generation,
                enable_incomplete_feature_pruning,
                concept_complexity_limit,
                role_complexity_limit,
                boolean_complexity_limit,
                count_numerical_complexity_limit,
                distance_numerical_complexity_limit,
                feature_limit,
                additional_booleans,
                additional_numericals
            )

            logging.info(colored("Constructing PerStateFeatureValuations...", "blue", "on_grey"))
            iteration_data.gfa_state_global_idx_to_feature_evaluations = compute_per_state_feature_valuations(preprocessing_data, iteration_data)
            logging.info(colored("..done", "blue", "on_grey"))

            logging.info(colored("Constructing StatePairEquivalenceDatas...", "blue", "on_grey"))
            iteration_data.state_pair_equivalences, iteration_data.gfa_state_global_idx_to_state_pair_equivalence = compute_state_pair_equivalences(preprocessing_data, iteration_data)
            logging.info(colored("..done", "blue", "on_grey"))

            logging.info(colored("Constructing TupleGraphEquivalences...", "blue", "on_grey"))
            iteration_data.gfa_state_global_idx_to_tuple_graph_equivalence = compute_tuple_graph_equivalences(preprocessing_data, iteration_data)
            logging.info(colored("..done", "blue", "on_grey"))

            logging.info(colored("Minimizing TupleGraphEquivalences...", "blue", "on_grey"))
            minimize_tuple_graph_equivalences(preprocessing_data, iteration_data)
            logging.info(colored("..done", "blue", "on_grey"))
            preprocessing_timer.stop()
            asp_timer.resume()

            # We count the number of subgoal tuples for which no rule could be found.
            # This usually happens if the pool of features is not sufficiently rich.
            count_unsat_tuples = 0

            for gfa_state in iteration_data.gfa_states:
                gfa_state_global_idx = gfa_state.get_global_index()
                print(f"Keys in gfa_state_global_idx_to_tuple_graph: {list(preprocessing_data.gfa_state_global_idx_to_tuple_graph.keys())}")

                print(f"gfa_state_global_idx: {gfa_state_global_idx}")
                tuple_graph = preprocessing_data.gfa_state_global_idx_to_tuple_graph[gfa_state_global_idx][1:]

                for distance, group in enumerate(tuple_graph.get_vertices_grouped_by_distance()):
                    if distance == 0:
                        # We skip subgoal tuples at distance zero because they do not encode progress towards a goal.
                        continue
                    for vertex in group:
                        # Here we find all simplest single sketch rules for a pair (state, subgoal tuple).
                        t_idx = vertex.get_index()

                        asp_factory = ASPFactory(encoding_type, enable_goal_separating_features, max_num_rules)
                        facts = asp_factory.make_facts(preprocessing_data, iteration_data)
                        # The create_selected_tuple_fact creates a fact selected_tuple(s,t).
                        # This allows access to the seed state, as well as the tuple
                        facts.append(asp_factory.create_selected_tuple_fact(gfa_state_global_idx, t_idx))
                        asp_factory.ground(facts)
                        # TODO: we currently only return one of the optimal solutions since I updated the code of the ASP factory.
                        symbolss, returncode = asp_factory.solve_all_opt()
                        #symbolss = [symbols,]

                        if returncode in [ClingoExitCode.UNSATISFIABLE, ClingoExitCode.EXHAUSTED]:
                            print(colored("ASP is unsatisfiable or exhausted!", "red", "on_grey"))
                            print(colored(f"No sketch of width {width} exists that solves all instances!", "red", "on_grey"))
                            # There might be tuples where we get unsat.
                            # If such cases occur, which is very likely in complex domains, then we must look closer into this.
                            count_unsat_tuples += 1
                            continue
                        elif returncode == ClingoExitCode.UNKNOWN:
                            print(colored("ASP solving throws unknown error!", "red", "on_grey"))
                            exit(ExitCode.UNKNOWN)
                        elif returncode == ClingoExitCode.INTERRUPTED:
                            print(colored("ASP solving interrupted!", "red", "on_grey"))
                            exit(ExitCode.INTERRUPTED)
                        else:
                            logging.info(f"Solution found with return code: {returncode}")
                        
                        for symbols in symbolss:
                            dlplan_policy = ExplicitDlplanPolicyFactory().make_dlplan_policy_from_answer_set(symbols, preprocessing_data, iteration_data)
                            sketch = Sketch(dlplan_policy, width)
                            sketches.add(sketch)
                            print(dlplan_policy)

    else:
        raise Exception("No implementation for the given encoding type.")

    # Output the result
    with change_dir("output"):
        print_separation_line()
        logging.info(colored("Summary:", "green", "on_grey"))

        print(f"Preprocessing time: {int(preprocessing_timer.get_elapsed_sec()) + 1} seconds.")
        print(f"ASP time: {int(asp_timer.get_elapsed_sec()) + 1} seconds.")
        print(f"Verification time: {int(verification_timer.get_elapsed_sec()) + 1} seconds.")
        print(f"Total time: {int(total_timer.get_elapsed_sec()) + 1} seconds.")
        print(f"Total memory: {int(memory_usage() / 1024)} GiB.")
        print(f"Total number of states: {num_ss_states}")
        print(f"Total number of abstract states: {num_gfa_states}")
        print(f"Number of unsat tuples: {count_unsat_tuples}")
        print(f"Number of sketch rules: {len(sketches)}")
        
        

