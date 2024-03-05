from typing import List

from dlplan.novelty import NoveltyBase, TupleGraph

from .instance_data import InstanceData
from .tuple_graph import PerStateTupleGraphs

from ..util.command import change_dir, write_file


def compute_tuple_graphs(width: int, instance_datas: List[InstanceData], enable_dump_files: bool):
    with change_dir("tuple_graphs"):
        for instance_data in instance_datas:
            name = instance_data.instance_filepath.stem
            with change_dir(name):
                per_state_tuple_graphs = PerStateTupleGraphs()
                novelty_base = NoveltyBase(len(instance_data.state_space.get_instance_info().get_atoms()), width)
                for s_idx in instance_data.state_space.get_states().keys():
                    if instance_data.is_deadend(s_idx):
                        continue

                    with change_dir(str(s_idx)):
                        tuple_graph = TupleGraph(novelty_base, instance_data.state_space, s_idx, True)
                        per_state_tuple_graphs.s_idx_to_tuple_graph[s_idx] = tuple_graph

                        if enable_dump_files:
                            write_file(f"{tuple_graph.get_root_state_index()}.dot", tuple_graph.to_dot(1))

                instance_data.per_state_tuple_graphs = per_state_tuple_graphs
