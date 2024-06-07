#!/usr/bin/env python
import sys
import random
import argparse
from typing import Dict
import os


def get_block(b: int) -> str:
    return f"b{b}"


def get_objects(blocks: int, **kwargs: dict) -> str:
    return " ".join([get_block(i) for i in range(1, 1 + blocks)]) + " - object"


def get_state(blocks: int, percentage_new_tower: int, is_goal: bool = False, **kwargs: dict) -> str:
    offset = "\n    "
    str_state = ""
    # In blocks three we do atomic move actions
    #if not is_goal:
    #    str_state += offset + "(arm-empty)"

    vblocks = list(range(1, blocks + 1))
    random.shuffle(vblocks)

    str_state += offset + f"(clear {get_block(vblocks[0])})"
    for i in range(0, len(vblocks)-1):
        if random.randint(0, 99) < percentage_new_tower:  # chance of building a new tower
            str_state += offset + f"(on-table {get_block(vblocks[i])})"
            str_state += offset + f"(clear {get_block(vblocks[i+1])})"
        else:
            str_state += offset + f"(on {get_block(vblocks[i])} {get_block(vblocks[i+1])})"
    str_state += offset + f"(on-table {get_block(vblocks[-1])})"
    return str_state


def get_init(blocks: int, percentage_new_tower: int, **kwargs: Dict) -> str:
    return get_state(blocks=blocks, percentage_new_tower=percentage_new_tower)


def get_goal(blocks: int, percentage_new_tower: int, **kwargs: Dict) -> str:
    return " (and " + get_state(blocks=blocks, percentage_new_tower=percentage_new_tower, is_goal=True) + ")"


def generate_problem(args: Dict):
    str_config = ', '.join([f'{k}={v}' for k, v in args.items()])
    str_objects = get_objects(**args)
    str_init = get_init(**args)
    str_goal = get_goal(**args)
    with open(f"{args['out_folder']}/p{args['instance_id']:02}.pddl", "w") as f_problem:
        f_problem.write(
            f";; {str_config}\n\n"
            f"(define (problem blocksworld-{args['instance_id']:02})\n"
            f" (:domain blocksworld)\n"
            f" (:objects {str_objects})\n"
            f" (:init {str_init})\n"
            f" (:goal {str_goal}))\n")


def parse_args() -> Dict[str, int]:
    # Parser descriptor
    parser = argparse.ArgumentParser(description="Blocksworld generator")
    parser.add_argument("-b", "--blocks", type=int, help="number of boxes (min 2)", required=True)
    parser.add_argument("-t", "--percentage_new_tower", type=int, default=10, help="Percentage to start building a new tower [0,100]", required=True)
    parser.add_argument("--seed", type=int, default=42, help="random seed (default: 42)")
    parser.add_argument("-out", "--out_folder", type=str, default=".", help="output folder (default: \".\")")
    parser.add_argument("-id", "--instance_id", type=int, default=0, help="instance id (default: 0)")

    # Parse arguments
    args = parser.parse_args()
    blocks = args.blocks
    out_f = args.out_folder
    ins_id = args.instance_id

    # Input sanity checks
    if blocks < 2:
        sys.exit(-1)

    # Initialize data
    random.seed(args.seed)  # set the random seed here
    os.makedirs(name=out_f, exist_ok=True)  # create the output folder if that doesn't exist

    return {'blocks': blocks, 'out_folder': out_f, 'instance_id': ins_id, 'seed': args.seed}


def main():
    num_blocks = [1,2,3,4,5]
    percentage_new_towers = [0,10,20,40]
    num_seeds = 30
    instance_id = 0
    for num_block in num_blocks:
        for percentage_new_tower in percentage_new_towers:
            for seed in range(num_seeds):
                # args_dict = parse_args()
                args_dict = {"blocks": num_block, "percentage_new_tower": percentage_new_tower, "out_folder": ".", "instance_id": instance_id, "seed": seed}
                generate_problem(args=args_dict)
                instance_id += 1


if __name__ == "__main__":
    main()

