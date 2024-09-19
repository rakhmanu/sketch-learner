% Optimization for fewest number of rules, then smallest sum over feature complexities
minimize { C@2,complexity(F, C) : complexity(F, C), feature(F), select(F) }.

%%%%%%%%%% C1 %%%%%%%%%%
% Generate selected features
{ select(F) } :- feature(F).
% Generate rules
rule(1).

% Generate feature conditions and effects
{ c_n_eq(R, F); c_n_gt(R, F) } = 1 :- rule(R), numerical(F), select(F).
{ c_b_pos(R, F); c_b_neg(R, F) } = 1 :- rule(R), boolean(F), select(F).
{ e_n_dec(R, F); e_n_inc(R, F); e_n_bot(R, F) } = 1 :- rule(R), numerical(F), select(F).
{ e_b_pos(R, F); e_b_neg(R, F); e_b_bot(R, F) } = 1 :- rule(R), boolean(F), select(F).


% Define equivalence class C and rule R have same feature condition on F
c_satisfied(R, F, C) :- c_n_eq(R, F), feature_condition(C, F, "c_n_eq"), numerical(F),  rule(R), state_pair_class(C).
c_satisfied(R, F, C) :- c_n_gt(R, F), feature_condition(C, F, "c_n_gt"), numerical(F),  rule(R), state_pair_class(C).
c_satisfied(R, F, C) :- c_b_pos(R, F), feature_condition(C, F, "c_b_pos"), boolean(F),  rule(R), state_pair_class(C).
c_satisfied(R, F, C) :- c_b_neg(R, F), feature_condition(C, F, "c_b_neg"), boolean(F),  rule(R), state_pair_class(C).

e_satisfied(R, F, C) :- e_n_dec(R, F), feature_effect(C, F, "e_n_dec"), numerical(F),  rule(R), state_pair_class(C).
e_satisfied(R, F, C) :- e_n_inc(R, F), feature_effect(C, F, "e_n_inc"), numerical(F),  rule(R), state_pair_class(C).
e_satisfied(R, F, C) :- e_b_pos(R, F), feature_effect(C, F, "e_b_pos"), boolean(F), rule(R), state_pair_class(C).
e_satisfied(R, F, C) :- e_b_neg(R, F), feature_effect(C, F, "e_b_neg"), boolean(F),  rule(R), state_pair_class(C).
e_satisfied(R, F, C) :- e_b_bot(R, F), feature_effect(C, F, "e_b_bot"), boolean(F),  rule(R), state_pair_class(C).
e_satisfied(R, F, C) :- e_n_bot(R, F), feature_effect(C, F, "e_n_bot"), numerical(F),  rule(R), state_pair_class(C).
e_satisfied(R, F, C) :- e_b_pos(R, F), feature_effect(C, F, "e_b_bot"), feature_condition(C, F, "c_b_pos"), boolean(F),  rule(R), state_pair_class(C).
e_satisfied(R, F, C) :- e_b_neg(R, F), feature_effect(C, F, "e_b_bot"), feature_condition(C, F, "c_b_neg"), boolean(F),  rule(R), state_pair_class(C).

%%%%%%%%%% C2 %%%%%%%%%%

%%% Require that all equivalence classes C underlying a tuple T of subproblem P[S] are good, effectively bounding the width of P[S]
:- not sat_rule(R, C), selected_tuple(S, T), rule(R), contain(S, T, C).

%%%%%%%%%% C3 %%%%%%%%%%

%%% Unsolvable states can be addressed with d_distance
:- D <= D', t_distance(S, T, D), selected_tuple(S, T), sat_rule(R, C), d_distance(S, C, D'), initial(S), unsolvable(S), nongoal(S).

%%%%%%%%%% C4 %%%%%%%%%%

% (Optimal-width): Require solvable states S' closer than subgoal to not be assigned to any rule.
:- D < D', r_distance(S, C, D), sat_rule(R, C), selected_tuple(S, T), t_distance(S, T, D'), state(S).

%%%%%%%%%% C5 %%%%%%%%%%

% Define state pair equivalences that are compatible with rules.
sat_rule(R, C) :- { feature(F) : not c_satisfied(R, F, C), select(F);
     feature(F) : not e_satisfied(R, F, C), select(F) } = 0, rule(R), state_pair_class(C).

% python3 learning/main.py --domain_filepath learning/benchmarks/gripper/domain.pddl --problems_directory learning/benchmarks/gripper/instances_debug --width 1 --workspace workspace --additional_numericals "n_count(r_primitive(carry,0,1))" "n_count(r_diff(r_primitive(at_g,0,1), r_primitive(at,0,1)))" --disable_feature_generation
% python3 learning/main.py --domain_filepath learning/benchmarks/gripper/domain.pddl --problems_directory learning/benchmarks/gripper/instances --width 1 --workspace workspace 

% Display
#show rule/1.
#show select/1.

#show c_n_eq/2.
#show c_n_gt/2.

#show e_n_dec/2.
#show e_n_inc/2.
#show e_n_bot/2.

#show c_b_pos/2.
#show c_b_neg/2.

#show e_b_pos/2.
#show e_b_neg/2.
#show e_b_bot/2.
