;; blocks=1, percentage_new_tower=20, out_folder=., instance_id=68, seed=8

(define (problem blocksworld-68)
 (:domain blocksworld)
 (:objects b1 - object)
 (:init 
    (arm-empty)
    (clear b1)
    (on-table b1))
 (:goal  (and 
    (clear b1)
    (on-table b1))))
