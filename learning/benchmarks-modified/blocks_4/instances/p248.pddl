;; blocks=3, percentage_new_tower=0, out_folder=., instance_id=248, seed=8

(define (problem blocksworld-248)
 (:domain blocksworld)
 (:objects b1 b2 b3 - object)
 (:init 
    (arm-empty)
    (clear b1)
    (on b1 b3)
    (on b3 b2)
    (on-table b2))
 (:goal  (and 
    (clear b3)
    (on b3 b1)
    (on b1 b2)
    (on-table b2))))
