;; blocks=3, percentage_new_tower=20, out_folder=., instance_id=319, seed=19

(define (problem blocksworld-319)
 (:domain blocksworld)
 (:objects b1 b2 b3 - object)
 (:init 
    (arm-empty)
    (clear b2)
    (on-table b2)
    (clear b1)
    (on-table b1)
    (clear b3)
    (on-table b3))
 (:goal  (and 
    (clear b2)
    (on b2 b3)
    (on b3 b1)
    (on-table b1))))
