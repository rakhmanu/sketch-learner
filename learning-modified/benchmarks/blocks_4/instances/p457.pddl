;; blocks=4, percentage_new_tower=40, out_folder=., instance_id=457, seed=7

(define (problem blocksworld-457)
 (:domain blocksworld)
 (:objects b1 b2 b3 b4 - object)
 (:init 
    (arm-empty)
    (clear b1)
    (on-table b1)
    (clear b3)
    (on b3 b4)
    (on-table b4)
    (clear b2)
    (on-table b2))
 (:goal  (and 
    (clear b3)
    (on-table b3)
    (clear b1)
    (on b1 b2)
    (on-table b2)
    (clear b4)
    (on-table b4))))
