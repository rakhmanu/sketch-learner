;; blocks=3, percentage_new_tower=20, out_folder=., instance_id=310, seed=10

(define (problem blocksworld-310)
 (:domain blocksworld)
 (:objects b1 b2 b3 - object)
 (:init 
    (arm-empty)
    (clear b2)
    (on b2 b1)
    (on b1 b3)
    (on-table b3))
 (:goal  (and 
    (clear b1)
    (on b1 b3)
    (on b3 b2)
    (on-table b2))))
