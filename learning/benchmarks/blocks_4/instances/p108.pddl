;; blocks=3, percentage_new_tower=20, out_folder=., instance_id=108, seed=8

(define (problem blocksworld-108)
 (:domain blocksworld)
 (:objects b1 b2 b3 - object)
 (:init 
    (arm-empty)
    (clear b3)
    (on-table b3)
    (clear b1)
    (on b1 b2)
    (on-table b2))
 (:goal  (and 
    (clear b3)
    (on-table b3)
    (clear b2)
    (on b2 b1)
    (on-table b1))))