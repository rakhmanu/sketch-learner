;; blocks=5, percentage_new_tower=10, out_folder=., instance_id=176, seed=6

(define (problem blocksworld-176)
 (:domain blocksworld)
 (:objects b1 b2 b3 b4 b5 - object)
 (:init 
    (arm-empty)
    (clear b1)
    (on b1 b2)
    (on b2 b3)
    (on-table b3)
    (clear b5)
    (on b5 b4)
    (on-table b4))
 (:goal  (and 
    (clear b3)
    (on b3 b1)
    (on b1 b5)
    (on b5 b4)
    (on-table b4)
    (clear b2)
    (on-table b2))))