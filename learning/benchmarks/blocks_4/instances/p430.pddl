;; blocks=4, percentage_new_tower=20, out_folder=., instance_id=430, seed=10

(define (problem blocksworld-430)
 (:domain blocksworld)
 (:objects b1 b2 b3 b4 - object)
 (:init 
    (arm-empty)
    (clear b3)
    (on-table b3)
    (clear b2)
    (on-table b2)
    (clear b1)
    (on b1 b4)
    (on-table b4))
 (:goal  (and 
    (clear b3)
    (on b3 b1)
    (on b1 b4)
    (on b4 b2)
    (on-table b2))))