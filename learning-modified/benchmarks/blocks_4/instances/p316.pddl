;; blocks=3, percentage_new_tower=20, out_folder=., instance_id=316, seed=16

(define (problem blocksworld-316)
 (:domain blocksworld)
 (:objects b1 b2 b3 - object)
 (:init 
    (arm-empty)
    (clear b3)
    (on b3 b2)
    (on b2 b1)
    (on-table b1))
 (:goal  (and 
    (clear b1)
    (on b1 b2)
    (on b2 b3)
    (on-table b3))))
