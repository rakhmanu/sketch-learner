;; blocks=4, percentage_new_tower=0, out_folder=., instance_id=388, seed=28

(define (problem blocksworld-388)
 (:domain blocksworld)
 (:objects b1 b2 b3 b4 - object)
 (:init 
    (arm-empty)
    (clear b3)
    (on b3 b1)
    (on b1 b4)
    (on b4 b2)
    (on-table b2))
 (:goal  (and 
    (clear b1)
    (on b1 b3)
    (on b3 b4)
    (on b4 b2)
    (on-table b2))))
