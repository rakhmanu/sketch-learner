;; blocks=2, percentage_new_tower=20, out_folder=., instance_id=200, seed=20

(define (problem blocksworld-200)
 (:domain blocksworld)
 (:objects b1 b2 - object)
 (:init 
    (clear b2)
    (on b2 b1)
    (on-table b1))
 (:goal  (and 
    (clear b2)
    (on b2 b1)
    (on-table b1))))
