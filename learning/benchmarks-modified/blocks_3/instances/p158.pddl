;; blocks=2, percentage_new_tower=10, out_folder=., instance_id=158, seed=8

(define (problem blocksworld-158)
 (:domain blocksworld)
 (:objects b1 b2 - object)
 (:init 
    (clear b2)
    (on-table b2)
    (clear b1)
    (on-table b1))
 (:goal  (and 
    (clear b2)
    (on b2 b1)
    (on-table b1))))
