;; cars=3, locations=3, seed=6, instance_id=156, out_folder=.

(define (problem ferry-156)
 (:domain ferry)
 (:objects 
    car1 car2 car3 - car
    loc1 loc2 loc3 - location
 )
 (:init 
    (empty-ferry)
    (at-ferry loc3)
    (at car1 loc3)
    (at car2 loc1)
    (at car3 loc1)
)
 (:goal  (and (at car1 loc2)
   (at car2 loc3)
   (at car3 loc2))))
