(define (problem p01)
  (:domain ulzhal)
(:objects 
      down - direction 
      left - direction 
      right - direction 
      up - direction 
      robot - thing 
      block01 - thing 
      pos0 - location
      pos1 - location
      pos2 - location
)
 (:init
   (at robot pos2)
   (at block01 pos0)
   (clear pos1)

   (is-goal pos1)
   (is-nongoal pos0)
   (is-nongoal pos2)

   (is-agent robot)
   (is-block block01)
   (move-dir pos0 pos1 down)
   (move-dir pos1 pos0 up)
   (move-dir pos1 pos2 down)
   (move-dir pos2 pos1 up)
 )
 (:goal (and (at-goal block01) ))
)