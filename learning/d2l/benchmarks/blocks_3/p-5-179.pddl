

(define (problem BW-rand-5)
(:domain blocksworld)
(:objects b1 b2 b3 b4 b5  - block)
(:init
(on-table b1)
(on b2 b1)
(on b3 b4)
(on-table b4)
(on b5 b3)
(clear b2)
(clear b5)
)
(:goal
(and
(on b2 b1)
(on b4 b3)
(on b5 b2))
)
)

