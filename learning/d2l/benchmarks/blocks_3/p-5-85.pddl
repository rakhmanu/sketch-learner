

(define (problem BW-rand-5)
(:domain blocksworld)
(:objects b1 b2 b3 b4 b5  - block)
(:init
(on b1 b5)
(on-table b2)
(on-table b3)
(on-table b4)
(on b5 b4)
(clear b1)
(clear b2)
(clear b3)
)
(:goal
(and
(on b1 b4)
(on b3 b2)
(on b4 b5)
(on b5 b3))
)
)

