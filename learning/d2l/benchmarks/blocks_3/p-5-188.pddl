

(define (problem BW-rand-5)
(:domain blocksworld)
(:objects b1 b2 b3 b4 b5  - block)
(:init
(on b1 b3)
(on b2 b5)
(on b3 b4)
(on-table b4)
(on b5 b1)
(clear b2)
)
(:goal
(and
(on b2 b4)
(on b3 b1)
(on b5 b3))
)
)

