(define (problem grid-x1-y3-t1-k1-l1-p50)
(:domain grid)
(:objects 
        f0-0f 
        f0-1f 
        f0-2f 
        shape0 
        key0-0 
)
(:init
(arm-empty)
(place f0-0f)
(place f0-1f)
(place f0-2f)
(shape shape0)
(key key0-0)
(key-shape key0-0 shape0)
(conn f0-0f f0-1f)
(conn f0-1f f0-2f)
(conn f0-1f f0-0f)
(conn f0-2f f0-1f)
(open f0-0f)
(open f0-1f)
(locked f0-2f)
(lock-shape f0-2f shape0)
(at key0-0 f0-2f)
(at-robot f0-0f)
)
(:goal
(and
)
)
)
