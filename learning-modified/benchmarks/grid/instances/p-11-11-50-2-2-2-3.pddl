(define (problem grid-x2-y2-t2-k11-l11-p50)
(:domain grid)
(:objects 
        f0-0f f1-0f 
        f0-1f f1-1f 
        shape0 shape1 
        key0-0 
        key1-0 
)
(:init
(arm-empty)
(place f0-0f)
(place f1-0f)
(place f0-1f)
(place f1-1f)
(shape shape0)
(shape shape1)
(key key0-0)
(key-shape key0-0 shape0)
(key key1-0)
(key-shape key1-0 shape1)
(conn f0-0f f1-0f)
(conn f0-1f f1-1f)
(conn f0-0f f0-1f)
(conn f1-0f f1-1f)
(conn f1-0f f0-0f)
(conn f1-1f f0-1f)
(conn f0-1f f0-0f)
(conn f1-1f f1-0f)
(open f1-0f)
(open f1-1f)
(locked f0-1f)
(lock-shape f0-1f shape0)
(locked f0-0f)
(lock-shape f0-0f shape1)
(at key0-0 f1-0f)
(at key1-0 f0-0f)
(at-robot f1-1f)
)
(:goal
(and
(at key0-0 f0-0f)
(at key1-0 f1-1f)
)
)
)
