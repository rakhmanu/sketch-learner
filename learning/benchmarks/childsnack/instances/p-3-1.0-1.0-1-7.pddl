; child-snack task with 3 children and 1.0 gluten factor 
; constant factor of 1.0
; random seed: 7

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child1 child2 child3 - child
    bread1 bread2 bread3 - bread-portion
    content1 content2 content3 - content-portion
    tray1 - tray
    table1 table2 table3 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray1 kitchen)
     (at_kitchen_bread bread1)
     (at_kitchen_bread bread2)
     (at_kitchen_bread bread3)
     (at_kitchen_content content1)
     (at_kitchen_content content2)
     (at_kitchen_content content3)
     (no_gluten_bread bread2)
     (no_gluten_bread bread1)
     (no_gluten_bread bread3)
     (no_gluten_content content3)
     (no_gluten_content content1)
     (no_gluten_content content2)
     (allergic_gluten child2)
     (allergic_gluten child3)
     (allergic_gluten child1)
     (waiting child1 table3)
     (waiting child2 table1)
     (waiting child3 table3)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child1)
     (served child2)
     (served child3)
    )
  )
)
