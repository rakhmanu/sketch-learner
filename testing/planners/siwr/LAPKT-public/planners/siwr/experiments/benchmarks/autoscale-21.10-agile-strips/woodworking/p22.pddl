; woodworking task with 99 parts and 125% wood
; Machines:
;   3 highspeed-saw
;   3 glazer
;   3 grinder
;   3 immersion-varnisher
;   3 planer
;   3 saw
;   3 spray-varnisher

(define (problem wood-prob)
  (:domain woodworking)
  (:objects
    highspeed-saw0 highspeed-saw1 highspeed-saw2 - highspeed-saw
    glazer0 glazer1 glazer2 - glazer
    grinder0 grinder1 grinder2 - grinder
    immersion-varnisher0 immersion-varnisher1 immersion-varnisher2 - immersion-varnisher
    planer0 planer1 planer2 - planer
    saw0 saw1 saw2 - saw
    spray-varnisher0 spray-varnisher1 spray-varnisher2 - spray-varnisher
    green blue red mauve white black - acolour
    teak mahogany cherry pine oak beech walnut - awood
    p0 p1 p2 p3 p4 p5 p6 p7 p8 p9 p10 p11 p12 p13 p14 p15 p16 p17 p18 p19 p20 p21 p22 p23 p24 p25 p26 p27 p28 p29 p30 p31 p32 p33 p34 p35 p36 p37 p38 p39 p40 p41 p42 p43 p44 p45 p46 p47 p48 p49 p50 p51 p52 p53 p54 p55 p56 p57 p58 p59 p60 p61 p62 p63 p64 p65 p66 p67 p68 p69 p70 p71 p72 p73 p74 p75 p76 p77 p78 p79 p80 p81 p82 p83 p84 p85 p86 p87 p88 p89 p90 p91 p92 p93 p94 p95 p96 p97 p98 - part
    b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13 b14 b15 b16 b17 b18 b19 b20 b21 b22 b23 b24 b25 b26 - board
    s0 s1 s2 s3 s4 s5 s6 s7 s8 s9 s10 s11 s12 - aboardsize
  )
  (:init
    (grind-treatment-change varnished colourfragments)
    (grind-treatment-change glazed untreated)
    (grind-treatment-change untreated untreated)
    (grind-treatment-change colourfragments untreated)
    (is-smooth smooth)
    (is-smooth verysmooth)
    (= (total-cost) 0)
    (boardsize-successor s0 s1)
    (boardsize-successor s1 s2)
    (boardsize-successor s2 s3)
    (boardsize-successor s3 s4)
    (boardsize-successor s4 s5)
    (boardsize-successor s5 s6)
    (boardsize-successor s6 s7)
    (boardsize-successor s7 s8)
    (boardsize-successor s8 s9)
    (boardsize-successor s9 s10)
    (boardsize-successor s10 s11)
    (boardsize-successor s11 s12)
    (empty highspeed-saw0)
    (empty highspeed-saw1)
    (empty highspeed-saw2)
    (has-colour glazer0 mauve)
    (has-colour glazer0 green)
    (has-colour glazer0 blue)
    (has-colour glazer0 white)
    (has-colour glazer0 natural)
    (has-colour glazer1 mauve)
    (has-colour glazer1 blue)
    (has-colour glazer1 natural)
    (has-colour glazer1 black)
    (has-colour glazer1 red)
    (has-colour glazer2 mauve)
    (has-colour glazer2 green)
    (has-colour glazer2 white)
    (has-colour glazer2 natural)
    (has-colour glazer2 black)
    (has-colour glazer2 red)
    (has-colour immersion-varnisher0 natural)
    (has-colour immersion-varnisher0 blue)
    (has-colour immersion-varnisher0 white)
    (has-colour immersion-varnisher0 red)
    (has-colour immersion-varnisher1 mauve)
    (has-colour immersion-varnisher1 blue)
    (has-colour immersion-varnisher1 green)
    (has-colour immersion-varnisher1 natural)
    (has-colour immersion-varnisher1 black)
    (has-colour immersion-varnisher2 mauve)
    (has-colour immersion-varnisher2 blue)
    (has-colour immersion-varnisher2 green)
    (has-colour immersion-varnisher2 white)
    (has-colour immersion-varnisher2 black)
    (has-colour immersion-varnisher2 red)
    (has-colour spray-varnisher0 blue)
    (has-colour spray-varnisher0 green)
    (has-colour spray-varnisher0 white)
    (has-colour spray-varnisher0 natural)
    (has-colour spray-varnisher0 black)
    (has-colour spray-varnisher0 red)
    (has-colour spray-varnisher1 mauve)
    (has-colour spray-varnisher1 blue)
    (has-colour spray-varnisher1 white)
    (has-colour spray-varnisher1 black)
    (has-colour spray-varnisher1 red)
    (has-colour spray-varnisher2 mauve)
    (has-colour spray-varnisher2 green)
    (has-colour spray-varnisher2 white)
    (has-colour spray-varnisher2 natural)
    (has-colour spray-varnisher2 black)
    (unused p0)
    (goalsize p0 large)
    (= (spray-varnish-cost p0) 15)
    (= (glaze-cost p0) 20)
    (= (grind-cost p0) 45)
    (= (plane-cost p0) 30)
    (unused p1)
    (goalsize p1 large)
    (= (spray-varnish-cost p1) 15)
    (= (glaze-cost p1) 20)
    (= (grind-cost p1) 45)
    (= (plane-cost p1) 30)
    (unused p2)
    (goalsize p2 large)
    (= (spray-varnish-cost p2) 15)
    (= (glaze-cost p2) 20)
    (= (grind-cost p2) 45)
    (= (plane-cost p2) 30)
    (unused p3)
    (goalsize p3 medium)
    (= (spray-varnish-cost p3) 10)
    (= (glaze-cost p3) 15)
    (= (grind-cost p3) 30)
    (= (plane-cost p3) 20)
    (unused p4)
    (goalsize p4 medium)
    (= (spray-varnish-cost p4) 10)
    (= (glaze-cost p4) 15)
    (= (grind-cost p4) 30)
    (= (plane-cost p4) 20)
    (unused p5)
    (goalsize p5 medium)
    (= (spray-varnish-cost p5) 10)
    (= (glaze-cost p5) 15)
    (= (grind-cost p5) 30)
    (= (plane-cost p5) 20)
    (unused p6)
    (goalsize p6 small)
    (= (spray-varnish-cost p6) 5)
    (= (glaze-cost p6) 10)
    (= (grind-cost p6) 15)
    (= (plane-cost p6) 10)
    (unused p7)
    (goalsize p7 medium)
    (= (spray-varnish-cost p7) 10)
    (= (glaze-cost p7) 15)
    (= (grind-cost p7) 30)
    (= (plane-cost p7) 20)
    (unused p8)
    (goalsize p8 small)
    (= (spray-varnish-cost p8) 5)
    (= (glaze-cost p8) 10)
    (= (grind-cost p8) 15)
    (= (plane-cost p8) 10)
    (unused p9)
    (goalsize p9 small)
    (= (spray-varnish-cost p9) 5)
    (= (glaze-cost p9) 10)
    (= (grind-cost p9) 15)
    (= (plane-cost p9) 10)
    (unused p10)
    (goalsize p10 medium)
    (= (spray-varnish-cost p10) 10)
    (= (glaze-cost p10) 15)
    (= (grind-cost p10) 30)
    (= (plane-cost p10) 20)
    (unused p11)
    (goalsize p11 medium)
    (= (spray-varnish-cost p11) 10)
    (= (glaze-cost p11) 15)
    (= (grind-cost p11) 30)
    (= (plane-cost p11) 20)
    (unused p12)
    (goalsize p12 large)
    (= (spray-varnish-cost p12) 15)
    (= (glaze-cost p12) 20)
    (= (grind-cost p12) 45)
    (= (plane-cost p12) 30)
    (unused p13)
    (goalsize p13 medium)
    (= (spray-varnish-cost p13) 10)
    (= (glaze-cost p13) 15)
    (= (grind-cost p13) 30)
    (= (plane-cost p13) 20)
    (unused p14)
    (goalsize p14 medium)
    (= (spray-varnish-cost p14) 10)
    (= (glaze-cost p14) 15)
    (= (grind-cost p14) 30)
    (= (plane-cost p14) 20)
    (available p15)
    (treatment p15 varnished)
    (surface-condition p15 rough)
    (wood p15 mahogany)
    (colour p15 red)
    (goalsize p15 large)
    (= (spray-varnish-cost p15) 15)
    (= (glaze-cost p15) 20)
    (= (grind-cost p15) 45)
    (= (plane-cost p15) 30)
    (available p16)
    (treatment p16 varnished)
    (surface-condition p16 rough)
    (wood p16 cherry)
    (colour p16 natural)
    (goalsize p16 small)
    (= (spray-varnish-cost p16) 5)
    (= (glaze-cost p16) 10)
    (= (grind-cost p16) 15)
    (= (plane-cost p16) 10)
    (unused p17)
    (goalsize p17 large)
    (= (spray-varnish-cost p17) 15)
    (= (glaze-cost p17) 20)
    (= (grind-cost p17) 45)
    (= (plane-cost p17) 30)
    (unused p18)
    (goalsize p18 large)
    (= (spray-varnish-cost p18) 15)
    (= (glaze-cost p18) 20)
    (= (grind-cost p18) 45)
    (= (plane-cost p18) 30)
    (unused p19)
    (goalsize p19 medium)
    (= (spray-varnish-cost p19) 10)
    (= (glaze-cost p19) 15)
    (= (grind-cost p19) 30)
    (= (plane-cost p19) 20)
    (unused p20)
    (goalsize p20 medium)
    (= (spray-varnish-cost p20) 10)
    (= (glaze-cost p20) 15)
    (= (grind-cost p20) 30)
    (= (plane-cost p20) 20)
    (unused p21)
    (goalsize p21 large)
    (= (spray-varnish-cost p21) 15)
    (= (glaze-cost p21) 20)
    (= (grind-cost p21) 45)
    (= (plane-cost p21) 30)
    (available p22)
    (treatment p22 glazed)
    (surface-condition p22 smooth)
    (wood p22 beech)
    (colour p22 mauve)
    (goalsize p22 medium)
    (= (spray-varnish-cost p22) 10)
    (= (glaze-cost p22) 15)
    (= (grind-cost p22) 30)
    (= (plane-cost p22) 20)
    (available p23)
    (treatment p23 varnished)
    (surface-condition p23 verysmooth)
    (wood p23 mahogany)
    (colour p23 mauve)
    (goalsize p23 large)
    (= (spray-varnish-cost p23) 15)
    (= (glaze-cost p23) 20)
    (= (grind-cost p23) 45)
    (= (plane-cost p23) 30)
    (unused p24)
    (goalsize p24 medium)
    (= (spray-varnish-cost p24) 10)
    (= (glaze-cost p24) 15)
    (= (grind-cost p24) 30)
    (= (plane-cost p24) 20)
    (unused p25)
    (goalsize p25 large)
    (= (spray-varnish-cost p25) 15)
    (= (glaze-cost p25) 20)
    (= (grind-cost p25) 45)
    (= (plane-cost p25) 30)
    (unused p26)
    (goalsize p26 medium)
    (= (spray-varnish-cost p26) 10)
    (= (glaze-cost p26) 15)
    (= (grind-cost p26) 30)
    (= (plane-cost p26) 20)
    (unused p27)
    (goalsize p27 small)
    (= (spray-varnish-cost p27) 5)
    (= (glaze-cost p27) 10)
    (= (grind-cost p27) 15)
    (= (plane-cost p27) 10)
    (available p28)
    (treatment p28 varnished)
    (surface-condition p28 smooth)
    (wood p28 mahogany)
    (colour p28 red)
    (goalsize p28 small)
    (= (spray-varnish-cost p28) 5)
    (= (glaze-cost p28) 10)
    (= (grind-cost p28) 15)
    (= (plane-cost p28) 10)
    (unused p29)
    (goalsize p29 medium)
    (= (spray-varnish-cost p29) 10)
    (= (glaze-cost p29) 15)
    (= (grind-cost p29) 30)
    (= (plane-cost p29) 20)
    (unused p30)
    (goalsize p30 large)
    (= (spray-varnish-cost p30) 15)
    (= (glaze-cost p30) 20)
    (= (grind-cost p30) 45)
    (= (plane-cost p30) 30)
    (available p31)
    (treatment p31 glazed)
    (surface-condition p31 smooth)
    (wood p31 beech)
    (colour p31 blue)
    (goalsize p31 large)
    (= (spray-varnish-cost p31) 15)
    (= (glaze-cost p31) 20)
    (= (grind-cost p31) 45)
    (= (plane-cost p31) 30)
    (unused p32)
    (goalsize p32 small)
    (= (spray-varnish-cost p32) 5)
    (= (glaze-cost p32) 10)
    (= (grind-cost p32) 15)
    (= (plane-cost p32) 10)
    (available p33)
    (treatment p33 varnished)
    (surface-condition p33 rough)
    (wood p33 mahogany)
    (colour p33 blue)
    (goalsize p33 large)
    (= (spray-varnish-cost p33) 15)
    (= (glaze-cost p33) 20)
    (= (grind-cost p33) 45)
    (= (plane-cost p33) 30)
    (available p34)
    (treatment p34 varnished)
    (surface-condition p34 rough)
    (wood p34 beech)
    (colour p34 white)
    (goalsize p34 large)
    (= (spray-varnish-cost p34) 15)
    (= (glaze-cost p34) 20)
    (= (grind-cost p34) 45)
    (= (plane-cost p34) 30)
    (unused p35)
    (goalsize p35 large)
    (= (spray-varnish-cost p35) 15)
    (= (glaze-cost p35) 20)
    (= (grind-cost p35) 45)
    (= (plane-cost p35) 30)
    (unused p36)
    (goalsize p36 large)
    (= (spray-varnish-cost p36) 15)
    (= (glaze-cost p36) 20)
    (= (grind-cost p36) 45)
    (= (plane-cost p36) 30)
    (unused p37)
    (goalsize p37 large)
    (= (spray-varnish-cost p37) 15)
    (= (glaze-cost p37) 20)
    (= (grind-cost p37) 45)
    (= (plane-cost p37) 30)
    (unused p38)
    (goalsize p38 large)
    (= (spray-varnish-cost p38) 15)
    (= (glaze-cost p38) 20)
    (= (grind-cost p38) 45)
    (= (plane-cost p38) 30)
    (unused p39)
    (goalsize p39 large)
    (= (spray-varnish-cost p39) 15)
    (= (glaze-cost p39) 20)
    (= (grind-cost p39) 45)
    (= (plane-cost p39) 30)
    (unused p40)
    (goalsize p40 medium)
    (= (spray-varnish-cost p40) 10)
    (= (glaze-cost p40) 15)
    (= (grind-cost p40) 30)
    (= (plane-cost p40) 20)
    (unused p41)
    (goalsize p41 small)
    (= (spray-varnish-cost p41) 5)
    (= (glaze-cost p41) 10)
    (= (grind-cost p41) 15)
    (= (plane-cost p41) 10)
    (available p42)
    (treatment p42 glazed)
    (surface-condition p42 rough)
    (wood p42 walnut)
    (colour p42 white)
    (goalsize p42 medium)
    (= (spray-varnish-cost p42) 10)
    (= (glaze-cost p42) 15)
    (= (grind-cost p42) 30)
    (= (plane-cost p42) 20)
    (unused p43)
    (goalsize p43 small)
    (= (spray-varnish-cost p43) 5)
    (= (glaze-cost p43) 10)
    (= (grind-cost p43) 15)
    (= (plane-cost p43) 10)
    (unused p44)
    (goalsize p44 medium)
    (= (spray-varnish-cost p44) 10)
    (= (glaze-cost p44) 15)
    (= (grind-cost p44) 30)
    (= (plane-cost p44) 20)
    (unused p45)
    (goalsize p45 small)
    (= (spray-varnish-cost p45) 5)
    (= (glaze-cost p45) 10)
    (= (grind-cost p45) 15)
    (= (plane-cost p45) 10)
    (unused p46)
    (goalsize p46 large)
    (= (spray-varnish-cost p46) 15)
    (= (glaze-cost p46) 20)
    (= (grind-cost p46) 45)
    (= (plane-cost p46) 30)
    (unused p47)
    (goalsize p47 small)
    (= (spray-varnish-cost p47) 5)
    (= (glaze-cost p47) 10)
    (= (grind-cost p47) 15)
    (= (plane-cost p47) 10)
    (unused p48)
    (goalsize p48 medium)
    (= (spray-varnish-cost p48) 10)
    (= (glaze-cost p48) 15)
    (= (grind-cost p48) 30)
    (= (plane-cost p48) 20)
    (unused p49)
    (goalsize p49 large)
    (= (spray-varnish-cost p49) 15)
    (= (glaze-cost p49) 20)
    (= (grind-cost p49) 45)
    (= (plane-cost p49) 30)
    (available p50)
    (treatment p50 colourfragments)
    (surface-condition p50 smooth)
    (wood p50 oak)
    (colour p50 natural)
    (goalsize p50 small)
    (= (spray-varnish-cost p50) 5)
    (= (glaze-cost p50) 10)
    (= (grind-cost p50) 15)
    (= (plane-cost p50) 10)
    (unused p51)
    (goalsize p51 medium)
    (= (spray-varnish-cost p51) 10)
    (= (glaze-cost p51) 15)
    (= (grind-cost p51) 30)
    (= (plane-cost p51) 20)
    (unused p52)
    (goalsize p52 large)
    (= (spray-varnish-cost p52) 15)
    (= (glaze-cost p52) 20)
    (= (grind-cost p52) 45)
    (= (plane-cost p52) 30)
    (unused p53)
    (goalsize p53 small)
    (= (spray-varnish-cost p53) 5)
    (= (glaze-cost p53) 10)
    (= (grind-cost p53) 15)
    (= (plane-cost p53) 10)
    (unused p54)
    (goalsize p54 large)
    (= (spray-varnish-cost p54) 15)
    (= (glaze-cost p54) 20)
    (= (grind-cost p54) 45)
    (= (plane-cost p54) 30)
    (available p55)
    (treatment p55 glazed)
    (surface-condition p55 smooth)
    (wood p55 walnut)
    (colour p55 natural)
    (goalsize p55 medium)
    (= (spray-varnish-cost p55) 10)
    (= (glaze-cost p55) 15)
    (= (grind-cost p55) 30)
    (= (plane-cost p55) 20)
    (unused p56)
    (goalsize p56 small)
    (= (spray-varnish-cost p56) 5)
    (= (glaze-cost p56) 10)
    (= (grind-cost p56) 15)
    (= (plane-cost p56) 10)
    (available p57)
    (treatment p57 colourfragments)
    (surface-condition p57 verysmooth)
    (wood p57 teak)
    (colour p57 black)
    (goalsize p57 large)
    (= (spray-varnish-cost p57) 15)
    (= (glaze-cost p57) 20)
    (= (grind-cost p57) 45)
    (= (plane-cost p57) 30)
    (unused p58)
    (goalsize p58 large)
    (= (spray-varnish-cost p58) 15)
    (= (glaze-cost p58) 20)
    (= (grind-cost p58) 45)
    (= (plane-cost p58) 30)
    (unused p59)
    (goalsize p59 small)
    (= (spray-varnish-cost p59) 5)
    (= (glaze-cost p59) 10)
    (= (grind-cost p59) 15)
    (= (plane-cost p59) 10)
    (unused p60)
    (goalsize p60 medium)
    (= (spray-varnish-cost p60) 10)
    (= (glaze-cost p60) 15)
    (= (grind-cost p60) 30)
    (= (plane-cost p60) 20)
    (unused p61)
    (goalsize p61 small)
    (= (spray-varnish-cost p61) 5)
    (= (glaze-cost p61) 10)
    (= (grind-cost p61) 15)
    (= (plane-cost p61) 10)
    (unused p62)
    (goalsize p62 small)
    (= (spray-varnish-cost p62) 5)
    (= (glaze-cost p62) 10)
    (= (grind-cost p62) 15)
    (= (plane-cost p62) 10)
    (unused p63)
    (goalsize p63 small)
    (= (spray-varnish-cost p63) 5)
    (= (glaze-cost p63) 10)
    (= (grind-cost p63) 15)
    (= (plane-cost p63) 10)
    (unused p64)
    (goalsize p64 medium)
    (= (spray-varnish-cost p64) 10)
    (= (glaze-cost p64) 15)
    (= (grind-cost p64) 30)
    (= (plane-cost p64) 20)
    (unused p65)
    (goalsize p65 large)
    (= (spray-varnish-cost p65) 15)
    (= (glaze-cost p65) 20)
    (= (grind-cost p65) 45)
    (= (plane-cost p65) 30)
    (unused p66)
    (goalsize p66 small)
    (= (spray-varnish-cost p66) 5)
    (= (glaze-cost p66) 10)
    (= (grind-cost p66) 15)
    (= (plane-cost p66) 10)
    (available p67)
    (treatment p67 glazed)
    (surface-condition p67 rough)
    (wood p67 pine)
    (colour p67 blue)
    (goalsize p67 large)
    (= (spray-varnish-cost p67) 15)
    (= (glaze-cost p67) 20)
    (= (grind-cost p67) 45)
    (= (plane-cost p67) 30)
    (unused p68)
    (goalsize p68 large)
    (= (spray-varnish-cost p68) 15)
    (= (glaze-cost p68) 20)
    (= (grind-cost p68) 45)
    (= (plane-cost p68) 30)
    (unused p69)
    (goalsize p69 large)
    (= (spray-varnish-cost p69) 15)
    (= (glaze-cost p69) 20)
    (= (grind-cost p69) 45)
    (= (plane-cost p69) 30)
    (unused p70)
    (goalsize p70 small)
    (= (spray-varnish-cost p70) 5)
    (= (glaze-cost p70) 10)
    (= (grind-cost p70) 15)
    (= (plane-cost p70) 10)
    (unused p71)
    (goalsize p71 small)
    (= (spray-varnish-cost p71) 5)
    (= (glaze-cost p71) 10)
    (= (grind-cost p71) 15)
    (= (plane-cost p71) 10)
    (unused p72)
    (goalsize p72 large)
    (= (spray-varnish-cost p72) 15)
    (= (glaze-cost p72) 20)
    (= (grind-cost p72) 45)
    (= (plane-cost p72) 30)
    (available p73)
    (treatment p73 varnished)
    (surface-condition p73 smooth)
    (wood p73 oak)
    (colour p73 mauve)
    (goalsize p73 small)
    (= (spray-varnish-cost p73) 5)
    (= (glaze-cost p73) 10)
    (= (grind-cost p73) 15)
    (= (plane-cost p73) 10)
    (unused p74)
    (goalsize p74 large)
    (= (spray-varnish-cost p74) 15)
    (= (glaze-cost p74) 20)
    (= (grind-cost p74) 45)
    (= (plane-cost p74) 30)
    (unused p75)
    (goalsize p75 small)
    (= (spray-varnish-cost p75) 5)
    (= (glaze-cost p75) 10)
    (= (grind-cost p75) 15)
    (= (plane-cost p75) 10)
    (unused p76)
    (goalsize p76 large)
    (= (spray-varnish-cost p76) 15)
    (= (glaze-cost p76) 20)
    (= (grind-cost p76) 45)
    (= (plane-cost p76) 30)
    (unused p77)
    (goalsize p77 small)
    (= (spray-varnish-cost p77) 5)
    (= (glaze-cost p77) 10)
    (= (grind-cost p77) 15)
    (= (plane-cost p77) 10)
    (unused p78)
    (goalsize p78 medium)
    (= (spray-varnish-cost p78) 10)
    (= (glaze-cost p78) 15)
    (= (grind-cost p78) 30)
    (= (plane-cost p78) 20)
    (available p79)
    (treatment p79 glazed)
    (surface-condition p79 verysmooth)
    (wood p79 cherry)
    (colour p79 mauve)
    (goalsize p79 medium)
    (= (spray-varnish-cost p79) 10)
    (= (glaze-cost p79) 15)
    (= (grind-cost p79) 30)
    (= (plane-cost p79) 20)
    (unused p80)
    (goalsize p80 medium)
    (= (spray-varnish-cost p80) 10)
    (= (glaze-cost p80) 15)
    (= (grind-cost p80) 30)
    (= (plane-cost p80) 20)
    (available p81)
    (treatment p81 colourfragments)
    (surface-condition p81 rough)
    (wood p81 mahogany)
    (colour p81 white)
    (goalsize p81 large)
    (= (spray-varnish-cost p81) 15)
    (= (glaze-cost p81) 20)
    (= (grind-cost p81) 45)
    (= (plane-cost p81) 30)
    (unused p82)
    (goalsize p82 medium)
    (= (spray-varnish-cost p82) 10)
    (= (glaze-cost p82) 15)
    (= (grind-cost p82) 30)
    (= (plane-cost p82) 20)
    (available p83)
    (treatment p83 colourfragments)
    (surface-condition p83 smooth)
    (wood p83 cherry)
    (colour p83 blue)
    (goalsize p83 small)
    (= (spray-varnish-cost p83) 5)
    (= (glaze-cost p83) 10)
    (= (grind-cost p83) 15)
    (= (plane-cost p83) 10)
    (unused p84)
    (goalsize p84 medium)
    (= (spray-varnish-cost p84) 10)
    (= (glaze-cost p84) 15)
    (= (grind-cost p84) 30)
    (= (plane-cost p84) 20)
    (unused p85)
    (goalsize p85 medium)
    (= (spray-varnish-cost p85) 10)
    (= (glaze-cost p85) 15)
    (= (grind-cost p85) 30)
    (= (plane-cost p85) 20)
    (unused p86)
    (goalsize p86 medium)
    (= (spray-varnish-cost p86) 10)
    (= (glaze-cost p86) 15)
    (= (grind-cost p86) 30)
    (= (plane-cost p86) 20)
    (unused p87)
    (goalsize p87 large)
    (= (spray-varnish-cost p87) 15)
    (= (glaze-cost p87) 20)
    (= (grind-cost p87) 45)
    (= (plane-cost p87) 30)
    (available p88)
    (treatment p88 varnished)
    (surface-condition p88 rough)
    (wood p88 beech)
    (colour p88 natural)
    (goalsize p88 large)
    (= (spray-varnish-cost p88) 15)
    (= (glaze-cost p88) 20)
    (= (grind-cost p88) 45)
    (= (plane-cost p88) 30)
    (unused p89)
    (goalsize p89 small)
    (= (spray-varnish-cost p89) 5)
    (= (glaze-cost p89) 10)
    (= (grind-cost p89) 15)
    (= (plane-cost p89) 10)
    (unused p90)
    (goalsize p90 small)
    (= (spray-varnish-cost p90) 5)
    (= (glaze-cost p90) 10)
    (= (grind-cost p90) 15)
    (= (plane-cost p90) 10)
    (unused p91)
    (goalsize p91 medium)
    (= (spray-varnish-cost p91) 10)
    (= (glaze-cost p91) 15)
    (= (grind-cost p91) 30)
    (= (plane-cost p91) 20)
    (unused p92)
    (goalsize p92 small)
    (= (spray-varnish-cost p92) 5)
    (= (glaze-cost p92) 10)
    (= (grind-cost p92) 15)
    (= (plane-cost p92) 10)
    (unused p93)
    (goalsize p93 small)
    (= (spray-varnish-cost p93) 5)
    (= (glaze-cost p93) 10)
    (= (grind-cost p93) 15)
    (= (plane-cost p93) 10)
    (unused p94)
    (goalsize p94 large)
    (= (spray-varnish-cost p94) 15)
    (= (glaze-cost p94) 20)
    (= (grind-cost p94) 45)
    (= (plane-cost p94) 30)
    (available p95)
    (treatment p95 varnished)
    (surface-condition p95 verysmooth)
    (wood p95 pine)
    (colour p95 blue)
    (goalsize p95 small)
    (= (spray-varnish-cost p95) 5)
    (= (glaze-cost p95) 10)
    (= (grind-cost p95) 15)
    (= (plane-cost p95) 10)
    (unused p96)
    (goalsize p96 large)
    (= (spray-varnish-cost p96) 15)
    (= (glaze-cost p96) 20)
    (= (grind-cost p96) 45)
    (= (plane-cost p96) 30)
    (available p97)
    (treatment p97 glazed)
    (surface-condition p97 verysmooth)
    (wood p97 cherry)
    (colour p97 natural)
    (goalsize p97 large)
    (= (spray-varnish-cost p97) 15)
    (= (glaze-cost p97) 20)
    (= (grind-cost p97) 45)
    (= (plane-cost p97) 30)
    (unused p98)
    (goalsize p98 large)
    (= (spray-varnish-cost p98) 15)
    (= (glaze-cost p98) 20)
    (= (grind-cost p98) 45)
    (= (plane-cost p98) 30)
    (boardsize b0 s12)
    (wood b0 oak)
    (surface-condition b0 rough)
    (available b0)
    (boardsize b1 s9)
    (wood b1 oak)
    (surface-condition b1 smooth)
    (available b1)
    (boardsize b2 s4)
    (wood b2 oak)
    (surface-condition b2 rough)
    (available b2)
    (boardsize b3 s4)
    (wood b3 oak)
    (surface-condition b3 smooth)
    (available b3)
    (boardsize b4 s10)
    (wood b4 walnut)
    (surface-condition b4 smooth)
    (available b4)
    (boardsize b5 s6)
    (wood b5 walnut)
    (surface-condition b5 rough)
    (available b5)
    (boardsize b6 s7)
    (wood b6 walnut)
    (surface-condition b6 smooth)
    (available b6)
    (boardsize b7 s10)
    (wood b7 walnut)
    (surface-condition b7 smooth)
    (available b7)
    (boardsize b8 s7)
    (wood b8 walnut)
    (surface-condition b8 rough)
    (available b8)
    (boardsize b9 s3)
    (wood b9 walnut)
    (surface-condition b9 rough)
    (available b9)
    (boardsize b10 s10)
    (wood b10 teak)
    (surface-condition b10 smooth)
    (available b10)
    (boardsize b11 s8)
    (wood b11 teak)
    (surface-condition b11 smooth)
    (available b11)
    (boardsize b12 s6)
    (wood b12 teak)
    (surface-condition b12 smooth)
    (available b12)
    (boardsize b13 s8)
    (wood b13 pine)
    (surface-condition b13 smooth)
    (available b13)
    (boardsize b14 s9)
    (wood b14 pine)
    (surface-condition b14 rough)
    (available b14)
    (boardsize b15 s7)
    (wood b15 pine)
    (surface-condition b15 rough)
    (available b15)
    (boardsize b16 s6)
    (wood b16 pine)
    (surface-condition b16 rough)
    (available b16)
    (boardsize b17 s7)
    (wood b17 beech)
    (surface-condition b17 rough)
    (available b17)
    (boardsize b18 s7)
    (wood b18 beech)
    (surface-condition b18 rough)
    (available b18)
    (boardsize b19 s6)
    (wood b19 beech)
    (surface-condition b19 rough)
    (available b19)
    (boardsize b20 s7)
    (wood b20 beech)
    (surface-condition b20 smooth)
    (available b20)
    (boardsize b21 s11)
    (wood b21 cherry)
    (surface-condition b21 rough)
    (available b21)
    (boardsize b22 s9)
    (wood b22 cherry)
    (surface-condition b22 rough)
    (available b22)
    (boardsize b23 s9)
    (wood b23 cherry)
    (surface-condition b23 smooth)
    (available b23)
    (boardsize b24 s10)
    (wood b24 cherry)
    (surface-condition b24 rough)
    (available b24)
    (boardsize b25 s5)
    (wood b25 mahogany)
    (surface-condition b25 rough)
    (available b25)
    (boardsize b26 s9)
    (wood b26 mahogany)
    (surface-condition b26 rough)
    (available b26)
  )
  (:goal
    (and
    (available p0)
    (surface-condition p0 verysmooth)
    (colour p0 green)
    (available p1)
    (treatment p1 glazed)
    (colour p1 natural)
    (available p2)
    (wood p2 teak)
    (treatment p2 glazed)
    (available p3)
    (surface-condition p3 smooth)
    (colour p3 green)
    (available p4)
    (wood p4 teak)
    (treatment p4 glazed)
    (colour p4 white)
    (available p5)
    (surface-condition p5 smooth)
    (treatment p5 glazed)
    (colour p5 green)
    (available p6)
    (treatment p6 varnished)
    (colour p6 blue)
    (available p7)
    (wood p7 pine)
    (surface-condition p7 smooth)
    (treatment p7 glazed)
    (colour p7 red)
    (available p8)
    (wood p8 walnut)
    (surface-condition p8 verysmooth)
    (colour p8 natural)
    (available p9)
    (surface-condition p9 verysmooth)
    (colour p9 green)
    (available p10)
    (wood p10 walnut)
    (surface-condition p10 smooth)
    (available p11)
    (surface-condition p11 verysmooth)
    (treatment p11 glazed)
    (available p12)
    (wood p12 oak)
    (surface-condition p12 verysmooth)
    (treatment p12 varnished)
    (colour p12 blue)
    (available p13)
    (wood p13 pine)
    (treatment p13 varnished)
    (colour p13 mauve)
    (available p14)
    (surface-condition p14 smooth)
    (treatment p14 glazed)
    (colour p14 black)
    (available p15)
    (wood p15 mahogany)
    (surface-condition p15 smooth)
    (treatment p15 glazed)
    (colour p15 mauve)
    (available p16)
    (surface-condition p16 verysmooth)
    (treatment p16 varnished)
    (available p17)
    (wood p17 cherry)
    (surface-condition p17 verysmooth)
    (available p18)
    (wood p18 cherry)
    (surface-condition p18 smooth)
    (treatment p18 glazed)
    (colour p18 white)
    (available p19)
    (wood p19 mahogany)
    (treatment p19 glazed)
    (available p20)
    (wood p20 cherry)
    (treatment p20 varnished)
    (available p21)
    (surface-condition p21 smooth)
    (treatment p21 glazed)
    (colour p21 white)
    (available p22)
    (wood p22 beech)
    (surface-condition p22 smooth)
    (treatment p22 varnished)
    (colour p22 blue)
    (available p23)
    (surface-condition p23 verysmooth)
    (colour p23 white)
    (available p24)
    (wood p24 pine)
    (surface-condition p24 smooth)
    (available p25)
    (wood p25 teak)
    (treatment p25 varnished)
    (colour p25 mauve)
    (available p26)
    (wood p26 beech)
    (colour p26 white)
    (available p27)
    (wood p27 beech)
    (surface-condition p27 smooth)
    (treatment p27 varnished)
    (colour p27 natural)
    (available p28)
    (surface-condition p28 smooth)
    (colour p28 white)
    (available p29)
    (surface-condition p29 verysmooth)
    (treatment p29 varnished)
    (available p30)
    (surface-condition p30 verysmooth)
    (colour p30 white)
    (available p31)
    (wood p31 beech)
    (treatment p31 varnished)
    (available p32)
    (wood p32 beech)
    (colour p32 mauve)
    (available p33)
    (wood p33 mahogany)
    (colour p33 mauve)
    (available p34)
    (wood p34 beech)
    (colour p34 green)
    (available p35)
    (wood p35 teak)
    (colour p35 blue)
    (available p36)
    (surface-condition p36 smooth)
    (treatment p36 glazed)
    (colour p36 green)
    (available p37)
    (wood p37 cherry)
    (colour p37 red)
    (available p38)
    (wood p38 cherry)
    (treatment p38 varnished)
    (colour p38 mauve)
    (available p39)
    (surface-condition p39 verysmooth)
    (treatment p39 varnished)
    (available p40)
    (wood p40 cherry)
    (surface-condition p40 verysmooth)
    (treatment p40 glazed)
    (colour p40 mauve)
    (available p41)
    (surface-condition p41 verysmooth)
    (treatment p41 varnished)
    (available p42)
    (wood p42 walnut)
    (colour p42 green)
    (available p43)
    (surface-condition p43 verysmooth)
    (treatment p43 glazed)
    (available p44)
    (wood p44 walnut)
    (surface-condition p44 smooth)
    (available p45)
    (wood p45 beech)
    (surface-condition p45 verysmooth)
    (treatment p45 glazed)
    (available p46)
    (surface-condition p46 smooth)
    (treatment p46 glazed)
    (available p47)
    (surface-condition p47 verysmooth)
    (colour p47 green)
    (available p48)
    (surface-condition p48 smooth)
    (colour p48 natural)
    (available p49)
    (surface-condition p49 smooth)
    (treatment p49 glazed)
    (available p50)
    (surface-condition p50 smooth)
    (colour p50 red)
    (available p51)
    (wood p51 pine)
    (surface-condition p51 verysmooth)
    (treatment p51 varnished)
    (colour p51 black)
    (available p52)
    (surface-condition p52 smooth)
    (treatment p52 glazed)
    (available p53)
    (wood p53 walnut)
    (surface-condition p53 smooth)
    (treatment p53 glazed)
    (colour p53 mauve)
    (available p54)
    (wood p54 teak)
    (surface-condition p54 smooth)
    (treatment p54 varnished)
    (colour p54 red)
    (available p55)
    (wood p55 walnut)
    (surface-condition p55 smooth)
    (treatment p55 glazed)
    (colour p55 black)
    (available p56)
    (wood p56 mahogany)
    (surface-condition p56 verysmooth)
    (treatment p56 varnished)
    (colour p56 red)
    (available p57)
    (wood p57 teak)
    (treatment p57 varnished)
    (colour p57 white)
    (available p58)
    (wood p58 beech)
    (surface-condition p58 smooth)
    (available p59)
    (surface-condition p59 smooth)
    (treatment p59 varnished)
    (available p60)
    (wood p60 pine)
    (surface-condition p60 verysmooth)
    (treatment p60 glazed)
    (colour p60 black)
    (available p61)
    (wood p61 walnut)
    (treatment p61 glazed)
    (available p62)
    (wood p62 walnut)
    (surface-condition p62 smooth)
    (treatment p62 varnished)
    (colour p62 white)
    (available p63)
    (treatment p63 glazed)
    (colour p63 green)
    (available p64)
    (treatment p64 glazed)
    (colour p64 white)
    (available p65)
    (wood p65 beech)
    (treatment p65 glazed)
    (available p66)
    (wood p66 mahogany)
    (surface-condition p66 smooth)
    (colour p66 blue)
    (available p67)
    (surface-condition p67 smooth)
    (colour p67 white)
    (available p68)
    (wood p68 mahogany)
    (treatment p68 varnished)
    (available p69)
    (wood p69 cherry)
    (surface-condition p69 verysmooth)
    (treatment p69 varnished)
    (colour p69 black)
    (available p70)
    (surface-condition p70 smooth)
    (colour p70 white)
    (available p71)
    (wood p71 beech)
    (colour p71 red)
    (available p72)
    (wood p72 walnut)
    (surface-condition p72 smooth)
    (available p73)
    (wood p73 oak)
    (colour p73 blue)
    (available p74)
    (treatment p74 varnished)
    (colour p74 white)
    (available p75)
    (wood p75 teak)
    (surface-condition p75 smooth)
    (treatment p75 varnished)
    (colour p75 white)
    (available p76)
    (wood p76 walnut)
    (surface-condition p76 verysmooth)
    (treatment p76 glazed)
    (colour p76 mauve)
    (available p77)
    (wood p77 mahogany)
    (surface-condition p77 verysmooth)
    (available p78)
    (wood p78 cherry)
    (surface-condition p78 smooth)
    (treatment p78 glazed)
    (colour p78 green)
    (available p79)
    (wood p79 cherry)
    (treatment p79 varnished)
    (available p80)
    (surface-condition p80 smooth)
    (colour p80 black)
    (available p81)
    (surface-condition p81 verysmooth)
    (colour p81 red)
    (available p82)
    (wood p82 pine)
    (surface-condition p82 verysmooth)
    (treatment p82 varnished)
    (colour p82 green)
    (available p83)
    (wood p83 cherry)
    (colour p83 black)
    (available p84)
    (wood p84 beech)
    (surface-condition p84 smooth)
    (available p85)
    (wood p85 cherry)
    (colour p85 white)
    (available p86)
    (wood p86 oak)
    (surface-condition p86 smooth)
    (available p87)
    (wood p87 beech)
    (surface-condition p87 smooth)
    (treatment p87 glazed)
    (colour p87 mauve)
    (available p88)
    (wood p88 beech)
    (colour p88 white)
    (available p89)
    (wood p89 teak)
    (surface-condition p89 smooth)
    (treatment p89 glazed)
    (colour p89 red)
    (available p90)
    (wood p90 oak)
    (surface-condition p90 smooth)
    (available p91)
    (wood p91 walnut)
    (surface-condition p91 verysmooth)
    (available p92)
    (wood p92 mahogany)
    (treatment p92 glazed)
    (colour p92 blue)
    (available p93)
    (surface-condition p93 smooth)
    (treatment p93 varnished)
    (available p94)
    (wood p94 cherry)
    (treatment p94 glazed)
    (colour p94 natural)
    (available p95)
    (surface-condition p95 smooth)
    (treatment p95 glazed)
    (available p96)
    (wood p96 walnut)
    (surface-condition p96 smooth)
    (treatment p96 varnished)
    (colour p96 green)
    (available p97)
    (wood p97 cherry)
    (surface-condition p97 smooth)
    (treatment p97 glazed)
    (colour p97 blue)
    (available p98)
    (wood p98 pine)
    (surface-condition p98 smooth)
    )
  )
  (:metric minimize (total-cost))
)