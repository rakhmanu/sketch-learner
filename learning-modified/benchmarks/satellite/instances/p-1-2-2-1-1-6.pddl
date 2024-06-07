(define (problem strips-sat-x-1)
(:domain satellite)
(:objects
	satellite0 - satellite
	instrument0 - instrument
	infrared0 - mode
	thermograph1 - mode
	GroundStation0 - direction
	Planet1 - direction
	Phenomenon2 - direction
)
(:init
	(supports instrument0 thermograph1)
	(supports instrument0 infrared0)
	(calibration_target instrument0 GroundStation0)
	(on_board instrument0 satellite0)
	(power_avail satellite0)
	(pointing satellite0 Planet1)
)
(:goal (and
	(pointing satellite0 Phenomenon2)
	(have_image Planet1 infrared0)
	(have_image Phenomenon2 infrared0)
))

)
