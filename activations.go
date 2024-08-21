package main

type Activation interface {
	call(float64) float64
	ddx(float64) float64
}

type relU struct{}

func (shelf *relU) call(val float64) float64 {
	return max(0, val)
}

func (shelf *relU) ddx(val float64) float64 {
	if val > 0 {
		return 1
	} else {
		return 0
	}
}
