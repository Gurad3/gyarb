package main

import (
	"math"
	"math/rand/v2"
)

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

type Sigmoid struct{}

func (shelf *Sigmoid) call(val float64) float64 {
	return 1 / (1 + math.Exp(-val))
}

func (shelf *Sigmoid) ddx(val float64) float64 {
	sig := shelf.call(val)
	return sig * (1 - sig)
}

func initWeightXavierUniform(xavierRange float64) float64 {
	return rand.Float64()*2*xavierRange - xavierRange
}
