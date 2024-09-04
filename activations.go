package main

import (
	"math"
	"math/rand"
)

type Activation interface {
	call(float64) float64
	ddx(float64) float64
	get_name() string
}

func act_name_to_interface(name string) Activation {
	switch name {
	case "relu":
		return &relU{}

	case "sigmoid":
		return &Sigmoid{}

	}
	return &relU{}
}

type relU struct {
}

func (shelf *relU) get_name() string {
	return "relu"
}
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

func (shelf *Sigmoid) get_name() string {
	return "sigmoid"
}
func (shelf *Sigmoid) call(val float64) float64 {
	return 1 / (1 + math.Exp(-val))
}

func (shelf *Sigmoid) ddx(val float64) float64 {
	sig := shelf.call(val)
	return sig * (1 - sig)
}

func initWeightXavierUniform(xavierRange float64, r rand.Rand) float64 {
	return r.Float64()*2*xavierRange - xavierRange
}
