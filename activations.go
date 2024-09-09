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
		return &RelU{}

	case "sigmoid":
		return &Sigmoid{}

	case "tanh":
		return &TanH{}
	}
	return &RelU{}
}

type RelU struct {
}

func (shelf *RelU) get_name() string {
	return "relu"
}
func (shelf *RelU) call(val float64) float64 {
	return max(0, val)
}
func (shelf *RelU) ddx(val float64) float64 {
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

type TanH struct{}

func (shelf *TanH) get_name() string {
	return "tanh"
}
func (shelf *TanH) call(val float64) float64 {
	posExp := math.Exp(val)
	negExp := math.Exp(-val)
	return (posExp - negExp) / (posExp + negExp)
}

func (shelf *TanH) ddx(val float64) float64 {

	return 1 - math.Pow(shelf.call(val), 2)
}

func initWeightXavierUniform(xavierRange float64, r rand.Rand) float64 {
	return r.Float64()*2*xavierRange - xavierRange
}
