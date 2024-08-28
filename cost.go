package main

type Cost interface {
	call([]float64) float64

	ddx(float64) float64
}
