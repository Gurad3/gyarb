package main

import "math"

type Cost interface {
	call([]float64, []float64) float64

	ddx(float64, float64) float64
}

type MeanSquare struct {
}

func (shelf *MeanSquare) call(actual_values []float64, target_values []float64) float64 {
	sum := 0.0
	for i := range actual_values {
		sum += math.Pow(actual_values[i]-target_values[i], 2)
	}

	return sum / (2 * float64(len(actual_values)))
}

func (shelf *MeanSquare) ddx(actual_val float64, target_val float64) float64 {
	return actual_val - target_val
}
