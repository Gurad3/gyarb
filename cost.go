package main

type Cost interface {
	call([]float64, []float64) float64

	ddx(float64) float64
}

type MeanSquare struct {
}

func (shelf *MeanSquare) call(actual_values []float64, target_values []float64) float64 {

}
