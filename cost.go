package main

type Cost interface {
	call(out_values []float64, target_values []float64) float64

	ddx(out_val float64, target_val float64) float64
	get_name() string
}

func cost_name_to_interface(name string) Cost {
	switch name {
	case "meansquare":
		return &MeanSquare{}

	case "tmp":
		return &MeanSquare{}

	}
	return &MeanSquare{}
}

type MeanSquare struct {
}

func (shelf *MeanSquare) call(out_values []float64, target_values []float64) float64 {
	sum := 0.0
	for i := range out_values {
		//sum += math.Pow(out_values[i]-target_values[i], 2)
		sum += (out_values[i] - target_values[i]) * (out_values[i] - target_values[i])
	}

	return sum
}

func (shelf *MeanSquare) ddx(out_val float64, target_val float64) float64 {
	return (out_val - target_val) * 2
}
func (shelf *MeanSquare) get_name() string {
	return "meansquare"
}
