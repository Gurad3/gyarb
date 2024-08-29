package main

import (
	"math"
)

type Network struct {
	Layers []Layer `json:"Layers"`

	Learn_rate       float64 `json:"learn_rate"`
	learn_rate_decay float64
	momentum         float64

	input_size  int
	output_size int

	cost_interface Cost
	file_name      string
}

func (shelf *Network) init() {
	for layerID, layer := range shelf.Layers {
		layer.init(layerID + 1)
	}
}

func (shelf *Network) init_new_weights() {
	xavierRange := math.Sqrt(6 / float64(shelf.input_size+shelf.output_size))

	for _, layer := range shelf.Layers {
		layer.init_new_weights(xavierRange)
	}
}

func (shelf *Network) print_weights() {
	for _, layer := range shelf.Layers {
		layer.print_weights()
	}
}

func (shelf *Network) forward(mim *MiM, data []float64) {

	mim.data_flat = &data
	mim.data_type = OneD
	mim.data_type_history[0] = OneD
	mim.layers_out_flat[0] = data

	for _, layer := range shelf.Layers {
		layer.forward(mim)
	}
}
