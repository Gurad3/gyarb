package main

import (
	"math"
)

type Network struct {
	layers []Layer

	learn_rate       float64
	learn_rate_decay float64
	momentum         float64

	input_size  int
	output_size int

	cost_interface Cost
	file_name      string
}

func (shelf *Network) init() {
	for layerID, layer := range shelf.layers {
		layer.init(layerID + 1)
	}
}

func (shelf *Network) init_new_weights() {
	xavierRange := math.Sqrt(6 / float64(shelf.input_size+shelf.output_size))

	for _, layer := range shelf.layers {
		layer.init_new_weights(xavierRange)
	}
}

func (shelf *Network) print_weights() {
	for _, layer := range shelf.layers {
		layer.print_weights()
	}
}

func (shelf *Network) forward(mim *MiM, data []float64) {

	mim.data_flat = &data
	mim.data_type = OneD
	mim.data_type_history[0] = OneD
	mim.layers_out_flat[0] = data

	for _, layer := range shelf.layers {
		layer.forward(mim)
		layer.debug_print()
	}
}

func (shelf *Network) backprop(mim *MiM, labels []float64) {
	mim.data_flat = shelf.get_output_ddx(mim, labels)
	mim.data_type = OneD

	for layerID := len(shelf.layers) - 1; layerID >= 0; layerID++ {
		shelf.layers[layerID].backprop(mim)
	}

}

func (shelf *Network) get_output_ddx(mim *MiM, labels []float64) *[]float64 {
	gradiants := make([]float64, len(labels))

	for outID, output := range *mim.request_flat().data_flat {
		gradiants[outID] = shelf.layers[len(shelf.layers)-1].run_act_ddx(shelf.cost_interface.ddx(output, labels[outID]))
	}

	return &gradiants
}

func (shelf *Network) apply_gradients() {

	for _, layer := range shelf.layers {
		layer.apply_gradients()
	}
}
