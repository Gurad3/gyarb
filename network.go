package main

import (
	"fmt"
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

func (shelf *Network) forward(mim *MiM, data []float64, labels []float64) {

	mim.data_flat = &data
	mim.data_type = OneD
	mim.data_type_history[0] = OneD
	mim.layers_out_flat[0] = data

	for _, layer := range shelf.layers {
		layer.forward(mim)
		layer.debug_print()
	}
	mim.request_flat()
	fmt.Println(*mim.data_flat)
	fmt.Println(labels)
	fmt.Println(shelf.cost_interface.call(*mim.data_flat, data))

}

func (shelf *Network) get_output_ddx(mim *MiM, labels []float64) {

}
