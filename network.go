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
}

func (shelf *Network) init() {
	for layerID, layer := range shelf.layers {
		layer.init(layerID)
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
