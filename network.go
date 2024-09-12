package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"
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
	r := rand.New(rand.NewSource(time.Now().Unix()))

	for _, layer := range shelf.layers {
		layer.init_new_weights(xavierRange, *r)
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
	mim.layers_out_flat_non_activated[0] = data

	for _, layer := range shelf.layers {
		layer.forward(mim)
		// layer.debug_print()
	}
}

func (shelf *Network) backprop(mim *MiM, labels []float64) {
	mim.data_flat = shelf.get_output_ddx(mim, labels)
	mim.data_type = OneD

	for layerID := len(shelf.layers) - 1; layerID > 0; layerID-- {
		shelf.layers[layerID].backprop(mim, shelf.layers[layerID-1].get_act_interface())
	}
	shelf.layers[0].backprop(mim, shelf.layers[0].get_act_interface())
}

func (shelf *Network) get_output_ddx(mim *MiM, labels []float64) *[]float64 {
	gradiants := make([]float64, len(labels))

	for outID, output := range *mim.request_flat().data_flat {
		//gradiants[outID] = shelf.layers[len(shelf.layers)-1].get_act_interface().ddx(shelf.cost_interface.ddx(output, labels[outID]))

		//fmt.Println(len(shelf.layers), len(mim.layers_out_flat_non_activated), len(mim.layers_out_flat_non_activated[len(shelf.layers)]))

		if len(shelf.layers)-1 == 10 {
			fmt.Println(1, len(shelf.layers))
		}
		if len(shelf.layers) == 10 {
			fmt.Println(2, len(shelf.layers), len(mim.layers_out_flat_non_activated))
		}
		if outID == 10 {
			fmt.Println(len(*mim.data_flat))
			fmt.Println(3, len(mim.layers_out_flat_non_activated[len(shelf.layers)]))
		}

		gradiants[outID] = shelf.layers[len(shelf.layers)-1].get_act_interface().ddx(mim.layers_out_flat_non_activated[len(shelf.layers)][outID])
		gradiants[outID] *= shelf.cost_interface.ddx(output, labels[outID])

	}

	return &gradiants
}

func (shelf *Network) apply_gradients(batch_size int) {

	for _, layer := range shelf.layers {
		layer.apply_gradients(shelf.learn_rate, float64(batch_size))
	}
}
