package main

import (
	"math"
	"math/rand"
	"time"
)

type Network struct {
	layers []Layer

	learn_rate       float64
	learn_rate_decay float64
	momentum         float64

	input_shape []int
	input_size  int
	output_size int

	cost_interface Cost
	file_name      string
}

func (shelf *Network) init() {

	prev_layer_size := shelf.input_shape
	shelf.input_size = 1
	for _, v := range shelf.input_shape {
		shelf.input_size *= v
	}

	for layerID, layer := range shelf.layers {
		layer.init(layerID+1, prev_layer_size)
		prev_layer_size = layer.get_size()
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

	// 1d
	//mim.data_type = len(shelf.input_shape) - 1
	mim.data_type = OneD
	mim.data_type_history[0] = mim.data_type

	switch mim.data_type {
	case OneD:
		mim.data_flat = &data

		mim.layers_out_flat[0] = data
		mim.layers_out_flat_non_activated[0] = data

	case ThreeD:
		threeDIn := make([][][]float64, shelf.input_shape[0])
		for depth := 0; depth < shelf.input_shape[0]; depth++ {
			threeDIn[depth] = make([][]float64, shelf.input_shape[1])
			for x := 0; x < shelf.input_shape[1]; x++ {
				threeDIn[0][x] = make([]float64, shelf.input_shape[2])
				for y := 0; y < shelf.input_shape[2]; y++ {
					threeDIn[0][x][y] = data[depth*shelf.input_shape[0]+x*shelf.input_shape[1]+y]
				}
			}
		}
		mim.data_3d = &threeDIn
		mim.layers_out_3d[0] = threeDIn
		mim.layers_out_3d_non_activated[0] = threeDIn

	}

	// fmt.Println("---")
	for _, layer := range shelf.layers {
		layer.forward(mim)
		//layer.debug_print()
		// fmt.Println(layer.get_size())
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
