package main

import (
	"fmt"
	"math/rand"
)

type Layer interface {
	forward(*MiM)
	backprop(mim *MiM, prev_layer_act Activation)
	init(layerID int)
	init_new_weights(xavierRange float64, r rand.Rand)
	apply_gradients(learn_rate float64, batch_size float64)

	load_weights(flat_weights []float64)
	load_biases(flat_biases []float64)
	get_weights() []float64
	get_biases() []float64

	print_weights()
	get_size() []int
	get_name() string

	get_act_interface() Activation

	debug_print()
}

type DenseLayer struct {
	act_interface   Activation
	layerID         int
	size            int
	prev_layer_size int
	layer_type      string

	weights [][]float64
	bias    []float64

	weights_gradiants [][]float64
	bias_gradiants    []float64
}

type CNLayer struct {
	act_interface Activation
	layerID       int
	size          []int
	layer_type    string

	weights [][][]float64
	bias    [][]float64

	weights_gradiants [][][]float64
	bias_gradiants    [][]float64
}

func (shelf *DenseLayer) forward(mim *MiM) {
	data := *mim.request_flat().data_flat
	mim.layers_out_flat[shelf.layerID-1] = data

	for neuronID := range shelf.bias {
		neuronVal := shelf.bias[neuronID]

		for weightID, weight := range shelf.weights[neuronID] {
			neuronVal += data[weightID] * weight
		}
		mim.layers_out_flat_non_activated[shelf.layerID][neuronID] = neuronVal
		mim.layers_out_flat[shelf.layerID][neuronID] = shelf.act_interface.call(neuronVal)
	}
	mim.data_flat = &mim.layers_out_flat[shelf.layerID]

	mim.data_type = OneD
	mim.data_type_history[shelf.layerID] = OneD
}

func (shelf *DenseLayer) backprop(mim *MiM, prev_act_interface Activation) {
	//Grad = mim.request_data(), Gradiants is Cost/out not Cost/Act(Out)
	out_grade := *mim.request_flat().data_flat

	for neuronID := range out_grade {

		for weightID := range shelf.weights[neuronID] {
			shelf.weights_gradiants[neuronID][weightID] += out_grade[neuronID] * mim.layers_out_flat[shelf.layerID-1][weightID]
		}

		shelf.bias_gradiants[neuronID] += out_grade[neuronID]
	}

	if shelf.layerID > 1 { //First layerID == 1, behövr inte räkna ut nästa lagers: Cost/Out om vi är på första lagret.
		//MiM data_flat = nextLayerOut

		for prev_neuronID := 0; prev_neuronID < shelf.prev_layer_size; prev_neuronID++ {
			new_grade := 0.0

			for neuronID := 0; neuronID < shelf.size; neuronID++ {
				new_grade += shelf.weights[neuronID][prev_neuronID] * out_grade[neuronID]
			}
			new_grade = prev_act_interface.ddx(mim.layers_out_flat_non_activated[shelf.layerID-1][prev_neuronID]) * new_grade

			mim.layers_out_flat[shelf.layerID-1][prev_neuronID] = new_grade
		}

		mim.data_flat = &mim.layers_out_flat[shelf.layerID-1]
		mim.data_type = OneD
	}

}

func (shelf *DenseLayer) apply_gradients(learn_rate float64, batch_size float64) {
	for neuronID := range shelf.bias {
		shelf.bias[neuronID] -= shelf.bias_gradiants[neuronID] * learn_rate / batch_size
		shelf.bias_gradiants[neuronID] = 0

		for weightID := range shelf.weights[neuronID] {
			shelf.weights[neuronID][weightID] -= shelf.weights_gradiants[neuronID][weightID] * learn_rate / batch_size
			shelf.weights_gradiants[neuronID][weightID] = 0
		}
	}
}

func (shelf *DenseLayer) init(layerID int) {
	//Init all layer arrays sizes, Sets bias AND WEIGHTS to 0
	shelf.layer_type = "DenseLayer"
	shelf.layerID = layerID

	shelf.bias = make([]float64, shelf.size)
	shelf.weights = make([][]float64, shelf.size)

	shelf.bias_gradiants = make([]float64, shelf.size)
	shelf.weights_gradiants = make([][]float64, shelf.size)

	for i := 0; i < shelf.size; i++ {

		// neuroWeights := make([]float64, shelf.prev_layer_size)
		shelf.weights[i] = make([]float64, shelf.prev_layer_size)
		shelf.weights_gradiants[i] = make([]float64, shelf.prev_layer_size)

	}
}

func (shelf *DenseLayer) init_new_weights(xavierRange float64, r rand.Rand) {
	//Give each weights new random weights (Currentlu 0)

	for neuronID := range shelf.bias {
		shelf.bias[neuronID] = 0

		for weightID := range shelf.weights[neuronID] {

			shelf.weights[neuronID][weightID] = initWeightXavierUniform(xavierRange, r)
		}
	}
}

func (shelf *DenseLayer) load_weights(flat_weights []float64) {
	for i := 0; i < shelf.size; i++ {
		for i2 := 0; i2 < shelf.prev_layer_size; i2++ {
			shelf.weights[i][i2] = flat_weights[shelf.prev_layer_size*i+i2]
		}
	}

}
func (shelf *DenseLayer) load_biases(flat_bias []float64) {
	for i := 0; i < shelf.size; i++ {
		shelf.bias[i] = flat_bias[i]
	}
}
func (shelf *DenseLayer) get_weights() []float64 {
	flat_weights := make([]float64, shelf.prev_layer_size*shelf.size)

	for i := 0; i < shelf.size; i++ {
		for i2 := 0; i2 < shelf.prev_layer_size; i2++ {
			flat_weights[shelf.prev_layer_size*i+i2] = shelf.weights[i][i2]
		}
	}

	return flat_weights
}
func (shelf *DenseLayer) get_biases() []float64 {
	return shelf.bias
}

func (shelf *DenseLayer) print_weights() {
	for neuronID := range shelf.bias {
		fmt.Println(shelf.weights[neuronID])
	}

	//fmt.Println(shelf.bias)
}

func (shelf *DenseLayer) get_size() []int {
	return []int{shelf.size}
}
func (shelf *DenseLayer) get_name() string {
	return "DenseLayer"
}

func (shelf *DenseLayer) get_act_interface() Activation {
	return shelf.act_interface
}

func (shelf *DenseLayer) debug_print() {
	fmt.Println(shelf.layerID)
	fmt.Println(shelf.size)
}
