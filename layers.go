package main

import "fmt"

type Layer interface {
	forward(*MiM)
	init(layerID int)
	init_new_weights(xavierRange float64)

	load_weights(flat_weights []float64)
	load_biases(flat_biases []float64)
	get_weights() []float64
	get_biases() []float64

	print_weights()
	get_size() []int
	get_name() string
	get_act_name() string
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

	for neuronID := range shelf.bias {
		neuronVal := shelf.bias[neuronID]

		for weightID, weight := range shelf.weights[neuronID] {
			neuronVal += data[weightID] * weight
		}
		mim.layers_out_flat_non_activated[shelf.layerID][neuronID] = neuronVal
		mim.layers_out_flat[shelf.layerID][neuronID] = shelf.act_interface.call(neuronVal)
	}
	mim.data_flat = &mim.layers_out_flat[shelf.layerID]
}

func (shelf *DenseLayer) init(layerID int) {
	//Init all layer arrays sizes, Sets bias AND WEIGHTS to 0
	shelf.layer_type = "DenseLayer"

	shelf.bias = make([]float64, shelf.size)
	shelf.weights = make([][]float64, shelf.size)

	shelf.bias_gradiants = make([]float64, shelf.size)
	shelf.weights_gradiants = make([][]float64, shelf.size)

	for i := 0; i < shelf.size; i++ {

		neuroWeights := make([]float64, shelf.prev_layer_size)
		shelf.weights[i] = neuroWeights
		shelf.weights_gradiants[i] = neuroWeights

	}
}

func (shelf *DenseLayer) init_new_weights(xavierRange float64) {
	//Give each weights new random weights (Currentlu 0)

	for neuronID := range shelf.bias {
		shelf.bias[neuronID] = 0

		for weightID := range shelf.weights[neuronID] {

			shelf.weights[neuronID][weightID] = initWeightXavierUniform(xavierRange)
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
}

func (shelf *DenseLayer) get_size() []int {
	return []int{shelf.size}
}
func (shelf *DenseLayer) get_name() string {
	return "DenseLayer"
}
func (shelf *DenseLayer) get_act_name() string {
	return shelf.act_interface.get_name()
}
