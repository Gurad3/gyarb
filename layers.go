package main

import "fmt"

type Layer interface {
	forward(*MiM)
	init(layerID int)
	init_new_weights(xavierRange float64)
	print_weights()
	get_size() []int
}

type DenseLayer struct {
	act_interface   Activation
	layerID         int
	size            int
	prev_layer_size int

	Weights [][]float64 `json:"weights"`
	Bias    []float64   `json:"bias"`

	weights_gradiants [][]float64
	bias_gradiants    []float64
}

type CNLayer struct {
	act_interface Activation
	layerID       int
	size          []int

	weights [][][]float64
	bias    [][]float64

	weights_gradiants [][][]float64
	bias_gradiants    [][]float64
}

func (shelf *DenseLayer) forward(mim *MiM) {
	data := *mim.request_flat().data_flat

	for neuronID := range shelf.Bias {
		neuronVal := shelf.Bias[neuronID]

		for weightID, weight := range shelf.Weights[neuronID] {
			neuronVal += data[weightID] * weight
		}
		mim.layers_out_flat_non_activated[shelf.layerID][neuronID] = neuronVal
		mim.layers_out_flat[shelf.layerID][neuronID] = shelf.act_interface.call(neuronVal)
	}
	mim.data_flat = &mim.layers_out_flat[shelf.layerID]
}

func (shelf *DenseLayer) init(layerID int) {
	//Init all layer arrays sizes, Set weights to 0

	shelf.Bias = make([]float64, shelf.size)
	shelf.Weights = make([][]float64, shelf.size)

	shelf.bias_gradiants = make([]float64, shelf.size)
	shelf.weights_gradiants = make([][]float64, shelf.size)

	for i := 0; i < shelf.size; i++ {

		neuroWeights := make([]float64, shelf.prev_layer_size)
		shelf.Weights[i] = neuroWeights
		shelf.weights_gradiants[i] = neuroWeights

	}
}

func (shelf *DenseLayer) init_new_weights(xavierRange float64) {
	//Give each weights new random weights (Currentlu 0)

	for neuronID := range shelf.Bias {
		shelf.Bias[neuronID] = 0

		for weightID := range shelf.Weights[neuronID] {

			shelf.Weights[neuronID][weightID] = initWeightXavierUniform(xavierRange)
		}
	}
}

func (shelf *DenseLayer) print_weights() {
	for neuronID := range shelf.Bias {

		fmt.Println(shelf.Weights[neuronID])

	}
}

func (shelf *DenseLayer) get_size() []int {
	return []int{shelf.size}
}
