package main

import (
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

type ConvLayer struct {
	act_interface Activation
	layerID       int
	size          []int
	layer_type    string

	weights [][][]float64
	bias    [][]float64

	weights_gradiants [][][]float64
	bias_gradiants    [][]float64

	kernal_size  int
	kernal_depth int
	input_depth  int
}
