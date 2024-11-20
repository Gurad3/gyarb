package main

import (
	"math/rand"
)

type Layer interface {
	forward(*MiM)                                      // Har
	backprop(mim *MiM, prev_layer_act Activation)      // Har
	init(layerID int, prev_layer_size []int)           // Har
	init_new_weights(xavierRange float64, r rand.Rand) // Har
	apply_gradients(learn_rate float64, batch_size int, regularization float64, momentum float64)

	load_weights(flat_weights []float64)
	load_biases(flat_biases []float64)
	get_weights() []float64
	get_biases() []float64

	print_weights()
	get_size() []int
	get_name() string
	get_init_vals() []int

	get_act_interface() Activation

	debug_print()
}
