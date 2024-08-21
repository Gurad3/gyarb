package main

type Network struct {
	layers []Layers

	learn_rate       float64
	learn_rate_decay float64
	momentum         float64
}
