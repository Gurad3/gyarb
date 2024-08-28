package main

type Network struct {
	layers []Layer

	learn_rate       float64
	learn_rate_decay float64
	momentum         float64

	cost_interface Cost
}

func (shelf *Network) init() {
	for layerID, layer := range shelf.layers {
		layer.init(layerID)
	}
}

func (shelf *Network) init_new_weights() {
	for _, layer := range shelf.layers {
		layer.init_new_weights()
	}
}
