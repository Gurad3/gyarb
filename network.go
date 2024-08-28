package main

type Network struct {
	layers []Layer

	learn_rate       float64
	learn_rate_decay float64
	momentum         float64
}

func (shelf *Network) init() {
	for layerID, layer := range shelf.layers {
		layer.init(layerID)
	}
}
