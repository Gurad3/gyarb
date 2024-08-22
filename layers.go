package main

type Layers interface {
	forward()
}

type BaseLayer struct {
	act_func func(float64) float64
}

type DenseLayer struct {
	base_layer BaseLayer
}

type CNLayer struct {
	base_layer BaseLayer
}
