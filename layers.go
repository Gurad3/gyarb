package main

type Layers interface {
	forward()
}

type BaseLayer struct {
	act_interface Activation
}

type DenseLayer struct {
	weights [][]float64
	biases  []float64
}

type CNLayer struct {
	weights [][][]float64
	biases  [][]float64
}
