package main

type Layers interface {
	forward()
}

type DenseLayer struct {
	act_interface Activation

	weights [][]float64
	biases  []float64
}

type CNLayer struct {
	act_interface Activation

	weights [][][]float64
	biases  [][]float64
}
