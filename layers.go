package main

type Layers interface {
	forward()
}

type DenseLayer struct {
}

type CNLayer struct {
}
