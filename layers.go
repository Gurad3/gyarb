package main

type Layers interface {
	forward()
}

type DenseLayer struct {
	act_interface Activation

	weights [][]float64
	bias    []float64
}

type CNLayer struct {
	act_interface Activation

	weights [][][]float64
	bias    [][]float64
}

func (shelf *DenseLayer) forward(mim MiM) {
	data := mim.request_flat().data_flat

	for neuronID := range shelf.bias {
		neuronVal := shelf.bias[neuronID]

		for weightID, weight := range shelf.weights[neuronID] {
			neuronVal += data[weightID] * weight
		}

		shelf.base.nonAktOut[neuronID] = neuronVal
		shelf.base.out[neuronID] = shelf.base.aktfunc(neuronVal)
	}
	return shelf.base.out, shelf.base.nonAktOut
}
