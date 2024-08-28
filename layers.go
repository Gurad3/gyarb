package main

type Layers interface {
	forward()
}

type DenseLayer struct {
	act_interface Activation
	layerID       int

	weights [][]float64
	bias    []float64
}

type CNLayer struct {
	act_interface Activation
	layerID       int

	weights [][][]float64
	bias    [][]float64
}

func (shelf *DenseLayer) forward(mim MiM) {
	data := *mim.request_flat().data_flat

	for neuronID := range shelf.bias {
		neuronVal := shelf.bias[neuronID]

		for weightID, weight := range shelf.weights[neuronID] {
			neuronVal += data[weightID] * weight
		}

		mim.layers_out_flat_non_activated[shelf.layerID][neuronID] = neuronVal
		mim.layers_out_flat[shelf.layerID][neuronID] = shelf.act_interface.call(neuronVal)

	}
}
