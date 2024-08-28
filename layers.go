package main

type Layer interface {
	forward(*MiM)
	init(layerID int)
	init_new_weights()
}

type DenseLayer struct {
	act_interface   Activation
	layerID         int
	size            int
	prev_layer_size int

	weights [][]float64
	bias    []float64

	weights_gradiants [][]float64
	bias_gradiants    []float64
}

type CNLayer struct {
	act_interface Activation
	layerID       int
	size          []int

	weights [][][]float64
	bias    [][]float64

	weights_gradiants [][][]float64
	bias_gradiants    [][]float64
}

func (shelf *DenseLayer) forward(mim *MiM) {
	data := *mim.request_flat().data_flat

	for neuronID := range shelf.bias {
		neuronVal := shelf.bias[neuronID]

		for weightID, weight := range shelf.weights[neuronID] {
			neuronVal += data[weightID] * weight
		}
		mim.layers_out_flat_non_activated[shelf.layerID][neuronID] = neuronVal
		mim.layers_out_flat[shelf.layerID][neuronID] = shelf.act_interface.call(neuronVal)
	}
	mim.data_flat = &mim.layers_out_flat[shelf.layerID]
}

func (shelf *DenseLayer) init(layerID int) {
	//Init all layer arrays sizes, Set weights to 0

	shelf.bias = make([]float64, shelf.size)
	shelf.weights = make([][]float64, shelf.size)

	shelf.bias_gradiants = make([]float64, shelf.size)
	shelf.weights_gradiants = make([][]float64, shelf.size)

	for i := 0; i < shelf.size; i++ {
		neuroWeights := make([]float64, shelf.prev_layer_size)
		shelf.weights = append(shelf.weights, neuroWeights)
		shelf.weights_gradiants = append(shelf.weights_gradiants, neuroWeights)
	}
}

func (shelf *DenseLayer) init_new_weights() {
	//Give each weights new random weights (Currentlu 0)

	//xavierRange := math.Sqrt(6 / float64(netData.inpSize+layersData[net.numLayers-1].size))

	for neuronID := range shelf.bias {
		shelf.bias[neuronID] = 0

		for weightID := range shelf.weights[neuronID] {
			shelf.weights[neuronID][weightID] = initWeightXavierUniform(xavierRange)
		}
	}
}
