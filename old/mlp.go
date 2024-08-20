package main

import (
	"fmt"
)

type MLPLayer struct {
	base       BaseLayer
	bias       []float64
	weights    [][]float64
	biasAdd    []float64
	weightsAdd [][]float64
	size       int
}

func (self *MLPLayer) printDebug() {
	fmt.Println(self.bias)
}
func (self *MLPLayer) forWard(data []float64) ([]float64, []float64) {

	for neuronID := range self.bias {
		neuronVal := self.bias[neuronID]

		for weightID, weight := range self.weights[neuronID] {

			neuronVal += data[weightID] * weight
		}

		self.base.nonAktOut[neuronID] = neuronVal
		self.base.out[neuronID] = self.base.aktfunc(neuronVal)
	}
	return self.base.out, self.base.nonAktOut
}

func (self *MLPLayer) initLayer(prevLayerSize int, xavierRange float64, initData initLayerData) {

	layerSize := initData.size
	self.size = layerSize
	self.base.OutSize = layerSize

	self.base.nonAktOut = make([]float64, layerSize)
	self.base.out = make([]float64, layerSize)

	self.bias = make([]float64, layerSize)

	self.biasAdd = make([]float64, layerSize)

	for neuronID := 0; neuronID < layerSize; neuronID++ {
		neWeights := make([]float64, prevLayerSize)

		for weightID := 0; weightID < prevLayerSize; weightID++ {
			neWeights[weightID] = initWeightXavierUniform(xavierRange)
		}
		self.weights = append(self.weights, neWeights)
		self.weightsAdd = append(self.weightsAdd, make([]float64, prevLayerSize))
	}

	self.base.aktfunc = getAktFuncFromName(false, initData.aktFunName)
	self.base.aktDevFunc = getAktFuncFromName(true, initData.aktFunName)
	self.base.aktFunName = initData.aktFunName

}

func (self *MLPLayer) applyGrad(batchSize float64, lRate float64) {

	mult := lRate / batchSize
	for neuronID, neuronWeights := range self.weights {

		for weightID := range neuronWeights {
			neuronWeights[weightID] -= self.weightsAdd[neuronID][weightID] * mult

			self.weightsAdd[neuronID][weightID] = 0
		}

		self.bias[neuronID] -= self.biasAdd[neuronID] * mult
		self.biasAdd[neuronID] = 0
	}
}

func (self *MLPLayer) updateGrad(noneOutputGrad []float64, prevLayAkt []float64) {

	for neuronID := range noneOutputGrad {

		for weightID := range self.weights[neuronID] {
			//fmt.Println(len(lay.weightsAdd), len(nodeVals), len(prevLayAkt))
			//	fmt.Println(prevLayAkt)
			//fmt.Println(*prevLayAkt)
			self.weightsAdd[neuronID][weightID] += noneOutputGrad[neuronID] * prevLayAkt[weightID]
		}

		self.biasAdd[neuronID] += noneOutputGrad[neuronID]
	}
}

func (self *MLPLayer) backProp(noneOutputGrad []float64, prevNoneOut []float64) []float64 {

	prevOutputGrad := make([]float64, len(prevNoneOut))

	for prevGradID := 0; prevGradID < len(prevNoneOut); prevGradID++ {
		val := float64(0)

		for gradID, gradVal := range noneOutputGrad {
			val += gradVal * self.weights[gradID][prevGradID]
		}
		//PrevNoneOut or prevOut här under???
		prevOutputGrad[prevGradID] = val * self.base.aktDevFunc(prevNoneOut[prevGradID])
	}

	return prevOutputGrad
}

func (net *netWork) backProp(target []float64) {

	outLayer := net.layers[net.numLayers-1]
	noneOutputGrad := outLayer.lastLayerbackProp(target)
	//Not running on last
	for layerID := net.numLayers - 1; layerID > 0; layerID-- {

		prevOut := net.outArray[layerID-1]
		prevNoneOut := net.nonOutArray[layerID-1]

		net.layers[layerID].updateGrad(noneOutputGrad, prevOut)

		noneOutputGrad = net.layers[layerID].backProp(noneOutputGrad, prevNoneOut)

	}
	// Cuase we dont need to calc outgrad on last layer

	net.layers[0].updateGrad(noneOutputGrad, net.inp)
}

func (self *MLPLayer) lastLayerbackProp(targets []float64) []float64 {
	nodeValues := make([]float64, self.base.OutSize)

	for neuronID := 0; neuronID < self.base.OutSize; neuronID++ {
		addVal := devGetCost(self.base.out[neuronID], targets[neuronID]) * self.base.aktDevFunc(self.base.nonAktOut[neuronID])
		nodeValues[neuronID] = addVal
	}

	return nodeValues
}
