package main

import (
	"math"
	"math/rand"
	"time"
)

type netWork struct {
	learnRate   float64
	inp         []float64
	sizes       []int
	layers      []Layer
	fileName    string
	OutSize     int
	outArray    [][]float64
	nonOutArray [][]float64
	numLayers   int
}

type BaseLayer struct {
	aktfunc    func(val float64) float64
	aktDevFunc func(val float64) float64
	aktFunName string

	out       []float64
	nonAktOut []float64
	OutSize   int
}

type Layer interface {
	forWard([]float64) ([]float64, []float64)
	initLayer(int, float64, initLayerData)
	getOutSize() int
	getOutDim() (int, int)
	printDebug()
	updateGrad([]float64, []float64)

	lastLayerbackProp([]float64) []float64
	backProp(oldNodeVals []float64, noneprevOut []float64) []float64
	applyGrad(batchSize float64, lRate float64)
}

func (self *MLPLayer) getOutSize() int {
	return self.base.OutSize
}
func (self *CNLayer) getOutSize() int {
	return self.base.OutSize
}

func (self *MLPLayer) getOutDim() (int, int) {
	return self.base.OutSize, 1
}
func (self *CNLayer) getOutDim() (int, int) {
	return self.OutSizeX, self.OutSizeY
}

func initWeightXavierUniform(xavierRange float64) float64 {
	return rand.Float64()*2*xavierRange - xavierRange
}

func initNet(netData initNetData, layersData []initLayerData) *netWork {
	net := new(netWork)
	//net.inp = make([]float64, inpSize)

	net.learnRate = netData.learnRate
	net.fileName = netData.fileName
	net.numLayers = len(layersData)
	net.outArray = make([][]float64, net.numLayers)
	net.nonOutArray = make([][]float64, net.numLayers)

	rand.Seed(time.Now().UnixNano())
	xavierRange := math.Sqrt(6 / float64(netData.inpSize+layersData[net.numLayers-1].size))

	for layerID, layerData := range layersData {
		prevSize := netData.inpSize
		if layerID != 0 {
			prevSize = net.layers[layerID-1].getOutSize()
		}

		var newLayer Layer
		if layerData.typ == "MLP" {
			newLayer = new(MLPLayer)
		}

		if layerData.typ == "CN" {
			newLayer = new(CNLayer)

			if layerID != 0 && layerData.cnInpFetures == 0 {
				layerData.cnInpFetures = layersData[layerID-1].cnFillterCount
				layerData.cnInSizeX, layerData.cnInSizeY = net.layers[layerID-1].getOutDim()
			}
		}

		newLayer.initLayer(prevSize, xavierRange, layerData)

		net.layers = append(net.layers, newLayer)
	}

	net.OutSize = net.layers[len(net.layers)-1].getOutSize()
	return net
}

func (self *netWork) runNet(data []float64) []float64 {
	self.inp = data

	var curVals = data

	var curNonVals []float64
	for layerID, layer := range self.layers {

		//fmt.Println(curVals)
		curVals, curNonVals = layer.forWard(curVals)
		self.outArray[layerID] = curVals
		self.nonOutArray[layerID] = curNonVals
	}

	//fmt.Println(curVals)
	return curVals
}
