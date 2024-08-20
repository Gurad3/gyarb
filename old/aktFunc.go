package main

import "math"

var aktNameMap = map[string][]func(v float64) float64{
	"sigmoid": {sigmoid, devSigmoid},
	"reLU":    {reLU, devReLU},
}

func getAktFuncFromName(dev bool, name string) func(v float64) float64 {
	typ := 0
	if dev {
		typ = 1
	}
	return aktNameMap[name][typ]
}

func sigmoid(val float64) float64 {
	return 1 / (1 + math.Exp(-val))
}

func devSigmoid(val float64) float64 {
	akt := sigmoid(val)
	return akt * (1 - akt)
}

func reLU(val float64) float64 {
	return math.Max(val, 0)
}

func devReLU(val float64) float64 {
	if val <= 0 {
		return 0
	}
	return 1
}

func getCost(neuronVal float64, targetVal float64) float64 {
	er := neuronVal - targetVal
	return er * er
	//return math.Pow(neuronVal-targetVal, 2)
}

func devGetCost(neuronVal float64, targetVal float64) float64 {
	return 2 * (neuronVal - targetVal)
}

func getCostOutPut(netOut []float64, target []float64) float64 {
	cost := float64(0)
	for i, val := range netOut {
		cost += getCost(val, target[i])
	}
	return cost
}

/*
func (net *netWork) getCostBatch(data [][]float64, expected [][]float64) float64 {
	totalCost := float64(0)
	for i, inputs := range data {
		out := net.runNet(inputs)
		totalCost += getCostOutPut(out, expected[i])
	}

	return totalCost / float64(len(data))
}
*/
