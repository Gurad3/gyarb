package main

import "fmt"

var TrainData [][]float64
var TrainDataLabel [][]float64
var TestData [][]float64
var TestDataLabel [][]float64

type initLayerData struct {
	typ        string
	aktFunName string

	//MLP
	size int

	//CN
	cnPadding      int
	cnKernlSize    int
	cnFillterCount int
	cnStride       int

	//First ConvLayer
	cnInSizeX    int
	cnInpFetures int

	//Ignore

	layerIN   int
	cnInSizeY int
}

type initNetData struct {
	inpSize   int
	learnRate float64
	fileName  string
}

var NetworksFolder string = "./NetWorks/"

func main() {
	loadMnist()
	newNetData := initNetData{
		inpSize:   28 * 28, // If fetures, then add in first cnLayer inpFetures, If ConvNet, add xSize in firstConvLayer
		learnRate: float64(.2),
		fileName:  "First",
	}

	newLayerData := []initLayerData{
		/*
			{
				typ:            "CN",
				aktFunName:     "reLU",
				cnPadding:      0,
				cnKernlSize:    3,
				cnFillterCount: 4,
				cnStride:       1,

				cnInpFetures: 1,
				cnInSizeX:    28,
				cnInSizeY:    28,
			},
		*/
		{
			size:       100,
			typ:        "MLP",
			aktFunName: "reLU",
		},

		{
			size:       10,
			typ:        "MLP",
			aktFunName: "sigmoid",
		},
	}

	net := initNet(newNetData, newLayerData)
	fmt.Println(net.layers[0].getOutSize())
	fmt.Println(net.layers[1].getOutSize())
	//data := make([]float64, 28*28)

	net.TrainSplit(TrainData, TrainDataLabel, TestData, TestDataLabel)
}
