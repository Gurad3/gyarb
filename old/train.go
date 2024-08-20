package main

import (
	"fmt"
)

func (net *netWork) TrainSplit(data [][]float64, expectedData [][]float64, testData [][]float64, exTest [][]float64) {
	batchSize := 100
	i := 0
	fmt.Println(net.runNet(data[0]))
	for true {
		for dataIndex, dataSlice := range data {

			net.Train(dataSlice, expectedData[dataIndex])

			if i%batchSize == 0 {
				net.applyGrad(float64(batchSize))
			}
			//net.applyGrad(float64(1))
			if i%(batchSize*500) == 0 {

				fmt.Println("Performance on 100 first TrainData: ", net.TestPerformance(data, expectedData, 100))
				fmt.Println("Performance on Test Data", net.TestPerformance(testData, exTest, len(testData)))

				//net.SaveToFile()
			}
			i++
		}
	}
}

func (net *netWork) TestPerformance(data [][]float64, expectedData [][]float64, ln int) float64 {
	score := 0
	i := 0
	for true {
		for dataIndex, dataSlice := range data {
			if i >= ln {
				return float64(score) / float64(ln)
			}
			out := net.runNet(dataSlice)
			if checkAnswer(out, expectedData[dataIndex]) {
				score += 1
			}
			i++
		}
	}
	return 0
}

func checkAnswer(out []float64, ex []float64) bool {
	right := 0
	for pos, val := range ex {
		if val == 1 {
			right = pos
			break
		}
	}

	p := 0
	hig := float64(0)
	for i, val := range out {
		if val > hig {
			p = i
			hig = val
		}
	}
	return p == right
}

func (net *netWork) Train(data []float64, expectedData []float64) {
	net.runNet(data)
	net.backProp(expectedData)
	//Run net.ApplayGrad, dock efter hela batchen
}

func (self *netWork) applyGrad(batchSize float64) {
	for _, currLayer := range self.layers {

		currLayer.applyGrad(batchSize, self.learnRate)
	}
}
