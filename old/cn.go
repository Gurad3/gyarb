package main

import "fmt"

type CNLayer struct {
	base BaseLayer

	kernelSize int
	strideLen  int
	padding    int
	inpFetures int

	filterCount int
	weights     [][][]float64
	//filters, feature inp/prev,  kernal,
	//varje filter = ny feture nästa
	//ny feature = sum(feture inp*feture wie) + bias
	//
	bias [][]float64
	//filters //EachOut

	biasAdd    [][]float64
	weightsAdd [][][]float64

	OutSizeX      int
	OutSizeY      int
	inSizeX       int
	inSizeY       int
	outFetureSize int
}

func (self *CNLayer) printDebug() {
	fmt.Println(self.weights)
}

func (self *CNLayer) initLayer(prevLayerSize int, xavierRange float64, initData initLayerData) {

	self.kernelSize = initData.cnKernlSize
	self.strideLen = initData.cnStride
	self.padding = initData.cnPadding
	self.filterCount = initData.cnFillterCount

	self.inpFetures = initData.cnInpFetures

	self.inSizeX = initData.cnInSizeX
	self.inSizeY = initData.cnInSizeY

	self.OutSizeX = (self.inSizeX-self.kernelSize+2*self.padding)/self.strideLen + 1
	self.OutSizeY = (self.inSizeY-self.kernelSize+2*self.padding)/self.strideLen + 1

	self.base.OutSize = self.OutSizeX * self.OutSizeY * self.filterCount
	self.outFetureSize = self.OutSizeX * self.OutSizeY

	self.base.nonAktOut = make([]float64, self.base.OutSize)
	self.base.out = make([]float64, self.base.OutSize)

	self.bias = make([][]float64, self.filterCount)
	self.biasAdd = make([][]float64, self.filterCount)

	self.weights = make([][][]float64, self.filterCount)
	self.weightsAdd = make([][][]float64, self.filterCount)

	for filterID := 0; filterID < self.filterCount; filterID++ {

		self.bias[filterID] = make([]float64, self.outFetureSize)
		self.biasAdd[filterID] = make([]float64, self.outFetureSize)

		neWeights := make([][]float64, self.inpFetures)
		newAdd := make([][]float64, self.inpFetures)

		for fetureID := 0; fetureID < self.inpFetures; fetureID++ {
			newFetureWeights := make([]float64, self.kernelSize*self.kernelSize)
			newFetureAdd := make([]float64, self.kernelSize*self.kernelSize)

			for weightID := 0; weightID < self.kernelSize*self.kernelSize; weightID++ {
				newFetureWeights[weightID] = initWeightXavierUniform(xavierRange)
				newFetureAdd[weightID] = 0
			}

			neWeights[fetureID] = newFetureWeights
			newAdd[fetureID] = newFetureAdd
		}
		self.weights[filterID] = neWeights
		self.weightsAdd[filterID] = newAdd
		//self.weightsAdd = append(self.weightsAdd, make([]float64, prevLayerSize))
	}

	self.base.aktfunc = getAktFuncFromName(false, initData.aktFunName)
	self.base.aktDevFunc = getAktFuncFromName(true, initData.aktFunName)
	self.base.aktFunName = initData.aktFunName

}

func (self *CNLayer) forWard(data []float64) ([]float64, []float64) {
	for filterID := 0; filterID < self.filterCount; filterID++ {
		outFilterOffset := filterID * self.outFetureSize

		for outY := 0; outY < self.OutSizeY; outY++ {
			for outX := 0; outX < self.OutSizeX; outX++ {

				outIndex := outY*self.OutSizeY + outX
				outNodeVal := self.bias[filterID][outIndex]

				for fetureID := 0; fetureID < self.inpFetures; fetureID++ {
					currentFetureVal := float64(0)
					inOffset := fetureID * self.inSizeX * self.inSizeY
					for kY := 0; kY < self.kernelSize; kY++ {
						for kX := 0; kX < self.kernelSize; kX++ {

							xOff := outX * self.strideLen
							yOff := outY * self.strideLen
							currInpVal := data[inOffset+yOff*self.inSizeY+xOff]
							currWeight := self.weights[filterID][fetureID][kY*self.kernelSize+kX]

							currentFetureVal += currWeight * currInpVal

						}
					}

					outNodeVal += currentFetureVal
				}

				self.base.nonAktOut[outFilterOffset+outIndex] = outNodeVal
				self.base.out[outFilterOffset+outIndex] = self.base.aktfunc(outNodeVal)
			}
		}

	}
	return self.base.out, self.base.nonAktOut
}

func (self *CNLayer) updateGrad(OutputGrad []float64, prevNoneOut []float64) {
	//fmt.Println("d")
}

func (self *CNLayer) backProp(OutputGrad []float64, prevOut []float64) []float64 {

	return []float64{}
}

func (self *CNLayer) applyGrad(batchSize float64, lRate float64) {

}
func (self *CNLayer) lastLayerbackProp(targets []float64) []float64 {
	return []float64{}
}
