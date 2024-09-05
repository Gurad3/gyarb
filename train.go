package main

import "fmt"

type trainer struct {
	TrainData      [][]float64
	TrainDataLabel [][]float64
	TestData       [][]float64
	TestDataLabel  [][]float64

	train_batches       [][][]float64
	train_label_batches [][][]float64

	batch_size     int
	info_milestone int
	epochs         int
}

func (shelf *Network) train_network(mim *MiM, trainData trainer) {

	for i := 0; i < len(trainData.TrainData); i += trainData.batch_size {
		end := i + trainData.batch_size

		if end > len(trainData.TrainData) {
			end = len(trainData.TrainData)
		}
		trainData.train_batches = append(trainData.train_batches, trainData.TrainData[i:end])
		trainData.train_label_batches = append(trainData.train_label_batches, trainData.TrainDataLabel[i:end])
	}

	totalSamples := 0

	for {
		for batchID, batch := range trainData.train_batches {
			for sampleID, sample := range batch {
				shelf.forward(mim, sample)

				// shelf.print_weights()
				// fmt.Println(mim.data_flat)

				shelf.backprop(mim, trainData.train_label_batches[batchID][sampleID])
				if totalSamples%trainData.info_milestone == 0 {
					shelf.Test(mim, trainData.TestData, trainData.TestDataLabel)
				}
			}

			shelf.apply_gradients(trainData.batch_size)

		}
	}

}

func (shelf *Network) Test(mim *MiM, TestData [][]float64, TestLabels [][]float64) float64 {
	totalCost := 0.0

	correct_choises := 0
	for sampleID, sample := range TestData {
		shelf.forward(mim, sample)
		totalCost += shelf.cost_interface.call(*mim.request_flat().data_flat, TestLabels[sampleID])

		if isCorrect(*mim.data_flat, TestLabels[sampleID]) {
			correct_choises += 1
		}
	}
	fmt.Println(*mim.data_flat, TestLabels[1], isCorrect(*mim.data_flat, TestLabels[1]))
	fmt.Println("Percantage Correct on Test Data: ", float64(correct_choises)/float64(len(TestLabels)))
	fmt.Println("Cost: ", totalCost/float64(len(TestLabels)))

	return totalCost / float64(len(TestLabels))
}

func isCorrect(values []float64, target_values []float64) bool {
	higT := 0
	for id, v := range target_values {
		if v == 1 {
			higT = id
			break
		}
	}
	highV := 0.0
	highVID := 0
	for id, v := range values {
		if v > highV {
			highVID = id
			highV = v
		}
	}

	// fmt.Print("--")
	// fmt.Println(higT == highVID, values, target_values)
	// fmt.Print("--")
	return higT == highVID
}
