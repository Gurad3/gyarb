package main

import (
	"fmt"
	"math/rand"
	"sync"
	"time"
)

type trainer struct {
	TrainData      [][]float64
	TrainDataLabel [][]float64
	TestData       [][]float64
	TestDataLabel  [][]float64

	train_batches       [][][]float64
	train_label_batches [][][]float64

	batch_size        int
	info_milestone    int
	save_at_milestone bool
	epochs            int
}

func (shelf *Network) train_network(trainData trainer, threded bool) {

	for i := 0; i < len(trainData.TrainData); i += trainData.batch_size {
		end := i + trainData.batch_size
		if end > len(trainData.TrainData) {
			end = len(trainData.TrainData)
		}
		trainData.train_batches = append(trainData.train_batches, trainData.TrainData[i:end])
		trainData.train_label_batches = append(trainData.train_label_batches, trainData.TrainDataLabel[i:end])
	}

	totalSamples := 0
	mim := new(MiM)
	mim.init(shelf)

	if threded {
		mimArray := make([]MiM, trainData.batch_size)
		for i := 0; i < trainData.batch_size; i++ {
			mimArray[i].init(shelf)
		}
		var wg sync.WaitGroup
		b := 0
		for b < 2 {
			for batchID, batch := range trainData.train_batches {
				wg.Add(trainData.batch_size)
				for sampleID, sample := range batch {
					go shelf.tmpRun(&mimArray[sampleID], sample, trainData.train_label_batches[batchID][sampleID], &wg)

					if totalSamples%trainData.info_milestone == 0 {
						shelf.Test(mim, trainData.TestData, trainData.TestDataLabel)

						b++

						if trainData.save_at_milestone {
							encode_to_json(shelf)
						}
					}
					totalSamples++

				}
				wg.Wait()
				shelf.apply_gradients(trainData.batch_size)
				if b == 2 {
					break
				}
			}

			trainData.shuffle_batches()

		}
		wg.Wait()
	} else {
		for {
			for batchID, batch := range trainData.train_batches {
				for sampleID, sample := range batch {
					shelf.forward(mim, sample)
					// shelf.print_weights()
					// fmt.Println(mim.data_flat)
					shelf.backprop(mim, trainData.train_label_batches[batchID][sampleID])
					if totalSamples%trainData.info_milestone == 0 {
						shelf.Test(mim, trainData.TestData, trainData.TestDataLabel)
						//fmt.Println("Epoch")
						if trainData.save_at_milestone {
							encode_to_json(shelf)
						}
					}
					totalSamples++
				}
				shelf.apply_gradients(trainData.batch_size)
			}
			trainData.shuffle_batches()
		}
	}

}

func (shelf *Network) tmpRun(mim *MiM, sample []float64, label []float64, wg *sync.WaitGroup) {
	defer wg.Done()
	shelf.forward(mim, sample)
	// shelf.print_weights()
	// fmt.Println(mim.data_flat)
	shelf.backprop(mim, label)

}

func (shelf *Network) Test(mim *MiM, TestData [][]float64, TestLabels [][]float64) float64 {
	// return 0
	totalCost := 0.0

	correct_choises := 0
	for sampleID, sample := range TestData {
		shelf.forward(mim, sample)
		totalCost += shelf.cost_interface.call(*mim.data_flat, TestLabels[sampleID])

		// fmt.Println("--")
		// fmt.Println(&sample, *mim.data_flat, TestLabels[sampleID], isCorrect(*mim.data_flat, TestLabels[sampleID]))
		// fmt.Println("--")
		if isCorrect(*mim.data_flat, TestLabels[sampleID]) {
			correct_choises += 1
		}
	}
	// fmt.Println(*mim.data_flat, TestLabels[len(TestLabels)-1], isCorrect(*mim.data_flat, TestLabels[len(TestLabels)-1]))
	fmt.Println("Percantage Correct on Test Data: ", float64(correct_choises)/float64(len(TestLabels)))
	fmt.Println("Cost: ", totalCost/float64(len(TestLabels)))

	//shelf.layers[0].debug_print()

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

func (shelf *trainer) shuffle_batches() {
	rng := rand.New(rand.NewSource(time.Now().Unix()))
	for batch_index := range shelf.train_batches {
		rand_index := rng.Int() % len(shelf.train_batches)

		shelf.train_batches[batch_index], shelf.train_batches[rand_index] = shelf.train_batches[rand_index], shelf.train_batches[batch_index]
		shelf.train_label_batches[batch_index], shelf.train_label_batches[rand_index] = shelf.train_label_batches[rand_index], shelf.train_label_batches[batch_index]
	}
}
