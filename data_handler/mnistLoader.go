package data_handler

import (
	"os"
)

func Load_mnist() ([][]float64, [][]float64, [][]float64, [][]float64) {

	folderLocation := "./data_handler/mnist/"
	TrainData := loadImageFile(folderLocation+"train-images.idx3-ubyte", 60_000, 16)
	TestData := loadImageFile(folderLocation+"t10k-images.idx3-ubyte", 10_000, 16)

	TrainDataLabel := loadLabelFile(folderLocation+"train-labels.idx1-ubyte", 60_000, 8)
	TestDataLabel := loadLabelFile(folderLocation+"t10k-labels.idx1-ubyte", 10_000, 8)
	return TrainDataLabel, TrainData, TestDataLabel, TestData
}

func Load_emnist_letters() ([][]float64, [][]float64, [][]float64, [][]float64) {

	folderLocation := "./data_handler/big/emnistLetters/"
	TrainData := loadImageFile(folderLocation+"emnist-letters-train-images-idx3-ubyte", 88_800, 16)
	TestData := loadImageFile(folderLocation+"emnist-letters-test-images-idx3-ubyte", 14_800, 16)

	TrainDataLabel := loadLabelFile(folderLocation+"emnist-letters-train-labels-idx1-ubyte", 88_800, 8)
	TestDataLabel := loadLabelFile(folderLocation+"emnist-letters-test-labels-idx1-ubyte", 14_800, 8)

	return TrainDataLabel, TrainData, TestDataLabel, TestData
}

func check(e error) {
	if e != nil {
		panic(e)
	}
}

func loadImageFile(file string, size int, bsBytes int) [][]float64 {
	content, err := os.ReadFile(file)
	check(err)

	newArray := make([][]float64, size)

	for img := 0; img < size; img++ {
		newIMG := make([]float64, 28*28)
		for i := 0; i < 28*28; i++ {
			newIMG[i] = float64(content[bsBytes+img*28*28+i]) / 255
		}
		newArray[img] = newIMG
	}

	return newArray
}

func loadLabelFile(file string, size int, bsBytes int) [][]float64 {
	content, err := os.ReadFile(file)
	check(err)

	newArray := make([][]float64, size)

	for i := 0; i < size; i++ {
		newArray[i] = getArrayFromNum(int(content[bsBytes+i]))
	}

	return newArray
}

func getArrayFromNum(val int) []float64 {
	newArray := make([]float64, 10)
	// newArray := make([]float64, 37)
	newArray[val] = 1
	return newArray
}
