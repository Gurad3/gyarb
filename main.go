package main

import (
	"fmt"
)

func main() {
	net := tmpNewNet()
	// net.print_weights()

	fmt.Println("-")

	//net := load_from_json("saves/TrainedMNIST.json")
	//net.print_weights()

	mim := new(MiM)
	mim.init(net)

	MNIST_TrainDataLabel, MNIST_TrainData, MNIST_TestDataLabel, MNIST_TestData := loadMnist()

	td := trainer{
		TrainData:      MNIST_TrainData,
		TestData:       MNIST_TestData,
		TrainDataLabel: MNIST_TrainDataLabel,
		TestDataLabel:  MNIST_TestDataLabel,

		batch_size:     10,
		info_milestone: 4000,
	}
	net.train_network(mim, td)

	//_, _, _, TestData := loadMnist()
	//net.forward(mim, TestData[1])

	//encode_to_json(net)
}

func tmpNewNet() *Network {
	net := new(Network)

	net.learn_rate = 2
	net.learn_rate_decay = 0.0001
	net.file_name = "TestNet"
	net.cost_interface = &MeanSquare{}

	net.input_size = 28 * 28
	net.output_size = 10
	net.layers = []Layer{

		&DenseLayer{
			act_interface:   &relU{},
			size:            100,
			prev_layer_size: 28 * 28,
		},

		&DenseLayer{
			act_interface:   &relU{},
			size:            10,
			prev_layer_size: 100,
		},
	}

	net.init()
	net.init_new_weights()

	return net
}
