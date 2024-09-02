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

	//TrainDataLabel, TrainData, TestDataLabel, TestData := loadMnist()

	_, _, TestDataLabel, TestData := loadMnist()
	net.forward(mim, TestData[1], TestDataLabel[1])

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
