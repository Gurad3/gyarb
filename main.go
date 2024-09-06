package main

import "ProjectX/data_handler"

func main() {

	net := tmpNewMNIST()
	//net := tmpNewXOR()
	// net.print_weights()

	//net := load_from_json("saves/TrainedMNIST.json")
	//net.print_weights()

	mim := new(MiM)
	mim.init(net)

	//xor(net, mim)
	mnist(net, mim)

	//net.print_weights()

	//_, _, _, TestData := loadMnist()
	//net.forward(mim, TestData[1])

	//encode_to_json(net)
}

func mnist(net *Network, mim *MiM) {
	MNIST_TrainDataLabel, MNIST_TrainData, MNIST_TestDataLabel, MNIST_TestData := data_handler.Load_mnist()

	td := trainer{
		TrainData:      MNIST_TrainData,
		TestData:       MNIST_TestData,
		TrainDataLabel: MNIST_TrainDataLabel,
		TestDataLabel:  MNIST_TestDataLabel,

		batch_size:        100,
		info_milestone:    60000,
		save_at_milestone: false,
	}
	net.train_network(mim, td)
}

func xor(net *Network, mim *MiM) {
	XOR_TrainDataLabel, XOR_TrainData, XOR_TestDataLabel, XOR_TestData := data_handler.Load_xor()

	td := trainer{
		TrainData:      XOR_TrainData,
		TestData:       XOR_TestData,
		TrainDataLabel: XOR_TrainDataLabel,
		TestDataLabel:  XOR_TestDataLabel,

		batch_size:        4,
		info_milestone:    60000,
		save_at_milestone: false,
	}
	net.train_network(mim, td)

}

func tmpNewMNIST() *Network {
	net := new(Network)

	net.learn_rate = .2
	net.learn_rate_decay = 0.0001
	net.file_name = "NewTrainedMNIST_2"
	net.cost_interface = &MeanSquare{}

	net.input_size = 28 * 28
	net.output_size = 10
	net.layers = []Layer{

		&DenseLayer{
			act_interface:   &relU{},
			size:            200,
			prev_layer_size: 28 * 28,
		},

		&DenseLayer{
			act_interface:   &Sigmoid{},
			size:            10,
			prev_layer_size: 200,
		},
	}

	net.init()
	net.init_new_weights()

	return net
}

func tmpNewXOR() *Network {
	net := new(Network)

	net.learn_rate = .5
	net.learn_rate_decay = 0.0001
	net.file_name = "TestNet"
	net.cost_interface = &MeanSquare{}

	net.input_size = 2
	net.output_size = 2
	net.layers = []Layer{

		&DenseLayer{
			act_interface:   &relU{},
			size:            8,
			prev_layer_size: 2,
		},

		&DenseLayer{
			act_interface:   &Sigmoid{},
			size:            2,
			prev_layer_size: 8,
		},
	}

	net.init()
	net.init_new_weights()

	return net
}
