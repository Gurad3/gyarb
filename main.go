package main

import (
	"ProjectX/data_handler"
)

func main() {
	//tmp()
	//cliHandler()

	net := CreateNewNetwork()
	train_mnist(net)
}

func train_mnist(net *Network) {
	MNIST_TrainDataLabel, MNIST_TrainData, MNIST_TestDataLabel, MNIST_TestData := data_handler.Load_mnist()
	td := trainer{
		TrainData:      MNIST_TrainData,
		TestData:       MNIST_TestData,
		TrainDataLabel: MNIST_TrainDataLabel,
		TestDataLabel:  MNIST_TestDataLabel,

		batch_size:        50,
		info_milestone:    60_000,
		save_at_milestone: true,

		threded: true,
	}

	net.train_network(td)
}

func CreateNewNetwork() *Network {
	net := new(Network)

	net.learn_rate = .02
	net.regularization = 0.1
	net.momentum = 0.4

	net.file_name = "MNIST_Example_Net" // Filnamnet som nätvärket sparas till. (JSON)
	net.cost_interface = &MeanSquare{}

	net.input_shape = []int{1, 28, 28} // Djup, Bredd, Höjd.
	net.output_size = 10               // Antal neuroner i sista lagret.

	net.layers = []Layer{

		&ConvLayer{
			act_interface: &RelU{},
			kernel_size:   3,
			depth:         2,
		},

		&ConvLayer{
			act_interface: &RelU{},
			kernel_size:   3,
			depth:         4,
		},

		&DenseLayer{ //Alias till fully-connected layer.
			act_interface: &RelU{},
			size:          120,
		},

		&DenseLayer{
			act_interface: &RelU{},
			size:          120,
		},

		&DenseLayer{
			act_interface: &Sigmoid{},
			size:          10,
		},
	}

	net.init()             // Allokerar minne för nätvärket.
	net.init_new_weights() //Slumpar nya weights

	return net
}
