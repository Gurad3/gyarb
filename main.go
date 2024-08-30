package main

import (
	"fmt"
)

func main() {
	net := tmpNewNet()
	net.print_weights()

	fmt.Println("-")
	encode_to_json(net)

	NewNet := load_from_json("saves/TestNet.json")
	NewNet.print_weights()

	//mim := new(MiM)
	//mim.init(net)

	// data := []float64{}
	// net.forward(mim, data)

}

func tmpNewNet() *Network {
	net := new(Network)

	net.learn_rate = 2
	net.learn_rate_decay = 0.0001
	net.file_name = "TestNet"
	net.cost_interface = &MeanSquare{}

	net.input_size = 2
	net.output_size = 4
	net.layers = []Layer{

		&DenseLayer{
			act_interface:   &relU{},
			size:            4,
			prev_layer_size: 2,
		},

		&DenseLayer{
			act_interface:   &relU{},
			size:            2,
			prev_layer_size: 4,
		},
	}

	net.init()
	net.init_new_weights()

	return net
}
