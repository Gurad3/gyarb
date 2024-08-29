package main

import (
	"fmt"
)

func main() {
	fmt.Println("Hasdejdå")
	fmt.Println("hej")
	fmt.Println("hj")

	net := new(Network)

	net.learn_rate = 2
	net.learn_rate_decay = 0.0001

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

	net.print_weights()

	mim := new(MiM)

	mim.init(net)
	data := []float64{}
	net.forward(mim, data)

}
