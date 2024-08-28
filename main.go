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
	net.learn_rate_decay = 2

	net.cost_interface = &MeanSquare{}

	net.layers = []Layer{
		&DenseLayer{
			act_interface:   &relU{},
			size:            50,
			prev_layer_size: 28 * 28,
		},

		&DenseLayer{
			act_interface:   &relU{},
			size:            100,
			prev_layer_size: 50,
		},
	}

	net.init()
	net.init_new_weights()
}
