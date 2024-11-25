package main

import (
	"ProjectX/data_handler"
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

func cliHandler() {
	reader := bufio.NewReader(os.Stdin)
	var net *Network
	var mim *MiM
	// Loop to read commands from the GUI
	for {
		command, err := reader.ReadString('\n')
		if err != nil {
			fmt.Fprintln(os.Stderr, "Error reading input:", err)
			os.Exit(0)
			continue
		}
		command = strings.TrimSpace(command)
		// Parse the command
		parts := strings.Fields(command)
		if len(parts) == 0 {
			continue
		}
		switch parts[0] {
		case "load":
			net = load_from_json(parts[1])
			mim = new(MiM)
			mim.init(net)

			fmt.Println("loadok")

		case "eval":
			jsonData := strings.Join(parts[1:], " ")
			var data []float64

			// Parse the JSON data
			err := json.Unmarshal([]byte(jsonData), &data)
			if err != nil {

			}
			net.forward(mim, data)
			//response, _ := json.Marshal(*mim.data_flat)
			fmt.Println(*mim.data_flat)

		case "hello":
			fmt.Println("hellook" + parts[1])
		case "isready":
			fmt.Println("readyok")
		case "quit":
			os.Exit(0)
		case "exit":
			os.Exit(0)
		}
	}

}

func main() {
	//tmp()
	//cliHandler()
	//TMPCLI()
	net := tmpNewMNIST()
	//net := tmpNewXOR()
	// net.print_weights()

	//net := load_from_json("saves/ConvMNIST_1.json")

	// net.file_name = "test"

	// encode_to_json(net)

	//net.layers[1].debug_print()

	//xor(net)
	mnist(net)

	// mim := new(MiM)
	// mim.init(net)
	// _, _, MNIST_TestDataLabel, MNIST_TestData := data_handler.Load_mnist()
	// net.Test(mim, MNIST_TestData, MNIST_TestDataLabel)

	//net.print_weights()
	//_, _, _, TestData := loadMnist()
	//net.forward(mim, TestData[1])

	//encode_to_json(net)

	// _, _, _, MNIST_TestData := data_handler.Load_mnist()
	// data_handler.SaveImage(MNIST_TestData[0], "./images/image.png")
	// fmt.Println(MNIST_TestData[0], MNIST_TestDataLabel[0])
}

func mnist(net *Network) {
	MNIST_TrainDataLabel, MNIST_TrainData, MNIST_TestDataLabel, MNIST_TestData := data_handler.Load_mnist()
	td := trainer{
		TrainData:      MNIST_TrainData,
		TestData:       MNIST_TestData,
		TrainDataLabel: MNIST_TrainDataLabel,
		TestDataLabel:  MNIST_TestDataLabel,

		batch_size:        50,
		info_milestone:    60_000,
		save_at_milestone: true,
	}

	net.train_network(td, true)

}

func xor(net *Network) {
	XOR_TrainDataLabel, XOR_TrainData, XOR_TestDataLabel, XOR_TestData := data_handler.Load_xor()

	td := trainer{
		TrainData:      XOR_TrainData,
		TestData:       XOR_TestData,
		TrainDataLabel: XOR_TrainDataLabel,
		TestDataLabel:  XOR_TestDataLabel,

		batch_size:        4,
		info_milestone:    10_000,
		save_at_milestone: false,
	}
	net.train_network(td, true)

}

func tmpNewMNIST() *Network {
	net := new(Network)

	net.learn_rate = .2
	//net.learn_rate_decay = 0.0001
	net.regularization = 0.1
	net.momentum = 0
	net.file_name = "MNIST_SiLU_Cross"
	net.cost_interface = &CrossEntropy{}

	net.input_shape = []int{1, 28, 28}
	//net.input_shape = []int{28 * 28}
	net.output_size = 10

	net.layers = []Layer{

		// &ConvLayer{
		// 	act_interface: &SiLU{},
		// 	kernel_size:   4,
		// 	depth:         2,
		// },

		// &ConvLayer{
		// 	act_interface: &SiLU{},
		// 	kernel_size:   3,
		// 	depth:         4,
		// },

		&DenseLayer{
			act_interface: &SiLU{},
			size:          120,
		},

		&DenseLayer{
			act_interface: &SiLU{},
			size:          120,
		},

		&DenseLayer{
			act_interface: &Sigmoid{},
			size:          10,
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
			act_interface:   &RelU{},
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
