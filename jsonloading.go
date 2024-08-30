package main

import (
	"encoding/json"
	"os"
)

type NetworkData struct {
	File_name        string  `json:"file_name"`
	Learn_rate       float64 `json:"learn_rate"`
	Learn_rate_decay float64 `json:"learn_rate_decay"`
	Input_size       int     `json:"input_size"`
	Output_size      int     `json:"output_size"`
	Cost_interface   string  `json:"cost_interface"`

	Layer_types       []string `json:"layer_types"`
	Layer_activations []string `json:"layer_activations"`

	//All flated
	Layer_biases  [][]float64 `json:"layer_biases"`
	Layer_weights [][]float64 `json:"layer_weights"`

	Layer_sizes [][]int `json:"layer_sizes"`
}

func load_from_net_data(net_data NetworkData) Network {
	net := new(Network)

	net.file_name = net_data.File_name
	net.learn_rate = net_data.Learn_rate
	net.learn_rate_decay = net_data.Learn_rate_decay
	net.input_size = net_data.Input_size
	net.output_size = net_data.Output_size

	net.layers = make([]Layer, len(net_data.Layer_types))

	for i := 0; i < len(net_data.Layer_types); i++ {
		if net_data.Layer_types[i] == "DenseLayer" {
			New_layer := DenseLayer{
				act_interface:   Act_name_to_func(net_data.Layer_activations[i]),
				size:            net_data.Layer_sizes[i][0],
				prev_layer_size: len(net_data.Layer_biases[i]) / net_data.Layer_sizes[i][0],
			}
			New_layer.init(i + 1)
			New_layer.load_weights(net_data.Layer_weights[i])
			net.layers[i] = &New_layer

		}

		if net_data.Layer_types[i] == "CNNLayer" {

		}
	}

	return *net
}

func save_to_net_data(net Network) NetworkData {

	net_data := new(NetworkData)

	net_data.File_name = net.file_name

	net_data.Learn_rate = net.learn_rate
	net_data.Learn_rate_decay = net.learn_rate_decay
	net_data.Input_size = net.input_size
	net_data.Output_size = net.output_size

	layers_len := len(net.layers)
	net_data.Layer_types = make([]string, layers_len)
	net_data.Layer_activations = make([]string, layers_len)
	net_data.Layer_biases = make([][]float64, layers_len)
	net_data.Layer_weights = make([][]float64, layers_len)
	net_data.Layer_sizes = make([][]int, layers_len)
	for layerID, layer := range net.layers {
		net_data.Layer_types[layerID] = layer.get_name()
		net_data.Layer_activations[layerID] = layer.get_act_name()
		net_data.Layer_biases[layerID] = layer.get_biases()
		net_data.Layer_weights[layerID] = layer.get_weights()
		net_data.Layer_sizes[layerID] = layer.get_size()
	}

	return *net_data
}

func encode_to_json(netData NetworkData) {
	net_file, _ := os.Create("saves/" + netData.File_name + ".json")
	encoder := json.NewEncoder(net_file)
	encoder.Encode(netData)
}

func load_from_json(path string) NetworkData {
	file, _ := os.Open(path)
	decoder := json.NewDecoder(file)

	net := new(NetworkData)

	decoder.Decode(net)

	return *net
}
