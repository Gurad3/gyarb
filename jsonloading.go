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

	return *net
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
