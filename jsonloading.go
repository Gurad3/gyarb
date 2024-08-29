package main

import (
	"encoding/json"
	"os"
)

type JsonWriter struct {
	path string
}

func encode_to_json(net Network) {
	net_file, _ := os.Create("saves/" + net.file_name + ".json")
	encoder := json.NewEncoder(net_file)

	encoder.Encode(net)
}

func load_from_json(path string) Network {
	file, _ := os.Open(path)
	decoder := json.NewDecoder(file)

	net := new(Network)

	decoder.Decode(net)

	return *net
}
