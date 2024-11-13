package main

import (
	"ProjectX/data_handler"
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"strconv"
	"strings"
)

func tmp() {
	_, _, MNIST_TestDataLabel, MNIST_TestData := data_handler.Load_mnist()

	// t := 9
	// net := load_from_json("MULT/MULT_" + strconv.Itoa(t) + ".json")
	// mim := new(MiM)
	// mim.init(net)

	// for i := 0; i < len(MNIST_TestDataLabel); i++ {
	// 	if MNIST_TestDataLabel[i][t] == 1 {
	// 		MNIST_TestDataLabel[i] = []float64{1, 0}
	// 	} else {
	// 		MNIST_TestDataLabel[i] = []float64{0, 1}
	// 	}
	// }
	// net.Test(mim, MNIST_TestData, MNIST_TestDataLabel)

	nets := make([]*Network, 10)

	for i := 0; i < 10; i++ {
		nets[i] = load_from_json("MULT/MULT_" + strconv.Itoa(i) + ".json")
	}
	mim := new(MiM)
	mim.init(nets[0])

	NETresults := make([]float64, 10)
	correct := 0
	for sampleID := 0; sampleID < len(MNIST_TestDataLabel); sampleID++ {

		big := 0.0
		currIndex := 0
		for i := 0; i < 10; i++ {
			nets[i].forward(mim, MNIST_TestData[sampleID])
			NETresults[i] = (*mim.data_flat)[0]
			if NETresults[i] > big {
				big = NETresults[i]
				currIndex = i
			}
		}

		if MNIST_TestDataLabel[sampleID][currIndex] == 1 {
			correct++
		}
	}

	fmt.Println(float64(correct) / float64(len(MNIST_TestData)))
}

func TMPCLI() {
	reader := bufio.NewReader(os.Stdin)

	nets := make([]*Network, 10)
	for i := 0; i < 10; i++ {
		nets[i] = load_from_json("MULT/MULT_" + strconv.Itoa(i) + ".json")
	}
	mim := new(MiM)
	mim.init(nets[0])

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

			fmt.Println("loadok")

		case "eval":
			jsonData := strings.Join(parts[1:], " ")
			var data []float64

			// Parse the JSON data
			err := json.Unmarshal([]byte(jsonData), &data)
			if err != nil {

			}
			big := 0.0
			currIndex := 0
			for i := 0; i < 10; i++ {
				nets[i].forward(mim, data)

				if (*mim.data_flat)[0] > big {
					big = (*mim.data_flat)[0]
					currIndex = i
				}
			}
			res := make([]float64, 10)
			res[currIndex] = 1

			fmt.Println(res)

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
