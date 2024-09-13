package main

import (
	"errors"
	"math/rand"
)

type ConvLayer struct {
	act_interface Activation
	layerID       int
	out_size      []int
	layer_type    string

	depth       int
	kernel_size int

	input_depth  int
	input_width  int
	input_height int

	output_width  int
	output_height int
	kernels       [][][][]float64
	bias          [][][]float64

	kernels_gradients [][][][]float64
	bias_gradients    [][][]float64
}

func (shelf *ConvLayer) init(layerID int) {
	//Init all layer arrays sizes, Sets bias AND WEIGHTS to 0
	shelf.layer_type = "ConvLayer"
	shelf.layerID = layerID

	shelf.output_width = shelf.input_width - shelf.kernel_size + 1
	shelf.output_height = shelf.input_height - shelf.kernel_size + 1

	shelf.out_size = []int{shelf.depth, shelf.output_height, shelf.output_width}

	shelf.bias = make([][][]float64, shelf.depth)
	shelf.kernels = make([][][][]float64, shelf.depth)

	shelf.bias_gradients = make([][][]float64, shelf.depth)
	shelf.kernels_gradients = make([][][][]float64, shelf.depth)

	for i := 0; i < shelf.depth; i++ {
		//kernels
		shelf.kernels[i] = make([][][]float64, shelf.input_depth)
		shelf.kernels_gradients[i] = make([][][]float64, shelf.input_depth)

		for j := 0; j < shelf.input_depth; j++ {
			shelf.kernels[i][j] = make([][]float64, shelf.kernel_size)
			shelf.kernels_gradients[i][j] = make([][]float64, shelf.kernel_size)

			for k := 0; k < shelf.kernel_size; k++ {
				shelf.kernels[i][j][k] = make([]float64, shelf.kernel_size)
				shelf.kernels_gradients[i][j][k] = make([]float64, shelf.kernel_size)

			}
		}

		//bias
		shelf.bias[i] = make([][]float64, shelf.output_height)
		shelf.bias_gradients[i] = make([][]float64, shelf.output_height)
		for j := 0; j < shelf.output_height; j++ {
			shelf.bias[i][j] = make([]float64, shelf.output_width)
			shelf.bias_gradients[i][j] = make([]float64, shelf.output_width)
		}
	}
}

func (shelf *ConvLayer) init_new_weights(xavierRange float64, r rand.Rand) {
	//Give each weights new random weights (Currentlu 0)

	for i := 0; i < shelf.depth; i++ {
		//kernels
		for j := 0; j < shelf.input_depth; j++ {
			for k := 0; k < shelf.kernel_size; k++ {
				for l := 0; l < shelf.kernel_size; l++ {
					shelf.kernels[i][j][k][l] = initWeightXavierUniform(xavierRange, r)
				}
			}
		}
	}
}

func (shelf *ConvLayer) forward(mim *MiM) {
	mim.layers_out_3d_non_activated[shelf.layerID] = shelf.bias

	mim.request_3d(shelf.layerID)
	for i := 0; i < shelf.depth; i++ {
		for j := 0; j < shelf.input_depth; j++ {
			conv, _ := correlateOrConvolve2d((*mim.data_3d)[j], shelf.kernels[i][j], false, "valid", 0)

			for k, k2 := range mim.layers_out_3d_non_activated[shelf.layerID][i] {
				for l := range k2 {
					mim.layers_out_3d_non_activated[shelf.layerID][i][k][l] += conv[k][l]
					mim.layers_out_3d[shelf.layerID][i][k][l] = shelf.act_interface.call(mim.layers_out_3d_non_activated[shelf.layerID][i][k][l])
				}
			}

		}
	}

	mim.data_3d = &mim.layers_out_3d[shelf.layerID]
	mim.data_type = ThreeD
}

// flipKernel flips the kernel (matrix) horizontally and vertically for convolution.
func flipKernel(kernel [][]float64) [][]float64 {
	rows := len(kernel)
	cols := len(kernel[0])
	flipped := make([][]float64, rows)
	for i := range flipped {
		flipped[i] = make([]float64, cols)
		for j := range flipped[i] {
			flipped[i][j] = kernel[rows-1-i][cols-1-j] // Flip both rows and columns.
		}
	}
	return flipped
}

// correlateOrConvolve2d computes the 2D cross-correlation or convolution of two 2D slices with specified modes.
func correlateOrConvolve2d(in1 [][]float64, in2 [][]float64, convolution bool, mode string, fillvalue float64) ([][]float64, error) {
	rows1 := len(in1)
	cols1 := len(in1[0])
	rows2 := len(in2)
	cols2 := len(in2[0])

	if mode == "valid" && (rows1 < rows2 || cols1 < cols2) {
		return nil, errors.New("input matrix dimensions are smaller than kernel dimensions for 'valid' mode")
	}

	// Flip the kernel for convolution.
	if convolution {
		in2 = flipKernel(in2)
	}

	// Determine the output size based on the mode.
	var outRows, outCols int
	switch mode {
	case "full":
		outRows = rows1 + rows2 - 1
		outCols = cols1 + cols2 - 1
	case "valid":
		outRows = rows1 - rows2 + 1
		outCols = cols1 - cols2 + 1
	default:
		return nil, errors.New("unsupported mode; use 'full' or 'valid'")
	}

	// Initialize the output array with the fill value.
	output := make([][]float64, outRows)
	for i := range output {
		output[i] = make([]float64, outCols)
		for j := range output[i] {
			output[i][j] = fillvalue
		}
	}

	// Perform the 2D correlation or convolution.
	for i := 0; i < outRows; i++ {
		for j := 0; j < outCols; j++ {
			sum := 0.0
			for m := 0; m < rows2; m++ {
				for n := 0; n < cols2; n++ {
					rowIndex := i + m
					colIndex := j + n

					if mode == "full" {
						// Check boundary conditions for 'full' mode.
						if rowIndex >= rows1 || colIndex >= cols1 || rowIndex < 0 || colIndex < 0 {
							continue
						}
						sum += in1[rowIndex][colIndex] * in2[m][n]
					} else if mode == "valid" {
						// Direct index access for 'valid' mode.
						sum += in1[i+m][j+n] * in2[m][n]
					}
				}
			}
			output[i][j] = sum
		}
	}

	return output, nil
}

// class Convolutional(Layer):
//     def __init__(self, input_shape, kernel_size, depth):
//         input_depth, input_height, input_width = input_shape
//         self.depth = depth
//         self.input_shape = input_shape
//         self.input_depth = input_depth
//         self.output_shape = (depth, input_height - kernel_size + 1, input_width - kernel_size + 1)
//         self.kernels_shape = (depth, input_depth, kernel_size, kernel_size)
//         self.kernels = np.random.randn(*self.kernels_shape)
//         self.biases = np.random.randn(*self.output_shape)

//     def forward(self, input):
//         self.input = input
//         self.output = np.copy(self.biases)
//         for i in range(self.depth):
//             for j in range(self.input_depth):
//                 self.output[i] += signal.correlate2d(self.input[j], self.kernels[i, j], "valid")
//         return self.output

//     def backward(self, output_gradient, learning_rate):
//         kernels_gradient = np.zeros(self.kernels_shape)
//         input_gradient = np.zeros(self.input_shape)

//         for i in range(self.depth):
//             for j in range(self.input_depth):
//                 kernels_gradient[i, j] = signal.correlate2d(self.input[j], output_gradient[i], "valid")
//                 input_gradient[j] += signal.convolve2d(output_gradient[i], self.kernels[i, j], "full")

//         self.kernels -= learning_rate * kernels_gradient
//         self.biases -= learning_rate * output_gradient
//         return input_gradient
