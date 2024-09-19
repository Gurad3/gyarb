package main

import (
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

	kernels [][]Kernel
}

type Kernel struct {
	window [][]float64
	bias   float64

	window_gradients [][]float64
	bias_gradient    float64
}

func (shelf *ConvLayer) init(layerID int) {
	//Init all layer arrays sizes, Sets bias AND WEIGHTS to 0
	shelf.layer_type = "ConvLayer"
	shelf.layerID = layerID

	shelf.output_width = shelf.input_width - shelf.kernel_size + 1
	shelf.output_height = shelf.input_height - shelf.kernel_size + 1

	shelf.out_size = []int{shelf.depth, shelf.output_height, shelf.output_width}

	shelf.kernels = make([][]Kernel, shelf.depth)

	for i := 0; i < shelf.depth; i++ {
		//kernels
		shelf.kernels[i] = make([]Kernel, shelf.input_depth)

		for j := 0; j < shelf.input_depth; j++ {
			shelf.kernels[i][j].bias = 0
			shelf.kernels[i][j].bias_gradient = 0
			shelf.kernels[i][j].window = make([][]float64, shelf.kernel_size)
			shelf.kernels[i][j].window_gradients = make([][]float64, shelf.kernel_size)

			for k := 0; k < shelf.kernel_size; k++ {
				shelf.kernels[i][j].window[k] = make([]float64, shelf.kernel_size)
				shelf.kernels[i][j].window_gradients[k] = make([]float64, shelf.kernel_size)

			}
		}

	}
}

func (shelf *ConvLayer) init_new_weights(xavierRange float64, r rand.Rand) {
	//Give each weights new random weights (Currentlu 0)

	for i := 0; i < shelf.depth; i++ {
		for j := 0; j < shelf.input_depth; j++ {
			for k := 0; k < shelf.kernel_size; k++ {
				for l := 0; l < shelf.kernel_size; l++ {
					shelf.kernels[i][j].window[k][l] = initWeightXavierUniform(xavierRange, r)
				}
			}
		}
	}
}

func forward2(mim *MiM) {

}

func (shelf *ConvLayer) forward(mim *MiM) {
	mim.layers_out_3d_non_activated[shelf.layerID] = shelf.bias

	mim.request_3d(shelf.layerID - 1)
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
	// fmt.Println(len(mim.layers_out_3d[shelf.layerID]))
	mim.data_3d = &mim.layers_out_3d[shelf.layerID]
	mim.data_type = ThreeD
}

func (shelf *ConvLayer) backprop(mim *MiM, prev_layer_act Activation) {
	output_gradient := *mim.request_3d(shelf.layerID).data_3d

	next_gradient := make([][][]float64, len(mim.layers_out_3d[shelf.layerID-1]))
	for i := 0; i < len(mim.layers_out_3d[shelf.layerID-1]); i++ {
		next_gradient[i] = make([][]float64, len(mim.layers_out_3d[shelf.layerID-1][i]))

		for i2 := 0; i2 < len(mim.layers_out_3d[shelf.layerID-1][i]); i2++ {
			next_gradient[i][i2] = make([]float64, len(mim.layers_out_3d[shelf.layerID-1][i][i2]))
		}
	}
	//fmt.Println(output_gradient[0][0])
	for i := 0; i < shelf.depth; i++ {
		for j := 0; j < shelf.output_height; j++ {
			for k := 0; k < shelf.output_width; k++ {
				shelf.bias_gradients[i][j][k] += output_gradient[i][j][k]
			}
		}

		for j := 0; j < shelf.input_depth; j++ {

			kernalCorr, _ := correlateOrConvolve2d(mim.layers_out_3d[shelf.layerID-1][j], output_gradient[i], false, "valid", 0)
			// fmt.Println(len(output_gradient), i)
			input_convlove, _ := correlateOrConvolve2d(output_gradient[i], shelf.kernels[i][j], true, "full", 0)

			for k, k2 := range mim.layers_out_3d[shelf.layerID-1][j] {
				for l := range k2 {
					//mim.layers_out_3d[shelf.layerID-1][j][k][l] += input_convlove[k][l]
					next_gradient[j][k][l] += input_convlove[k][l] * prev_layer_act.ddx(mim.layers_out_3d[shelf.layerID-1][j][k][l])

				}
			}

			for k, k2 := range kernalCorr {
				for l := range k2 {

					shelf.kernels_gradients[i][j][k][l] += kernalCorr[k][l]

				}
			}

		}
	}
	mim.layers_out_3d[shelf.layerID-1] = next_gradient
	mim.data_3d = &mim.layers_out_3d[shelf.layerID-1]
	mim.data_type = ThreeD
}

func (shelf *Kernel) correlation(matrix [][]float64, target *[][]float64) {

	for i := 0; i < len(matrix)-len(shelf.window)+1; i++ {
		for j := 0; j < len(matrix[0])-len(shelf.window[0])+1; j++ {
			sum := shelf.bias
			for k := 0; k < len(shelf.window); k++ {
				for l := 0; l < len(shelf.window[0]); l++ {
					sum += matrix[i+k][j+l] * shelf.window[k][l]
				}
			}

			(*target)[i][j] = sum
		}
	}
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

func (shelf *ConvLayer) apply_gradients(learn_rate float64, batch_size float64) {
	mult := learn_rate / batch_size
	for i := 0; i < shelf.depth; i++ {
		for j := 0; j < shelf.input_depth; j++ {
			for k := 0; k < shelf.kernel_size; k++ {
				for l := 0; l < shelf.kernel_size; l++ {
					shelf.kernels[i][j].window[k][l] -= shelf.kernels[i][j].window_gradients[k][l] * mult
					shelf.kernels[i][j].window_gradients[k][l] = 0
				}
			}
			shelf.kernels[i][j].bias -= shelf.kernels[i][j].bias_gradient
			shelf.kernels[i][j].bias_gradient = 0
		}
	}

}

func (shelf *ConvLayer) load_weights(flat_weights []float64) {

}
func (shelf *ConvLayer) load_biases(flat_weights []float64) {

}
func (shelf *ConvLayer) get_weights() []float64 {
	return []float64{}
}
func (shelf *ConvLayer) get_biases() []float64 {
	return []float64{}
}
func (shelf *ConvLayer) print_weights() {

}

func (shelf *ConvLayer) get_act_interface() Activation {
	return shelf.act_interface
}
func (shelf *ConvLayer) get_size() []int {
	return shelf.out_size
}
func (shelf *ConvLayer) get_name() string {
	return shelf.layer_type
}

func (shelf *ConvLayer) debug_print() {

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
