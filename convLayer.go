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

	filters []Filter
}

type Filter struct {
	kernels [][][]float64
	bias    float64

	kernel_gradients [][][]float64
	bias_gradient    float64
}

func (shelf *ConvLayer) init(layerID int, prev_layer_size []int) {
	//Init all layer arrays sizes, Sets bias AND WEIGHTS to 0
	shelf.layer_type = "ConvLayer"
	shelf.layerID = layerID

	shelf.input_depth = prev_layer_size[0]
	shelf.input_width = prev_layer_size[1]
	shelf.input_height = prev_layer_size[2]

	shelf.output_width = shelf.input_width - shelf.kernel_size + 1
	shelf.output_height = shelf.input_height - shelf.kernel_size + 1

	shelf.out_size = []int{shelf.depth, shelf.output_height, shelf.output_width}

	shelf.filters = make([]Filter, shelf.depth)

	for i := 0; i < shelf.depth; i++ {
		//kernels
		shelf.filters[i].kernels = make([][][]float64, shelf.input_depth)
		shelf.filters[i].kernel_gradients = make([][][]float64, shelf.input_depth)
		shelf.filters[i].bias = 0
		shelf.filters[i].bias_gradient = 0
		for j := 0; j < shelf.input_depth; j++ {

			shelf.filters[i].kernels[j] = make([][]float64, shelf.kernel_size)
			shelf.filters[i].kernel_gradients[j] = make([][]float64, shelf.kernel_size)

			for k := 0; k < shelf.kernel_size; k++ {
				shelf.filters[i].kernels[j][k] = make([]float64, shelf.kernel_size)
				shelf.filters[i].kernel_gradients[j][k] = make([]float64, shelf.kernel_size)
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
					//shelf.filters[i].kernels[j][k][l] = initWeightXavierUniform(xavierRange, r)

					shelf.filters[i].kernels[j][k][l] = tmp_manualKernel[i][k][l]
				}
			}
		}
	}
}

func forward2(mim *MiM) {

}

func (shelf *ConvLayer) forward(mim *MiM) {

	matrix := mim.request_3d(shelf.layerID - 1).data_3d

	for i, filter := range shelf.filters {
		filter.correlation(matrix, &mim.layers_out_3d[shelf.layerID][i], &mim.layers_out_3d_non_activated[shelf.layerID][i], &shelf.act_interface)
	}

	mim.data_3d = &mim.layers_out_3d[shelf.layerID]
	mim.data_type = ThreeD
}

func (shelf *ConvLayer) backprop(mim *MiM, prev_layer_act Activation) {

}

/*func (shelf *ConvLayer) backprop(mim *MiM, prev_layer_act Activation) {
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
}*/

func (shelf *Filter) correlation(matrix *[][][]float64, target_activated *[][]float64, target_non_activated *[][]float64, activation *Activation) {
	for i := 0; i < len((*matrix)[0])-len(shelf.kernels[0])+1; i++ {

		for j := 0; j < len((*matrix)[0][0])-len(shelf.kernels[0][0])+1; j++ {

			sum := shelf.bias
			for kernel_index := 0; kernel_index < len(shelf.kernels); kernel_index++ {
				for k := 0; k < len(shelf.kernels[kernel_index]); k++ {
					for l := 0; l < len(shelf.kernels[kernel_index][0]); l++ {
						sum += (*matrix)[kernel_index][i+k][j+l] * shelf.kernels[kernel_index][k][l]
					}
				}
			}
			(*target_activated)[i][j] = (*activation).call(sum)
			(*target_non_activated)[i][j] = sum
			// fmt.Println(sum)
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
					shelf.filters[i].kernels[j][k][l] -= shelf.filters[i].kernel_gradients[j][k][l] * mult
					shelf.filters[i].kernel_gradients[j][k][l] = 0
				}
			}

		}
		shelf.filters[i].bias -= shelf.filters[i].bias_gradient
		shelf.filters[i].bias_gradient = 0
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
