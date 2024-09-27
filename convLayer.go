package main

import (
	"fmt"
	"math/rand"
)

type ConvLayer struct {
	act_interface Activation
	layerID       int
	out_size      []int
	input_shape   []int
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
	shelf.input_shape = prev_layer_size

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

func (shelf *ConvLayer) forward(mim *MiM) {

	matrix := *mim.data_flat

	for f, filter := range shelf.filters {
		outRangeStart := f * shelf.output_width * shelf.output_height
		outRangeEnd := (f + 1) * shelf.output_width * shelf.output_height
		filter.correlation(matrix, mim.layers_out[shelf.layerID][outRangeStart:outRangeEnd], mim.layers_out_non_activated[shelf.layerID], shelf.act_interface, shelf.input_shape, shelf.out_size[1:])

	}

	//mim.data_3d = &mim.layers_out_3d[shelf.layerID]
	//mim.data_type = ThreeD

	mim.data_flat = &mim.layers_out[shelf.layerID]
	//fmt.Println(mim.data_flat)
}

func (shelf *ConvLayer) backprop(mim *MiM, prev_layer_act Activation) {

	//mim.request_3d(shelf.layerID)

	// for f, filter := range shelf.filters {
	// 	filter.compute_loss_kernel_gradient(mim, shelf.output_width, shelf.output_height, shelf.layerID, f)
	//}
}
func (shelf *Filter) compute_loss_kernel_gradient(mim *MiM, O_W int, O_H int, layerID int, filterID int, inp_shape []int) {
	// Compute convolution between the input this layer recieved, and the matrix respresenting
	// the partial derivates of the Cost function with respect to this layers output.

	for i := 0; i < len(shelf.kernels[0]); i++ {
		for j := 0; j < len(shelf.kernels[0][0]); j++ {
			for k := 0; k < O_H; k++ {
				for l := 0; l < O_W; l++ {
					for c := 0; c < len(shelf.kernels); c++ {
						//shelf.kernel_gradients[c][i][j] += mim.layers_out_3d[layerID-1][c][i+k][j+l] * (*mim.data_3d)[filterID][k][l]

						//shelf.kernel_gradients[c][i][j] += mim.layers_out[layerID-1][c*inp_shape[1]*inp_shape[2]+(i+k)*inp_shape[1]+j+l] * (*mim.data_flat)[filterID][k][l]
					}
				}
			}
		}
	}

	for k := 0; k < O_H; k++ {
		for l := 0; l < O_W; l++ {
			//shelf.bias_gradient += (*mim.data_3d)[filterID][k][l]
		}
	}
}

/*	func (shelf *ConvLayer) compute_output_gradient() [][]float64 {
		// Kommer skickas vidare bakåt till nästa lager i backpropagation följden.
		// Outputen här blir matrisen som representerar partiella derivatorna för Cost med respekt till
		// lagrets output. Detta kommer då sedan användas i `compute_loss_kernel_gradient()`.

		// Använder även det föregående lagrets `compute_output_gradient`

		return full_convolve()
	}
*/

func (shelf *Filter) correlation(matrix []float64, target_activated []float64, target_non_activated []float64, activation Activation, input_shape []int, out_shape []int) {

	for i := 0; i < out_shape[0]; i++ {

		for j := 0; j < out_shape[1]; j++ {

			sum := shelf.bias

			for kernel_index := 0; kernel_index < len(shelf.kernels); kernel_index++ {
				kernel_offset := kernel_index * input_shape[1] * input_shape[2]
				for k := 0; k < len(shelf.kernels[kernel_index]); k++ {

					for l := 0; l < len(shelf.kernels[kernel_index][0]); l++ {

						//sum += (*matrix)[kernel_index][i+k][j+l] * shelf.kernels[kernel_index][k][l]

						sum += matrix[kernel_offset+(i+k)*input_shape[1]+(j+l)] * shelf.kernels[kernel_index][k][l]
					}
				}
			}
			//(*target_activated)[i][j] = (*activation).call(sum)
			//(*target_non_activated)[i][j] = sum

			target_activated[i*out_shape[0]+j] = activation.call(sum)
			target_non_activated[i*out_shape[0]+j] = sum
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
	fmt.Println(shelf.filters[0].kernels)
	fmt.Println(shelf.filters[0].kernel_gradients)
}
