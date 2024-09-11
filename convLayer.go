package main

import "errors"

func (shelf *ConvLayer) init(layerID int) {
	shelf.layerID = layerID
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
func correlateOrConvolve2d(in1, in2 [][]float64, convolution bool, mode string, fillvalue float64) ([][]float64, error) {
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
