package main

const (
	OneD = iota
	TwoD
	ThreeD
)

type MiM struct {
	data_flat *[]float64
	//data_2d   *[][]float64
	data_3d *[][][]float64

	data_type         int32
	data_type_history []int32

	layers_out_flat [][]float64 //First = input data
	layers_out_3d   [][][][]float64

	layers_out_flat_non_activated [][]float64
	layers_out_3d_non_activated   [][][][]float64

	layersDimentions [][]int32
}

func (shelf *MiM) request_3d(x int32, y int32, z int32) *MiM {
	switch shelf.data_type {
	case OneD:
		arr3D := make([][][]float64, x)
		index := 0
		for i := int32(0); i < x; i++ {
			arr3D[i] = make([][]float64, y)
			for j := int32(0); j < y; j++ {
				arr3D[i][j] = make([]float64, z)
				for k := int32(0); k < z; k++ {
					arr3D[i][j][k] = (*shelf.data_flat)[index]
					index++
				}
			}
		}
		shelf.data_3d = &arr3D
	}
	return shelf
}

func (shelf *MiM) request_flat() *MiM {
	switch shelf.data_type {
	case ThreeD:

		new_flat := make([]float64, len(*shelf.data_3d)*len((*shelf.data_3d)[0])*len((*shelf.data_3d)[0][0]))

		for i := 0; i < len(*shelf.data_3d); i++ {
			for j := 0; j < len((*shelf.data_3d)[0]); j++ {
				new_flat = append(new_flat, (*shelf.data_3d)[i][j]...)
			}
		}

		shelf.data_flat = &new_flat
	}

	return shelf
}

func (shelf *MiM) init(net *Network) {
	shelf.layersDimentions = make([][]int32, len(net.layers))

	shelf.layers_out_flat = make([][]float64, len(net.layers))
	shelf.layers_out_flat_non_activated = make([][]float64, len(net.layers))

	shelf.layers_out_3d = make([][][][]float64, len(net.layers))
	shelf.layers_out_3d_non_activated = make([][][][]float64, len(net.layers))

	for layerID, layer := range net.layers {
		layerDim := layer.get_size()

		switch len(layerDim) - 1 {

		case OneD:
			new1d := make([]float64, layerDim[0])
			shelf.layers_out_flat[layerID] = new1d

		case ThreeD:
			new1d := make([]float64, layerDim[0]*layerDim[1]*layerDim[2])
			shelf.layers_out_flat[layerID] = new1d

		}
	}

}
