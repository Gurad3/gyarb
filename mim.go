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

	layers_dimentions [][]int
}

func (shelf *MiM) request_3d(layerID int) *MiM {
	switch shelf.data_type {
	case OneD:

		//X,Y,Z = prev_layer_dim[0,1,2]
		prev_layer_dim := shelf.layers_dimentions[layerID]
		arr3D := make([][][]float64, prev_layer_dim[0])
		index := 0
		for i := int(0); i < prev_layer_dim[0]; i++ {
			arr3D[i] = make([][]float64, prev_layer_dim[1])
			for j := int(0); j < prev_layer_dim[1]; j++ {
				arr3D[i][j] = make([]float64, prev_layer_dim[2])
				for k := int(0); k < prev_layer_dim[2]; k++ {
					arr3D[i][j][k] = (*shelf.data_flat)[index]
					// fmt.Println((*shelf.data_flat)[index])
					index++
				}
			}
		}
		shelf.data_3d = &arr3D
	}
	shelf.data_type = ThreeD

	return shelf
}

func (shelf *MiM) request_flat() *MiM {
	switch shelf.data_type {
	case ThreeD:

		new_flat := make([]float64, len(*shelf.data_3d)*len((*shelf.data_3d)[0])*len((*shelf.data_3d)[0][0]))
		//new_flat := make([]float64, 0, len(*shelf.data_3d)*len((*shelf.data_3d)[0])*len((*shelf.data_3d)[0][0]))
		f := 0
		for i := 0; i < len(*shelf.data_3d); i++ {
			for j := 0; j < len((*shelf.data_3d)[0]); j++ {

				//new_flat = append(new_flat, (*shelf.data_3d)[i][j]...)
				for k := 0; k < len((*shelf.data_3d)[0][0]); k++ {
					new_flat[f] = (*shelf.data_3d)[i][j][k]
					f++
				}
			}
		}
		shelf.data_flat = &new_flat

	}
	shelf.data_type = OneD
	return shelf
}

func (shelf *MiM) init(net *Network) {
	length := len(net.layers) + 1

	shelf.layers_dimentions = make([][]int, length)

	shelf.layers_dimentions[0] = []int{1, 28, 28} // TODO GLÖM INTE BORT

	shelf.data_type_history = make([]int32, length)

	shelf.layers_out_flat = make([][]float64, length)
	shelf.layers_out_flat_non_activated = make([][]float64, length)

	shelf.layers_out_3d = make([][][][]float64, length)
	shelf.layers_out_3d_non_activated = make([][][][]float64, length)

	for fake_layer_id, layer := range net.layers {
		layerDim := layer.get_size()
		layerID := fake_layer_id + 1
		switch len(layerDim) - 1 { //OneD == 0 ?????

		case OneD:
			shelf.layers_out_flat[layerID] = make([]float64, layerDim[0])
			shelf.layers_out_flat_non_activated[layerID] = make([]float64, layerDim[0])

		case ThreeD:
			shelf.layers_out_flat[layerID] = make([]float64, layerDim[0]*layerDim[1]*layerDim[2])
			shelf.layers_out_flat_non_activated[layerID] = make([]float64, layerDim[0]*layerDim[1]*layerDim[2])

			shelf.layers_out_3d_non_activated[layerID] = make([][][]float64, layerDim[0])
			shelf.layers_out_3d[layerID] = make([][][]float64, layerDim[0])
			for i := 0; i < layerDim[0]; i++ {
				shelf.layers_out_3d_non_activated[layerID][i] = make([][]float64, layerDim[1])
				shelf.layers_out_3d[layerID][i] = make([][]float64, layerDim[1])
				for j := 0; j < layerDim[1]; j++ {
					shelf.layers_out_3d_non_activated[layerID][i][j] = make([]float64, layerDim[2])
					shelf.layers_out_3d[layerID][i][j] = make([]float64, layerDim[2])
				}
			}
		}
		shelf.layers_dimentions[layerID] = layerDim
	}

}
