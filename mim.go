package main

const (
	OneD = iota
	TwoD
	ThreeD
)

type MiM struct {
	data_flat []float64
	data_2d   [][]float64
	data_3d   [][][]float64
	data_type int32
}

func (shelf *MiM) request_3d(x int32, y int32, z int32) {
	switch shelf.data_type {
	case OneD:

		arr3D := make([][][]float64, x)
		index := 0
		for i := int32(0); i < x; i++ {
			arr3D[i] = make([][]float64, y)
			for j := int32(0); j < y; j++ {
				arr3D[i][j] = make([]float64, z)
				for k := int32(0); k < z; k++ {
					arr3D[i][j][k] = shelf.data_flat[index]
					index++
				}
			}
		}
		shelf.data_3d = arr3D
	}
}

func (shelf *MiM) request_flat(x int32, y int32, z int32) {
	switch shelf.data_type {
	case ThreeD:

		new3d := make([][][]float64, x)
		index := 0
		for i := int32(0); i < x; i++ {
			new3d[i] = make([][]float64, y)
			for j := int32(0); j < y; j++ {
				new3d[i][j] = make([]float64, z)
				for k := int32(0); k < z; k++ {
					new3d[i][j][k] = shelf.data_flat[index]
					index++
				}
			}
		}

		shelf.data_3d = new3d
	}
}
