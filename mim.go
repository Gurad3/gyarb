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

func (self *MiM) requst_3d(x int32, y int32, z int32) [x][y][z]float64 {
	switch self.data_type {
	case OneD:

		break
	}
}
