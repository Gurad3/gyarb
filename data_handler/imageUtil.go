package data_handler

import (
	"fmt"
	"image"
	"image/color"
	"image/png"
	"math"
	"math/rand"
	"os"
	"time"
)

func SaveImage(data []float64, filename string) error {
	img := image.NewGray(image.Rect(0, 0, 28, 28))

	for i := 0; i < 28*28; i++ {
		pixelValue := uint8(data[i] * 255)
		img.SetGray(i%28, i/28, color.Gray{Y: pixelValue})
	}

	file, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("failed to create file: %v", err)
	}
	defer file.Close()

	if err := png.Encode(file, img); err != nil {
		return fmt.Errorf("failed to encode image: %v", err)
	}

	return nil
}

func clampFloat(value, min, max float64) float64 {
	if value < min {
		return min
	}
	if value > max {
		return max
	}
	return value
}

func addNoiseToFloatArray(data *[]float64, noiseLevel float64) {
	rand.Seed(time.Now().UnixNano())

	nyans := (rand.Float64()*2 - 1) * 0.1

	for i := range len(*data) {
		noise := (rand.Float64()*2 - 1) * noiseLevel
		noisyValue := (*data)[i] + noise + nyans
		(*data)[i] = clampFloat(noisyValue, 0.0, 1.0)
	}
}

func rotateFloatArray(data *[]float64, angle float64) {
	size := 28

	rotatedData := make([]float64, size*size)
	angleRad := angle * (math.Pi / 180.0)
	center := float64(size-1) / 2.0

	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {

			relX := float64(x) - center
			relY := float64(y) - center

			rotatedX := relX*math.Cos(angleRad) - relY*math.Sin(angleRad)
			rotatedY := relX*math.Sin(angleRad) + relY*math.Cos(angleRad)
			srcX := int(math.Round(rotatedX + center))
			srcY := int(math.Round(rotatedY + center))

			if srcX >= 0 && srcX < size && srcY >= 0 && srcY < size {
				rotatedData[y*size+x] = (*data)[srcY*size+srcX]
			} else {
				rotatedData[y*size+x] = 0.0
			}
		}
	}

	*data = rotatedData
}

func NoiseInplace(data *[]float64) {
	rotateFloatArray(data, (rand.Float64()*2-1)*45)
	addNoiseToFloatArray(data, 0.1)
}

func NoiseNewSet(data *[][]float64) *[][]float64 {
	out := make([][]float64, len(*data))
	copy(out, (*data))
	for i := 0; i < len(*data); i++ {
		NoiseInplace(&out[i])
	}

	return &out
}
