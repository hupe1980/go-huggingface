package main

import (
	"context"
	"fmt"
	"log"
	"os"

	huggingface "github.com/hupe1980/go-huggingface"
)

func main() {
	ic := huggingface.NewInferenceClient(os.Getenv("HUGGINGFACEHUB_API_TOKEN"))

	res, err := ic.FeatureExtraction(context.Background(), &huggingface.FeatureExtractionRequest{
		Inputs: []string{"Hello World"},
	})
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(res[0][0])
}
