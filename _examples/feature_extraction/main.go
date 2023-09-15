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

	res1, err1 := ic.FeatureExtraction(context.Background(), &huggingface.FeatureExtractionRequest{
		Inputs: []string{"Hello World"},
	})
	if err1 != nil {
		log.Fatal(err1)
	}

	fmt.Println(res1[0])

	res2, err2 := ic.FeatureExtractionWithAutomaticReduction(context.Background(), &huggingface.FeatureExtractionRequest{
		Inputs: []string{"Hello World"},
		Model:  "sentence-transformers/all-mpnet-base-v2",
	})
	if err2 != nil {
		log.Fatal(err2)
	}

	fmt.Println(res2[0])
}
