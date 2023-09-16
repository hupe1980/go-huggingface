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
		Options: huggingface.Options{
			WaitForModel: huggingface.PTR(true),
		},
	})
	if err1 != nil {
		log.Fatal(err1)
	}

	fmt.Println("FeatureExtraction:")
	fmt.Println(res1[0])
	fmt.Println()

	res2, err2 := ic.FeatureExtractionWithAutomaticReduction(context.Background(), &huggingface.FeatureExtractionRequest{
		Inputs: []string{"Hello World"},
		Model:  "sentence-transformers/all-mpnet-base-v2",
		Options: huggingface.Options{
			WaitForModel: huggingface.PTR(true),
		},
	})
	if err2 != nil {
		log.Fatal(err2)
	}

	fmt.Println("FeatureExtractionWithAutomaticReduction:")
	fmt.Println(res2[0])
	fmt.Println()
}
