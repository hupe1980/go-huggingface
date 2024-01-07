package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/hupe1980/go-huggingface"
)

func main() {
	ic := huggingface.NewInferenceClient(os.Getenv("HUGGINGFACEHUB_API_TOKEN"))

	res, err := ic.TextClassification(context.Background(), &huggingface.TextClassificationRequest{
		Inputs: "The answer to the universe is 42",
		//Model:  "deepset/deberta-v3-base-injection", // overwrite recommended model
	})
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(res[0])
}
