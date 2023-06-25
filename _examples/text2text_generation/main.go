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

	res, err := ic.Text2TextGeneration(context.Background(), &huggingface.Text2TextGenerationRequest{
		Inputs: "The answer to the universe is",
		Model:  "gpt2", // overwrite recommended model
	})
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(res[0].GeneratedText)
}
