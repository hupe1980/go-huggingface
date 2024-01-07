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

	res, err := ic.SentenceSimilarity(context.Background(), &huggingface.SentenceSimilarityRequest{
		Inputs: huggingface.SentenceSimilarityInputs{
			SourceSentence: "That is a happy person",
			Sentences:      []string{"That is a happy dog", "That is a very happy person", "Today is a sunny day"},
		},
		Model: "sentence-transformers/all-MiniLM-L6-v2",
	})
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(res)
}
