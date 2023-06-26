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

	res, err := ic.TableQuestionAnswering(context.Background(), &huggingface.TableQuestionAnsweringRequest{
		Inputs: huggingface.TableQuestionAnsweringInputs{
			Query: "How many stars does the transformers repository have?",
			Table: map[string][]string{
				"Repository":   {"Transformers", "Datasets", "Tokenizers"},
				"Stars":        {"36542", "4512", "3934"},
				"Contributors": {"651", "77", "34"},
			},
		},
	})
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("Answer:", res.Answer)
	fmt.Println("Coordinates:", res.Coordinates)
	fmt.Println("Cells:", res.Cells)
	fmt.Println("Aggregator:", res.Aggregator)
}
