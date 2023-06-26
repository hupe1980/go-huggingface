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

	res, err := ic.QuestionAnswering(context.Background(), &huggingface.QuestionAnsweringRequest{
		Inputs: huggingface.QuestionAnsweringInputs{
			Question: "What's my name?",
			Context:  "My name is Clara and I live in Berkeley.",
		},
	})
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("Answer:", res.Answer)
	fmt.Println("Score:", res.Score)
	fmt.Println("Start:", res.Start)
	fmt.Println("End:", res.End)
}
