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

	res, err := ic.Conversational(context.Background(), &huggingface.ConversationalRequest{
		Inputs: huggingface.ConverstationalInputs{
			PastUserInputs: []string{
				"Which movie is the best ?",
				"Can you explain why ?",
			},
			GeneratedResponses: []string{
				"It's Die Hard for sure.",
				"It's the best movie ever.",
			},
			Text: "Can you explain why ?",
		},
	})
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(res.GeneratedText)
}
