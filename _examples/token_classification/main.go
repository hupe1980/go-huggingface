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

	res, err := ic.TokenClassification(context.Background(), &huggingface.TokenClassificationRequest{
		Inputs: "My name is Sarah Jessica Parker but you can call me Jessica.",
	})
	if err != nil {
		log.Fatal(err)
	}

	for _, r := range res {
		fmt.Printf("Word: %s\n", r.Word)
		fmt.Printf("EntityGroup: %s\n", r.EntityGroup)
		fmt.Printf("Score: %f\n", r.Score)
		fmt.Printf("Start: %d\n", r.Start)
		fmt.Printf("End: %d\n", r.End)
		fmt.Println("::")
	}
}
