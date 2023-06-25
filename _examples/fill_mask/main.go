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

	res, err := ic.FillMask(context.Background(), &huggingface.FillMaskRequest{
		Inputs: []string{"The answer to the universe is <mask>."},
	})
	if err != nil {
		log.Fatal(err)
	}

	for _, r := range res {
		fmt.Println("Sequence:", r.Sequence)
		fmt.Println("Score:", r.Score)
		fmt.Println("TokenID:", r.TokenID)
		fmt.Println("TokenStr", r.TokenStr)
		fmt.Println("---")
	}
}
