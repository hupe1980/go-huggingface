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

	res, err := ic.ZeroShotClassification(context.Background(), &huggingface.ZeroShotClassificationRequest{
		Inputs: []string{"Hi, I recently bought a device from your company but it is not working as advertised and I would like to get reimbursed!"},
		Parameters: huggingface.ZeroShotClassificationParameters{
			CandidateLabels: []string{"refund", "faq", "legal"},
		},
	})
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(res[0].Sequence)
	fmt.Println("Labels:", res[0].Labels)
	fmt.Println("Scores:", res[0].Scores)
}
