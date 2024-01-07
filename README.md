# 🤗 go-huggingface
![Build Status](https://github.com/hupe1980/go-huggingface/workflows/build/badge.svg) 
[![Go Reference](https://pkg.go.dev/badge/github.com/hupe1980/go-huggingface.svg)](https://pkg.go.dev/github.com/hupe1980/go-huggingface)
> The Hugging Face Inference Client in Golang is a modul designed to interact with the Hugging Face model repository and perform inference tasks using state-of-the-art natural language processing models. Developed in Golang, it provides a seamless and efficient way to integrate Hugging Face models into your Golang applications.

## Installation
```
go get github.com/hupe1980/go-huggingface
```

## How to use
```golang
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
```
Output:
```text
Hi, I recently bought a device from your company but it is not working as advertised and I would like to get reimbursed!
Labels: [refund faq legal]
Scores: [0.8777876496315002 0.10522633790969849 0.016985949128866196]
```

For more example usage, see [_examples](./_examples).

## License
[MIT](LICENCE)