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

	res, err := ic.Translation(context.Background(), &huggingface.TranslationRequest{
		Inputs: []string{"Меня зовут Вольфганг и я живу в Берлине"},
		Model:  "Helsinki-NLP/opus-mt-ru-en",
	})
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(res[0].TranslationText)
}
