package huggingface

import (
	"context"
	"encoding/json"
	"errors"
)

// TextClassificationRequest represents a request for text classification.
type TextClassificationRequest struct {
	// Inputs is the string to be generated from.
	Inputs string `json:"inputs"`
	// Options represents optional settings for the classification.
	Options Options `json:"options,omitempty"`
	// Model is the name of the model to use for classification.
	Model string `json:"-"`
}

// TextClassificationResponse represents a response for text classification.
type TextClassificationResponse [][]struct {
	// Label is the label for the class (model-specific).
	Label string `json:"label,omitempty"`
	// Score is a float that represents how likely it is that the text belongs to this class.
	Score float32 `json:"score,omitempty"`
}

// TextClassification performs text classification using the provided request.
func (ic *InferenceClient) TextClassification(ctx context.Context, req *TextClassificationRequest) (TextClassificationResponse, error) {
	// Check if inputs are provided.
	if len(req.Inputs) == 0 {
		return nil, errors.New("inputs are required")
	}

	body, err := ic.post(ctx, req.Model, "text-classification", req)
	if err != nil {
		return nil, err
	}

	textClassificationResponse := TextClassificationResponse{}
	if err := json.Unmarshal(body, &textClassificationResponse); err != nil {
		return nil, err
	}

	return textClassificationResponse, nil
}
