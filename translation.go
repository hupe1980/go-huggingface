package gohuggingface

import (
	"context"
	"encoding/json"
	"errors"
)

// TranslationRequest represents a request for translation.
type TranslationRequest struct {
	Inputs  []string `json:"inputs"`
	Options Options  `json:"options,omitempty"`
	Model   string   `json:"-"`
}

// TranslationResponse represents the response for a translation request.
type TranslationResponse []struct {
	TranslationText string `json:"translation_text"`
}

// Translation sends a translation request to the InferenceClient and returns the translation response.
func (ic *InferenceClient) Translation(ctx context.Context, req *TranslationRequest) (TranslationResponse, error) {
	if len(req.Inputs) == 0 {
		return nil, errors.New("inputs are required")
	}

	body, err := ic.post(ctx, req.Model, "translation", req)

	if err != nil {
		return nil, err
	}

	translationResponse := TranslationResponse{}

	if err := json.Unmarshal(body, &translationResponse); err != nil {
		return nil, err
	}

	return translationResponse, nil
}
