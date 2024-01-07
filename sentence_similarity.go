package huggingface

import (
	"context"
	"encoding/json"
	"errors"
)

// SentenceSimilarityInputs represents the inputs for sentence similarity computation.
type SentenceSimilarityInputs struct {
	SourceSentence string   `json:"source_sentence"`
	Sentences      []string `json:"sentences"`
}

// SentenceSimilarityRequest represents a request for sentence similarity computation.
type SentenceSimilarityRequest struct {
	Inputs  SentenceSimilarityInputs `json:"inputs"`
	Options Options                  `json:"options,omitempty"`
	Model   string                   `json:"-"`
}

// SentenceSimilarityResponse represents the response for a sentence similarity computation request.
type SentenceSimilarityResponse []float32

// SentenceSimilarity sends a sentence similarity computation request to the InferenceClient
// and returns the sentence similarity response.
func (ic *InferenceClient) SentenceSimilarity(ctx context.Context, req *SentenceSimilarityRequest) (SentenceSimilarityResponse, error) {
	if len(req.Inputs.SourceSentence) == 0 || len(req.Inputs.Sentences) == 0 {
		return nil, errors.New("sourceSentence and sentences are required")
	}

	body, err := ic.post(ctx, req.Model, "sentence-similarity", req)

	if err != nil {
		return nil, err
	}

	sentenceSimilarityResponse := SentenceSimilarityResponse{}

	if err := json.Unmarshal(body, &sentenceSimilarityResponse); err != nil {
		return nil, err
	}

	return sentenceSimilarityResponse, nil
}
