package gohuggingface

import (
	"context"
	"encoding/json"
	"errors"
)

// TokenClassificationarameters represents the parameters for token classification.
type TokenClassificationarameters struct {
	// AggregationStrategy specifies the aggregation strategy.
	// - none: Every token gets classified without further aggregation.
	// - simple: Entities are grouped according to the default schema (B-, I- tags get merged when the tag is similar).
	// - first: Same as the simple strategy except words cannot end up with different tags. Words will use the tag of the first token when there is ambiguity.
	// - average: Same as the simple strategy except words cannot end up with different tags. Scores are averaged across tokens and then the maximum label is applied.
	// - max: Same as the simple strategy except words cannot end up with different tags. Word entity will be the token with the maximum score.
	AggregationStrategy string `json:"aggregation_strategy,omitempty"`
}

// TokenClassificationRequest represents the input parameters for token classification.
type TokenClassificationRequest struct {
	// Inputs is a string to be classified.
	Inputs string `json:"inputs"`
	// Parameters contains token classification parameters.
	Parameters TokenClassificationarameters `json:"parameters"`
	// Options contains token classification options.
	Options Options `json:"options"`
	Model   string  `json:"-"`
}

// TokenClassificationResponse  represents the output of the token classification.
type TokenClassificationResponse []struct {
	// EntityGroup is the type for the entity being recognized (model specific).
	EntityGroup string `json:"entity_group"`

	// Score indicates how likely the entity was recognized.
	Score float64 `json:"score"`

	// Word is the string that was captured.
	Word string `json:"word"`

	// Start is the offset stringwise where the answer is located. Useful to disambiguate if the word occurs multiple times.
	Start int `json:"start"`

	// End is the offset stringwise where the answer is located. Useful to disambiguate if the word occurs multiple times.
	End int `json:"end"`
}

func (ic *InferenceClient) TokenClassification(ctx context.Context, req *TokenClassificationRequest) (TokenClassificationResponse, error) {
	if req.Inputs == "" {
		return nil, errors.New("inputs are required")
	}

	if req.Parameters.AggregationStrategy == "" {
		req.Parameters.AggregationStrategy = "simple"
	}

	body, err := ic.post(ctx, req.Model, "token-classification", req)
	if err != nil {
		return nil, err
	}

	tokenClassificationResponse := TokenClassificationResponse{}
	if err := json.Unmarshal(body, &tokenClassificationResponse); err != nil {
		return nil, err
	}

	return tokenClassificationResponse, nil
}
