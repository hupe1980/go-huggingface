package gohuggingface

import (
	"context"
	"encoding/json"
	"errors"
)

type SummarizationParameters struct {
	// (Default: None). Integer to define the minimum length in tokens of the output summary.
	MinLength *int `json:"min_length,omitempty"`

	// (Default: None). Integer to define the maximum length in tokens of the output summary.
	MaxLength *int `json:"max_length,omitempty"`

	// (Default: None). Integer to define the top tokens considered within the sample operation to create
	// new text.
	TopK *int `json:"top_k,omitempty"`

	// (Default: None). Float to define the tokens that are within the sample` operation of text generation.
	// Add tokens in the sample for more probable to least probable until the sum of the probabilities is
	// greater than top_p.
	TopP *float64 `json:"top_p,omitempty"`

	// (Default: 1.0). Float (0.0-100.0). The temperature of the sampling operation. 1 means regular sampling,
	// 0 mens top_k=1, 100.0 is getting closer to uniform probability.
	Temperature *float64 `json:"temperature,omitempty"`

	// (Default: None). Float (0.0-100.0). The more a token is used within generation the more it is penalized
	// to not be picked in successive generation passes.
	RepetitionPenalty *float64 `json:"repetitionpenalty,omitempty"`

	// (Default: None). Float (0-120.0). The amount of time in seconds that the query should take maximum.
	// Network can cause some overhead so it will be a soft limit.
	MaxTime *float64 `json:"maxtime,omitempty"`
}

type SummarizationRequest struct {
	// String to be summarized
	Inputs     []string                `json:"inputs"`
	Parameters SummarizationParameters `json:"parameters,omitempty"`
	Options    Options                 `json:"options,omitempty"`
	Model      string                  `json:"-"`
}

type SummarizationResponse []struct {
	// The summarized input string
	SummaryText string `json:"summary_text,omitempty"`
}

// Summarization performs text summarization using the specified model.
// It sends a POST request to the Hugging Face inference endpoint with the provided inputs.
// The response contains the generated summary or an error if the request fails.
func (ic *InferenceClient) Summarization(ctx context.Context, req *SummarizationRequest) (SummarizationResponse, error) {
	if len(req.Inputs) == 0 {
		return nil, errors.New("inputs are required")
	}

	body, err := ic.post(ctx, req.Model, "summarization", req)
	if err != nil {
		return nil, err
	}

	summarizationResponse := SummarizationResponse{}
	if err := json.Unmarshal(body, &summarizationResponse); err != nil {
		return nil, err
	}

	return summarizationResponse, nil
}
