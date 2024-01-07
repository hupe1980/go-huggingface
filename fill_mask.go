package huggingface

import (
	"context"
	"encoding/json"
	"errors"
)

// Request structure for the Fill Mask endpoint
type FillMaskRequest struct {
	// (Required) a string to be filled from, must contain the [MASK] token (check model card for exact name of the mask)
	Inputs  []string `json:"inputs"`
	Options Options  `json:"options,omitempty"`
	Model   string   `json:"-"`
}

// Response structure for the Fill Mask endpoint
type FillMaskResponse []struct {
	// The actual sequence of tokens that ran against the model (may contain special tokens)
	Sequence string `json:"sequence,omitempty"`

	// The probability for this token.
	Score float64 `json:"score,omitempty"`

	// The id of the token
	TokenID int `json:"token,omitempty"`

	// The string representation of the token
	TokenStr string `json:"token_str,omitempty"`
}

// FillMask performs masked language modeling using the specified model.
// It sends a POST request to the Hugging Face inference endpoint with the provided inputs.
// The response contains the generated text with the masked tokens filled or an error if the request fails.
func (ic *InferenceClient) FillMask(ctx context.Context, req *FillMaskRequest) (FillMaskResponse, error) {
	if len(req.Inputs) == 0 {
		return nil, errors.New("inputs are required")
	}

	body, err := ic.post(ctx, req.Model, "fill-mask", req)
	if err != nil {
		return nil, err
	}

	fillMaskResponse := FillMaskResponse{}
	if err := json.Unmarshal(body, &fillMaskResponse); err != nil {
		return nil, err
	}

	return fillMaskResponse, nil
}
