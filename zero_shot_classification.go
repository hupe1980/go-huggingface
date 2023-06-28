package gohuggingface

import (
	"context"
	"encoding/json"
	"errors"
)

type ZeroShotClassificationParameters struct {
	// (Required) A list of strings that are potential classes for inputs. Max 10 candidate_labels,
	// for more, simply run multiple requests, results are going to be misleading if using
	// too many candidate_labels anyway. If you want to keep the exact same, you can
	// simply run multi_label=True and do the scaling on your end.
	CandidateLabels []string `json:"candidate_labels"`

	// (Default: false) Boolean that is set to True if classes can overlap
	MultiLabel *bool `json:"multi_label,omitempty"`
}

type ZeroShotClassificationRequest struct {
	// (Required) Input or Inputs are required request fields
	Inputs []string `json:"inputs"`
	// (Required)
	Parameters ZeroShotClassificationParameters `json:"parameters,omitempty"`
	Options    Options                          `json:"options,omitempty"`
	Model      string                           `json:"-"`
}

type ZeroShotClassificationResponse []struct {
	// The string sent as an input
	Sequence string `json:"sequence,omitempty"`

	// The list of labels sent in the request, sorted in descending order
	// by probability that the input corresponds to the to the label.
	Labels []string `json:"labels,omitempty"`

	// a list of floats that correspond the the probability of label, in the same order as labels.
	Scores []float64 `json:"scores,omitempty"`
}

// ZeroShotClassification performs zero-shot classification using the specified model.
// It sends a POST request to the Hugging Face inference endpoint with the provided inputs.
// The response contains the classification results or an error if the request fails.
func (ic *InferenceClient) ZeroShotClassification(ctx context.Context, req *ZeroShotClassificationRequest) (ZeroShotClassificationResponse, error) {
	if len(req.Inputs) == 0 {
		return nil, errors.New("inputs are required")
	}

	if len(req.Parameters.CandidateLabels) == 0 {
		return nil, errors.New("canidateLabels are required")
	}

	body, err := ic.post(ctx, req.Model, "zero-shot-classification", req)
	if err != nil {
		return nil, err
	}

	zeroShotClassificationResponse := ZeroShotClassificationResponse{}
	if err := json.Unmarshal(body, &zeroShotClassificationResponse); err != nil {
		return nil, err
	}

	return zeroShotClassificationResponse, nil
}
