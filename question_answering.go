package gohuggingface

import (
	"context"
	"encoding/json"
	"errors"
)

type QuestionAnsweringInputs struct {
	// (Required) The question as a string that has an answer within Context.
	Question string `json:"question"`

	// (Required) A string that contains the answer to the question
	Context string `json:"context"`
}

// Request structure for question answering model
type QuestionAnsweringRequest struct {
	// (Required)
	Inputs  QuestionAnsweringInputs `json:"inputs,omitempty"`
	Options Options                 `json:"options,omitempty"`
	Model   string                  `json:"-"`
}

// Response structure for question answering model
type QuestionAnsweringResponse struct {
	// A string that’s the answer within the Context text.
	Answer string `json:"answer,omitempty"`

	// A float that represents how likely that the answer is correct.
	Score float64 `json:"score,omitempty"`

	// The string index of the start of the answer within Context.
	Start int `json:"start,omitempty"`

	// The string index of the stop of the answer within Context.
	End int `json:"end,omitempty"`
}

// QuestionAnswering performs question answering using the specified model.
// It sends a POST request to the Hugging Face inference endpoint with the provided question and context inputs.
// The response contains the answer or an error if the request fails.
func (ic *InferenceClient) QuestionAnswering(ctx context.Context, req *QuestionAnsweringRequest) (*QuestionAnsweringResponse, error) {
	if req.Inputs.Question == "" {
		return nil, errors.New("question is required")
	}

	if req.Inputs.Context == "" {
		return nil, errors.New("context is required")
	}

	body, err := ic.post(ctx, req.Model, "question-answering", req)
	if err != nil {
		return nil, err
	}

	questionAnsweringResponse := QuestionAnsweringResponse{}
	if err := json.Unmarshal(body, &questionAnsweringResponse); err != nil {
		return nil, err
	}

	return &questionAnsweringResponse, nil
}
