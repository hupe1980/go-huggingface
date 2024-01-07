package huggingface

import (
	"context"
	"encoding/json"
	"errors"
)

// Request structure for table question answering model
type TableQuestionAnsweringRequest struct {
	Inputs  TableQuestionAnsweringInputs `json:"inputs"`
	Options Options                      `json:"options,omitempty"`
	Model   string                       `json:"-"`
}

type TableQuestionAnsweringInputs struct {
	// (Required) The query in plain text that you want to ask the table
	Query string `json:"query"`

	// (Required) A table of data represented as a dict of list where entries
	// are headers and the lists are all the values, all lists must
	// have the same size.
	Table map[string][]string `json:"table"`
}

// Response structure for table question answering model
type TableQuestionAnsweringResponse struct {
	// The plaintext answer
	Answer string `json:"answer,omitempty"`

	// A list of coordinates of the cells references in the answer
	Coordinates [][]int `json:"coordinates,omitempty"`

	// A list of coordinates of the cells contents
	Cells []string `json:"cells,omitempty"`

	// The aggregator used to get the answer
	Aggregator string `json:"aggregator,omitempty"`
}

// TableQuestionAnswering performs table-based question answering using the specified model.
// It sends a POST request to the Hugging Face inference endpoint with the provided inputs.
// The response contains the answer or an error if the request fails.
func (ic *InferenceClient) TableQuestionAnswering(ctx context.Context, req *TableQuestionAnsweringRequest) (*TableQuestionAnsweringResponse, error) {
	if req.Inputs.Query == "" {
		return nil, errors.New("query is required")
	}

	if req.Inputs.Table == nil {
		return nil, errors.New("table is required")
	}

	body, err := ic.post(ctx, req.Model, "table-question-answering", req)
	if err != nil {
		return nil, err
	}

	tablequestionAnsweringResponse := TableQuestionAnsweringResponse{}
	if err := json.Unmarshal(body, &tablequestionAnsweringResponse); err != nil {
		return nil, err
	}

	return &tablequestionAnsweringResponse, nil
}
