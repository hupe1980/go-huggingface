package gohuggingface

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
