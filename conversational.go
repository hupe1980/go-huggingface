package huggingface

import (
	"context"
	"encoding/json"
	"errors"
)

// Used with ConversationalRequest
type ConversationalParameters struct {
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

// Used with ConversationalRequest
type ConverstationalInputs struct {
	// (Required) The last input from the user in the conversation.
	Text string `json:"text"`

	// A list of strings corresponding to the earlier replies from the model.
	GeneratedResponses []string `json:"generated_responses,omitempty"`

	// A list of strings corresponding to the earlier replies from the user.
	// Should be of the same length of GeneratedResponses.
	PastUserInputs []string `json:"past_user_inputs,omitempty"`
}

// Request structure for the conversational endpoint
type ConversationalRequest struct {
	// (Required)
	Inputs ConverstationalInputs `json:"inputs,omitempty"`

	Parameters ConversationalParameters `json:"parameters,omitempty"`
	Options    Options                  `json:"options,omitempty"`
	Model      string                   `json:"-"`
}

// Used with ConversationalResponse
type Conversation struct {
	// The last outputs from the model in the conversation, after the model has run.
	GeneratedResponses []string `json:"generated_responses,omitempty"`

	// The last inputs from the user in the conversation, after the model has run.
	PastUserInputs []string `json:"past_user_inputs,omitempty"`
}

// Response structure for the conversational endpoint
type ConversationalResponse struct {
	// The answer of the model
	GeneratedText string `json:"generated_text,omitempty"`

	// A facility dictionary to send back for the next input (with the new user input addition).
	Conversation Conversation `json:"conversation,omitempty"`
}

// Conversational performs conversational AI using the specified model.
// It sends a POST request to the Hugging Face inference endpoint with the provided conversational inputs.
// The response contains the generated conversational response or an error if the request fails.
func (ic *InferenceClient) Conversational(ctx context.Context, req *ConversationalRequest) (*ConversationalResponse, error) {
	if len(req.Inputs.Text) == 0 {
		return nil, errors.New("text is required")
	}

	body, err := ic.post(ctx, req.Model, "conversational", req)
	if err != nil {
		return nil, err
	}

	conversationalResponse := ConversationalResponse{}
	if err := json.Unmarshal(body, &conversationalResponse); err != nil {
		return nil, err
	}

	return &conversationalResponse, nil
}
