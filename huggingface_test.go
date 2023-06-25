package gohuggingface

import (
	"bytes"
	"context"
	"errors"
	"io"
	"net/http"
	"testing"

	"github.com/stretchr/testify/assert"
)

// Mock HTTP Client for testing purposes
type mockHTTPClient struct {
	Response []byte
	Err      error
}

func (c *mockHTTPClient) Do(req *http.Request) (*http.Response, error) {
	if c.Err != nil {
		return nil, c.Err
	}

	return &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(bytes.NewBuffer(c.Response)),
	}, nil
}

func TestSummarization(t *testing.T) {
	client := NewInferenceClient("your-token")
	mockResponse := []byte(`[{"summary_text": "This is a summary"}]`)

	t.Run("Successful Request", func(t *testing.T) {
		// Mock HTTP Client with successful response
		mockHTTP := &mockHTTPClient{Response: mockResponse}
		client.httpClient = mockHTTP

		req := &SummarizationRequest{
			Inputs: []string{"This is a test input"},
			Model:  "t5-base",
		}

		response, err := client.Summarization(context.Background(), req)
		assert.NoError(t, err)
		assert.NotNil(t, response)
		assert.Equal(t, "This is a summary", response[0].SummaryText)
	})

	t.Run("Empty Inputs", func(t *testing.T) {
		req := &SummarizationRequest{
			Inputs: nil, // Empty inputs
			Model:  "t5-base",
		}

		response, err := client.Summarization(context.Background(), req)
		assert.Error(t, err)
		assert.Nil(t, response)
		assert.Equal(t, "inputs are required", err.Error())
	})

	t.Run("HTTP Request Error", func(t *testing.T) {
		// Mock HTTP Client with error response
		mockHTTP := &mockHTTPClient{Err: errors.New("request error")}
		client.httpClient = mockHTTP

		req := &SummarizationRequest{
			Inputs: []string{"This is a test input"},
			Model:  "t5-base",
		}

		response, err := client.Summarization(context.Background(), req)
		assert.Error(t, err)
		assert.Nil(t, response)
		assert.Equal(t, "request error", err.Error())
	})
}

func TestQuestionAnswering(t *testing.T) {
	client := NewInferenceClient("your-token")

	t.Run("Missing question input", func(t *testing.T) {
		req := &QuestionAnsweringRequest{
			Model: "distilbert-base-uncased-distilled-squad",
			Inputs: QuestionAnsweringInputs{
				Context: "Paris is the capital of France.",
			},
		}
		_, err := client.QuestionAnswering(context.Background(), req)
		assert.EqualError(t, err, "question is required")
	})

	t.Run("Missing context input", func(t *testing.T) {
		req := &QuestionAnsweringRequest{
			Model: "distilbert-base-uncased-distilled-squad",
			Inputs: QuestionAnsweringInputs{
				Question: "What is the capital of France?",
			},
		}
		_, err := client.QuestionAnswering(context.Background(), req)
		assert.EqualError(t, err, "context is required")
	})
}
