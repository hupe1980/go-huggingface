package gohuggingface

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
)

var (
	// recommendedModels stores the recommended models for each task.
	recommendedModels map[string]string
)

// HTTPClient is an interface representing an HTTP client.
type HTTPClient interface {
	Do(req *http.Request) (*http.Response, error)
}

// InferenceClientOptions represents options for the InferenceClient.
type InferenceClientOptions struct {
	Model             string
	Endpoint          string
	InferenceEndpoint string
	HTTPClient        HTTPClient
}

// InferenceClient is a client for performing inference using Hugging Face models.
type InferenceClient struct {
	httpClient HTTPClient
	token      string
	opts       InferenceClientOptions
}

// NewInferenceClient creates a new InferenceClient instance with the specified token.
func NewInferenceClient(token string, optFns ...func(o *InferenceClientOptions)) *InferenceClient {
	opts := InferenceClientOptions{
		Endpoint:          "https://huggingface.co",
		InferenceEndpoint: "https://api-inference.huggingface.co",
	}

	for _, fn := range optFns {
		fn(&opts)
	}

	if opts.HTTPClient == nil {
		opts.HTTPClient = http.DefaultClient
	}

	return &InferenceClient{
		httpClient: opts.HTTPClient,
		token:      token,
		opts:       opts,
	}
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

// TextGeneration performs text generation using the specified model.
// It sends a POST request to the Hugging Face inference endpoint with the provided inputs.
// The response contains the generated text or an error if the request fails.
func (ic *InferenceClient) TextGeneration(ctx context.Context, req *TextGenerationRequest) (TextGenerationResponse, error) {
	if req.Inputs == "" {
		return nil, errors.New("inputs are required")
	}

	body, err := ic.post(ctx, req.Model, "text-generation", req)
	if err != nil {
		return nil, err
	}

	textGenerationResponse := TextGenerationResponse{}
	if err := json.Unmarshal(body, &textGenerationResponse); err != nil {
		return nil, err
	}

	return textGenerationResponse, nil
}

// Text2TextGeneration performs text-to-text generation using the specified model.
// It sends a POST request to the Hugging Face inference endpoint with the provided inputs.
// The response contains the generated text or an error if the request fails.
func (ic *InferenceClient) Text2TextGeneration(ctx context.Context, req *Text2TextGenerationRequest) (Text2TextGenerationResponse, error) {
	if req.Inputs == "" {
		return nil, errors.New("inputs are required")
	}

	body, err := ic.post(ctx, req.Model, "text2text-generation", req)
	if err != nil {
		return nil, err
	}

	text2TextGenerationResponse := Text2TextGenerationResponse{}
	if err := json.Unmarshal(body, &text2TextGenerationResponse); err != nil {
		return nil, err
	}

	return text2TextGenerationResponse, nil
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

// QuestionAnswering performs question answering using the specified model.
// It sends a POST request to the Hugging Face inference endpoint with the provided question and context inputs.
// The response contains the answer or an error if the request fails.
func (ic *InferenceClient) QuestionAnswering(ctx context.Context, req *QuestionAnsweringRequest) (QuestionAnsweringResponse, error) {
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

	return questionAnsweringResponse, nil
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

// post sends a POST request to the specified model and task with the provided payload.
// It returns the response body or an error if the request fails.
func (ic *InferenceClient) post(ctx context.Context, model, task string, payload any) ([]byte, error) {
	url, err := ic.resolveURL(ctx, model, task)
	if err != nil {
		return nil, err
	}

	body, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "application/json")

	if ic.token != "" {
		httpReq.Header.Set("Authorization", fmt.Sprintf("Bearer %s", ic.token))
	}

	res, err := ic.httpClient.Do(httpReq)
	if err != nil {
		return nil, err
	}

	defer res.Body.Close()

	resBody, err := io.ReadAll(res.Body)
	if err != nil {
		return nil, err
	}

	if res.StatusCode != http.StatusOK {
		errResp := ErrorResponse{}
		if err := json.Unmarshal(resBody, &errResp); err != nil {
			return nil, fmt.Errorf("huggingfaces error: %s", resBody)
		}

		return nil, fmt.Errorf("huggingfaces error: %s", errResp.Error)
	}

	return resBody, nil
}

// resolveURL resolves the URL for the specified model and task.
// It returns the resolved URL or an error if resolution fails.
func (ic *InferenceClient) resolveURL(ctx context.Context, model, task string) (string, error) {
	if model == "" {
		model = ic.opts.Model
	}

	// If model is already a URL, ignore `task` and return directly
	if model != "" && (strings.HasPrefix(model, "http://") || strings.HasPrefix(model, "https://")) {
		return model, nil
	}

	if model == "" {
		var err error

		model, err = ic.getRecommendedModel(ctx, task)
		if err != nil {
			return "", err
		}
	}

	// Feature-extraction and sentence-similarity are the only cases where models support multiple tasks
	if contains([]string{"feature-extraction", "sentence-similarity"}, task) {
		return fmt.Sprintf("%s/pipeline/%s/%s", ic.opts.InferenceEndpoint, task, model), nil
	}

	return fmt.Sprintf("%s/models/%s", ic.opts.InferenceEndpoint, model), nil
}

// getRecommendedModel retrieves the recommended model for the specified task.
// It returns the recommended model or an error if retrieval fails.
func (ic *InferenceClient) getRecommendedModel(ctx context.Context, task string) (string, error) {
	rModels, err := ic.fetchRecommendedModels(ctx)
	if err != nil {
		return "", err
	}

	model, ok := rModels[task]
	if !ok {
		return "", fmt.Errorf("task %s has no recommended model", task)
	}

	return model, nil
}

// fetchRecommendedModels retrieves the recommended models for all available tasks.
// It returns a map of task names to recommended models or an error if retrieval fails.
func (ic *InferenceClient) fetchRecommendedModels(ctx context.Context) (map[string]string, error) {
	if recommendedModels == nil {
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, fmt.Sprintf("%s/api/tasks", ic.opts.Endpoint), nil)
		if err != nil {
			return nil, err
		}

		res, err := ic.httpClient.Do(req)
		if err != nil {
			return nil, err
		}
		defer res.Body.Close()

		var jsonResponse map[string]interface{}

		err = json.NewDecoder(res.Body).Decode(&jsonResponse)
		if err != nil {
			return nil, err
		}

		recommendedModels = make(map[string]string)

		for task, details := range jsonResponse {
			widgetModels, ok := details.(map[string]interface{})["widgetModels"].([]interface{})
			if !ok || len(widgetModels) == 0 {
				recommendedModels[task] = ""
			} else {
				firstModel, _ := widgetModels[0].(string)
				recommendedModels[task] = firstModel
			}
		}
	}

	return recommendedModels, nil
}

// Contains checks if the given element is present in the collection.
func contains[T comparable](collection []T, element T) bool {
	for _, item := range collection {
		if item == element {
			return true
		}
	}

	return false
}
