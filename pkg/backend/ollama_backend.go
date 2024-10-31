// Copyright 2024 Stacklok, Inc
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//	http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
package backend

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"time"
)

const (
	chatEndpoint     = "/api/chat"
	generateEndpoint = "/api/generate"
	embedEndpoint    = "/api/embeddings"
	defaultTimeout   = 30 * time.Second
)

// OllamaBackend represents a backend for interacting with the Ollama API.
type OllamaBackend struct {
	Model   string
	Client  *http.Client
	BaseURL string
}

// OllamaResponse represents the structure of the response received from the Ollama API.
type OllamaResponse struct {
	Model              string                `json:"model"`
	CreatedAt          string                `json:"created_at"`
	Response           string                `json:"response"`
	Done               bool                  `json:"done"`
	DoneReason         string                `json:"done_reason"`
	Context            []int                 `json:"context"`
	TotalDuration      int64                 `json:"total_duration"`
	LoadDuration       int64                 `json:"load_duration"`
	PromptEvalCount    int                   `json:"prompt_eval_count"`
	PromptEvalDuration int64                 `json:"prompt_eval_duration"`
	EvalCount          int                   `json:"eval_count"`
	EvalDuration       int64                 `json:"eval_duration"`
	Message            OllamaResponseMessage `json:"message"`
}

// OllamaResponseMessage represents the message part of the response from the Ollama API.
type OllamaResponseMessage struct {
	Role      string           `json:"role"`
	Content   string           `json:"content"`
	ToolCalls []OllamaToolCall `json:"tool_calls"`
}

// OllamaToolCall represents a tool call to be made by the Ollama API.
type OllamaToolCall struct {
	Function OllamaFunctionCall `json:"function"`
}

// OllamaFunctionCall represents a function call to be made by the Ollama API.
type OllamaFunctionCall struct {
	Name      string         `json:"name"`
	Arguments map[string]any `json:"arguments"`
}

// OllamaEmbeddingResponse represents the response from the Ollama API for embeddings.
type OllamaEmbeddingResponse struct {
	Embedding []float32 `json:"embedding"`
}

// NewOllamaBackend creates a new OllamaBackend instance.
func NewOllamaBackend(baseURL, model string, timeout time.Duration) *OllamaBackend {
	return &OllamaBackend{
		BaseURL: baseURL,
		Model:   model,
		Client: &http.Client{
			Timeout: timeout,
		},
	}
}

type ollamaConversationOption struct {
	disableTools bool
}

// Converse drives a conversation with the Ollama API based on the given conversation context.
func (o *OllamaBackend) Converse(ctx context.Context, prompt *Prompt) (PromptResponse, error) {
	resp, err := o.converseRoundTrip(ctx, prompt, ollamaConversationOption{})
	if errors.Is(err, ErrToolNotFound) {
		// retry without tools if the error is due to a tool not being found
		return o.converseRoundTrip(ctx, prompt, ollamaConversationOption{disableTools: true})
	}

	return resp, err
}

func (o *OllamaBackend) converseRoundTrip(ctx context.Context, prompt *Prompt, opts ollamaConversationOption) (PromptResponse, error) {
	msgMap, err := prompt.AsMap()
	if err != nil {
		return PromptResponse{}, fmt.Errorf("failed to convert messages to map: %w", err)
	}

	url := o.BaseURL + chatEndpoint
	reqBody := map[string]any{
		"model":    o.Model,
		"messages": msgMap,
		"stream":   false,
	}

	if !opts.disableTools {
		toolMap, err := prompt.Tools.ToolsMap()
		if err != nil {
			return PromptResponse{}, fmt.Errorf("failed to convert tools to map: %w", err)
		}
		reqBody["tools"] = toolMap
	}

	reqBodyBytes, err := json.Marshal(reqBody)
	if err != nil {
		return PromptResponse{}, fmt.Errorf("failed to marshal request body: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewBuffer(reqBodyBytes))
	if err != nil {
		return PromptResponse{}, fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := o.Client.Do(req)
	if err != nil {
		return PromptResponse{}, fmt.Errorf("HTTP request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return PromptResponse{}, fmt.Errorf("failed to generate response from Ollama: status code %d, response: %s", resp.StatusCode, string(bodyBytes))
	}

	var result OllamaResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return PromptResponse{}, fmt.Errorf("failed to decode response: %w", err)
	}

	if len(result.Message.ToolCalls) == 0 {
		prompt.AddMessage("assistant", result.Message.Content)
		return PromptResponse{
			Role:    "assistant",
			Content: result.Message.Content,
		}, nil
	}

	response := PromptResponse{
		Role:      "tool",
		ToolCalls: make([]ToolCall, 0, len(result.Message.ToolCalls)),
	}
	for _, toolCall := range result.Message.ToolCalls {
		toolName := toolCall.Function.Name
		toolArgs := toolCall.Function.Arguments

		toolResponse, err := prompt.Tools.ExecuteTool(toolName, toolArgs)
		if errors.Is(err, ErrToolNotFound) {
			// this is a bit of a hack. Ollama models always reply with tool calls and hallucinate
			// the tool names if tools are given in the request, but the request is not actually
			// tied to any tool. So we just ignore these and re-send the request with tools disabled
			return PromptResponse{}, ErrToolNotFound
		} else if err != nil {
			return PromptResponse{}, fmt.Errorf("failed to execute tool: %w", err)
		}
		prompt.AddMessage("tool", toolResponse)

		response.ToolCalls = append(response.ToolCalls, ToolCall{
			Function: FunctionCall{
				Name:      toolName,
				Arguments: toolArgs,
				Result:    toolResponse,
			},
		})
	}

	return response, nil

}

// Generate produces a response from the Ollama API based on the given structured prompt.
//
// Parameters:
//   - ctx: The context for the API request, which can be used for cancellation.
//   - prompt: A structured prompt containing messages and parameters.
//
// Returns:
//   - A string containing the generated response from the Ollama model.
//   - An error if the API request fails or if there's an issue processing the response.
func (o *OllamaBackend) Generate(ctx context.Context, prompt *Prompt) (string, error) {
	url := o.BaseURL + generateEndpoint

	// Concatenate the messages into a single prompt string
	var promptText string
	for _, message := range prompt.Messages {
		// Append role and content into one string (adjust formatting as needed)
		promptText += message.Role + ": " + message.Content + "\n"
	}

	// Construct the request body with concatenated prompt
	reqBody := map[string]interface{}{
		"model":             o.Model,
		"prompt":            promptText, // Use concatenated string
		"max_tokens":        prompt.Parameters.MaxTokens,
		"temperature":       prompt.Parameters.Temperature,
		"top_p":             prompt.Parameters.TopP,
		"frequency_penalty": prompt.Parameters.FrequencyPenalty,
		"presence_penalty":  prompt.Parameters.PresencePenalty,
		"stream":            false, // Explicitly set stream to false
	}

	reqBodyBytes, err := json.Marshal(reqBody)
	if err != nil {
		return "", fmt.Errorf("failed to marshal request body: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewBuffer(reqBodyBytes))
	if err != nil {
		return "", fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := o.Client.Do(req)
	if err != nil {
		return "", fmt.Errorf("HTTP request failed: %w", err)
	}
	defer resp.Body.Close()

	bodyBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("failed to read response body: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf(
			"failed to generate response from Ollama: "+
				"status code %d, response: %s",
			resp.StatusCode, string(bodyBytes),
		)
	}

	var result OllamaResponse
	if err := json.NewDecoder(bytes.NewReader(bodyBytes)).Decode(&result); err != nil {
		return "", fmt.Errorf("failed to decode response: %w", err)
	}

	return result.Response, nil
}

// Embed generates embeddings for the given input text using the Ollama API.
func (o *OllamaBackend) Embed(ctx context.Context, input string) ([]float32, error) {
	url := o.BaseURL + embedEndpoint
	reqBody := map[string]interface{}{
		"model":  o.Model,
		"prompt": input,
	}

	reqBodyBytes, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request body: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewBuffer(reqBodyBytes))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := o.Client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("HTTP request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, err := io.ReadAll(resp.Body)
		if err != nil {
			return nil, fmt.Errorf("failed to read response body: %w", err)
		}
		return nil, fmt.Errorf(
			"failed to generate embeddings from Ollama: "+
				"status code %d, response: %s",
			resp.StatusCode, string(bodyBytes),
		)
	}

	var result OllamaEmbeddingResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return result.Embedding, nil
}
