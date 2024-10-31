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
	"fmt"
	"io"
	"net/http"
	"time"
)

// OpenAIBackend represents a backend for interacting with the OpenAI API.
// It contains configuration details and methods for making API requests.
type OpenAIBackend struct {
	APIKey     string
	Model      string
	HTTPClient *http.Client
	BaseURL    string
}

// OpenAIEmbeddingResponse represents the structure of the response received from the OpenAI API
// for an embedding request. It contains the generated embeddings, usage statistics, and other
// metadata related to the API call.
type OpenAIEmbeddingResponse struct {
	Object string `json:"object"`
	Data   []struct {
		Object    string    `json:"object"`
		Embedding []float32 `json:"embedding"`
		Index     int       `json:"index"`
	} `json:"data"`
	Model string `json:"model"`
	Usage struct {
		PromptTokens int `json:"prompt_tokens"`
		TotalTokens  int `json:"total_tokens"`
	} `json:"usage"`
}

// NewOpenAIBackend creates and returns a new OpenAIBackend instance with a custom timeout.
//
// Parameters:
//   - apiKey: The API key for authenticating with the OpenAI API.
//   - model: The name of the OpenAI model to use for generating responses.
//   - timeout: The duration for the HTTP client timeout. If zero, the default timeout is used.
//
// Returns:
//   - A pointer to a new OpenAIBackend instance configured with the provided API key, model, and timeout.
func NewOpenAIBackend(apiKey, model string, timeout time.Duration) *OpenAIBackend {
	// Use defaultTimeout if the user passes 0 as the timeout value
	if timeout == 0 {
		timeout = defaultTimeout
	}

	return &OpenAIBackend{
		APIKey: apiKey,
		Model:  model,
		HTTPClient: &http.Client{
			Timeout: timeout, // Use the user-specified or default timeout here
		},
		BaseURL: "https://api.openai.com",
	}
}

// OpenAIResponse represents the structure of the response received from the OpenAI API
// for a chat completion request. It contains information about the generated text,
// usage statistics, and other metadata related to the API call.
type OpenAIResponse struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	Model   string `json:"model"`
	Choices []struct {
		Index   int `json:"index"`
		Message struct {
			Role      string           `json:"role"`
			Content   string           `json:"content"`
			ToolCalls []OpenAIToolCall `json:"tool_calls"`
		} `json:"message"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
	} `json:"usage"`
}

// OpenAIResponseMessage represents the message part of the response from the OpenAI API.
type OpenAIResponseMessage struct {
	Role      string           `json:"role"`
	Content   string           `json:"content"`
	ToolCalls []OpenAIToolCall `json:"tool_calls"`
}

// OpenAIToolCall represents the structure of a tool call made by the assistant.
type OpenAIToolCall struct {
	ID       string             `json:"id"`       // The unique ID of the tool call.
	Type     string             `json:"type"`     // The type of tool call (e.g., "function").
	Function OpenAIToolFunction `json:"function"` // The function being called.
}

// OpenAIToolFunction represents the function call made within a tool call.
type OpenAIToolFunction struct {
	Name      string `json:"name"`      // The name of the function being invoked.
	Arguments string `json:"arguments"` // The JSON string containing the arguments for the function.
}

// Converse sends a series of messages to the OpenAI API and returns the generated response.
func (o *OpenAIBackend) Converse(ctx context.Context, prompt *Prompt) (PromptResponse, error) {
	msgMap, err := prompt.AsMap()
	if err != nil {
		return PromptResponse{}, fmt.Errorf("failed to convert messages to map: %w", err)
	}

	toolMap, err := prompt.Tools.ToolsMap()
	if err != nil {
		return PromptResponse{}, fmt.Errorf("failed to convert tools to map: %w", err)
	}
	url := o.BaseURL + "/v1/chat/completions"
	reqBody := map[string]any{
		"model":    o.Model,
		"messages": msgMap,
		"stream":   false,
		"tools":    toolMap,
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
	req.Header.Set("Authorization", "Bearer "+o.APIKey)

	resp, err := o.HTTPClient.Do(req)
	if err != nil {
		return PromptResponse{}, fmt.Errorf("HTTP request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return PromptResponse{}, fmt.Errorf("failed to generate response from OpenAI: status code %d, response: %s", resp.StatusCode, string(bodyBytes))
	}

	var result OpenAIResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return PromptResponse{}, fmt.Errorf("failed to decode response: %w", err)
	}

	if len(result.Choices[0].Message.ToolCalls) == 0 {
		prompt.AddMessage("assistant", result.Choices[0].Message.Content)
		return PromptResponse{
			Role:    "assistant",
			Content: result.Choices[0].Message.Content,
		}, nil
	}

	response := PromptResponse{
		Role:      "tool",
		ToolCalls: make([]ToolCall, 0, len(result.Choices[0].Message.ToolCalls)),
	}
	for _, toolCall := range result.Choices[0].Message.ToolCalls {
		toolName := toolCall.Function.Name
		toolArgs := toolCall.Function.Arguments

		var parsedArgs map[string]interface{}
		err = json.Unmarshal([]byte(toolArgs), &parsedArgs)
		if err != nil {
			return PromptResponse{}, fmt.Errorf("failed to unmarshal tool arguments: %w", err)
		}

		toolResponse, err := prompt.Tools.ExecuteTool(toolName, parsedArgs)
		if err != nil {
			return PromptResponse{}, fmt.Errorf("failed to execute tool: %w", err)
		}

		// we also need to add the previous reply with the call ID to the conversation
		// todo: programatically convert
		prompt.AppendMessage(Message{
			Role: "assistant",
			Fields: map[string]any{
				"type": result.Choices[0].Message.ToolCalls[0].Type,
				"tool_calls": []map[string]any{
					{
						"id":   result.Choices[0].Message.ToolCalls[0].ID,
						"type": result.Choices[0].Message.ToolCalls[0].Type,
						"function": map[string]any{
							"name":      result.Choices[0].Message.ToolCalls[0].Function.Name,
							"arguments": result.Choices[0].Message.ToolCalls[0].Function.Arguments,
						},
					},
				},
			},
		})
		prompt.AppendMessage(Message{
			Role:    "tool",
			Content: toolResponse,
			Fields: map[string]any{
				"tool_call_id": result.Choices[0].Message.ToolCalls[0].ID,
			},
		})

		response.ToolCalls = append(response.ToolCalls, ToolCall{
			Function: FunctionCall{
				Name:      toolName,
				Arguments: parsedArgs,
				Result:    toolResponse,
			},
		})
	}
	return response, nil
}

// Generate sends a prompt to the OpenAI API and returns the generated response.
//
// Parameters:
//   - ctx: The context for the API request, which can be used for cancellation.
//   - prompt: The input text prompt for which to generate a response.
//
// Returns:
//   - A string containing the generated response from the OpenAI model.
//   - An error if the API request fails or if there's an issue processing the response.

// Generate sends a structured prompt to the OpenAI API and returns the generated response.
//
// Parameters:
//   - ctx: The context for the API request, which can be used for cancellation.
//   - prompt: A structured prompt containing messages and parameters.
//
// Returns:
//   - A string containing the generated response from the OpenAI model.
//   - An error if the API request fails or if there's an issue processing the response.
func (o *OpenAIBackend) Generate(ctx context.Context, prompt *Prompt) (string, error) {
	timeoutCtx, cancel := context.WithTimeout(ctx, defaultTimeout)
	defer cancel()

	url := o.BaseURL + "/v1/chat/completions"
	reqBody := map[string]interface{}{
		"model":             o.Model,
		"messages":          prompt.Messages,
		"max_tokens":        prompt.Parameters.MaxTokens,
		"temperature":       prompt.Parameters.Temperature,
		"top_p":             prompt.Parameters.TopP,
		"frequency_penalty": prompt.Parameters.FrequencyPenalty,
		"presence_penalty":  prompt.Parameters.PresencePenalty,
	}

	reqBodyBytes, err := json.Marshal(reqBody)
	if err != nil {
		return "", fmt.Errorf("failed to marshal request body: %w", err)
	}

	req, err := http.NewRequestWithContext(timeoutCtx, "POST", url, bytes.NewBuffer(reqBodyBytes))
	if err != nil {
		return "", fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+o.APIKey)

	resp, err := o.HTTPClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("HTTP request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, err := io.ReadAll(resp.Body)
		if err != nil {
			return "", fmt.Errorf("failed to read response body: %w", err)
		}
		return "", fmt.Errorf("failed to generate response from OpenAI: "+
			"status code %d, response: %s", resp.StatusCode, string(bodyBytes))
	}

	var result OpenAIResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return "", fmt.Errorf("failed to decode response: %w", err)
	}

	return result.Choices[0].Message.Content, nil
}

// Embed generates an embedding vector for the given text using the OpenAI API.
//
// Parameters:
//   - ctx: The context for the API request, which can be used for cancellation.
//   - text: The input text to be embedded.
//
// Returns:
//   - A slice of float32 values representing the embedding vector.
//   - An error if the API request fails or if there's an issue processing the response.
func (o *OpenAIBackend) Embed(ctx context.Context, text string) ([]float32, error) {
	timeoutCtx, cancel := context.WithTimeout(ctx, defaultTimeout)
	defer cancel()
	url := o.BaseURL + "/v1/embeddings"
	reqBody := map[string]interface{}{
		"model": o.Model,
		"input": text,
	}

	reqBodyBytes, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request body: %w", err)
	}

	req, err := http.NewRequestWithContext(timeoutCtx, "POST", url, bytes.NewBuffer(reqBodyBytes))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+o.APIKey)

	resp, err := o.HTTPClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("HTTP request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, err := io.ReadAll(resp.Body)
		if err != nil {
			return nil, fmt.Errorf("failed to read response body: %w", err)
		}
		return nil, fmt.Errorf("failed to generate embedding from OpenAI: "+
			"status code %d, response: %s", resp.StatusCode, string(bodyBytes))
	}

	var result OpenAIEmbeddingResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return result.Data[0].Embedding, nil
}
