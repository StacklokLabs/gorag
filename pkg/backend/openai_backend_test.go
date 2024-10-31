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
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func TestGenerate(t *testing.T) {
	t.Parallel()
	// Mock response from OpenAI API
	mockResponse := OpenAIResponse{
		ID:      "test-id",
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   "gpt-3.5-turbo",
		Choices: []struct {
			Index   int `json:"index"`
			Message struct {
				Role      string           `json:"role"`
				Content   string           `json:"content"`
				ToolCalls []OpenAIToolCall `json:"tool_calls"`
			} `json:"message"`
			FinishReason string `json:"finish_reason"`
		}{
			{
				Index: 0,
				Message: struct {
					Role      string           `json:"role"`
					Content   string           `json:"content"`
					ToolCalls []OpenAIToolCall `json:"tool_calls"`
				}{
					Role:    "assistant",
					Content: "This is a test response.",
				},
				FinishReason: "stop",
			},
		},
		Usage: struct {
			PromptTokens     int `json:"prompt_tokens"`
			CompletionTokens int `json:"completion_tokens"`
			TotalTokens      int `json:"total_tokens"`
		}{
			PromptTokens:     5,
			CompletionTokens: 5,
			TotalTokens:      10,
		},
	}

	mockServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Check the request method and URL path
		if r.Method != "POST" || r.URL.Path != "/v1/chat/completions" {
			t.Errorf("Unexpected request: %s %s", r.Method, r.URL.Path)
		}

		// Check headers
		if r.Header.Get("Content-Type") != "application/json" {
			t.Errorf("Expected Content-Type application/json, got %s", r.Header.Get("Content-Type"))
		}
		if r.Header.Get("Authorization") != "Bearer test-api-key" {
			t.Errorf("Expected Authorization Bearer test-api-key, got %s", r.Header.Get("Authorization"))
		}

		// Write the mock response
		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(mockResponse); err != nil {
			t.Errorf("Failed to encode mock response: %v", err)
		}
		if err := json.NewEncoder(w).Encode(mockResponse); err != nil {
			t.Errorf("Failed to encode mock response: %v", err)
		}
	}))
	defer mockServer.Close()

	// Create an instance of OpenAIBackend with the mock server
	backend := &OpenAIBackend{
		APIKey:     "test-api-key",
		Model:      "gpt-3.5-turbo",
		HTTPClient: mockServer.Client(),
		BaseURL:    mockServer.URL,
	}

	ctx := context.Background()
	msgPrompt := "Hello, openAI!"

	prompt := NewPrompt().
		AddMessage("system", "You are an AI assistant.").
		AddMessage("user", msgPrompt).
		SetParameters(Parameters{
			MaxTokens:        150,
			Temperature:      0.7,
			TopP:             0.9,
			FrequencyPenalty: 0.5,
			PresencePenalty:  0.6,
		})

	response, err := backend.Generate(ctx, prompt)
	if err != nil {
		t.Fatalf("Generate returned error: %v", err)
	}

	// Validate the response content as a string
	expectedContent := mockResponse.Choices[0].Message.Content
	if response != expectedContent {
		t.Errorf("Expected response '%s', got '%s'", expectedContent, response)
	}
}

func TestGenerateEmbedding(t *testing.T) {
	t.Parallel()
	// Mock response from OpenAI API
	mockResponse := OpenAIEmbeddingResponse{
		Object: "list",
		Data: []struct {
			Object    string    `json:"object"`
			Embedding []float32 `json:"embedding"`
			Index     int       `json:"index"`
		}{
			{
				Object:    "embedding",
				Embedding: []float32{0.1, 0.2, 0.3},
				Index:     0,
			},
		},
		Model: "text-embedding-ada-002",
		Usage: struct {
			PromptTokens int `json:"prompt_tokens"`
			TotalTokens  int `json:"total_tokens"`
		}{
			PromptTokens: 5,
			TotalTokens:  5,
		},
	}

	mockServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Check the request method and URL path
		if r.Method != "POST" || r.URL.Path != "/v1/embeddings" {
			t.Errorf("Unexpected request: %s %s", r.Method, r.URL.Path)
		}

		// Check headers
		if r.Header.Get("Content-Type") != "application/json" {
			t.Errorf("Expected Content-Type application/json, got %s", r.Header.Get("Content-Type"))
		}
		if r.Header.Get("Authorization") != "Bearer test-api-key" {
			t.Errorf("Expected Authorization Bearer test-api-key, got %s", r.Header.Get("Authorization"))
		}

		// Write the mock response
		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(mockResponse); err != nil {
			t.Errorf("Failed to encode mock response: %v", err)
		}
	}))
	defer mockServer.Close()

	// Create an instance of OpenAIBackend with the mock server
	backend := &OpenAIBackend{
		APIKey:     "test-api-key",
		HTTPClient: mockServer.Client(),
		BaseURL:    mockServer.URL,
	}

	ctx := context.Background()
	text := "Test embedding text."

	embedding, err := backend.Embed(ctx, text)
	if err != nil {
		t.Fatalf("GenerateEmbedding returned error: %v", err)
	}

	// Validate the response embedding as a slice of float32
	expectedEmbedding := mockResponse.Data[0].Embedding
	if len(embedding) != len(expectedEmbedding) {
		t.Errorf("Expected embedding length %d, got %d", len(expectedEmbedding), len(embedding))
	}
	for i, v := range embedding {
		if v != expectedEmbedding[i] {
			t.Errorf("Expected embedding[%d] = %f, got %f", i, expectedEmbedding[i], v)
		}
	}
}
