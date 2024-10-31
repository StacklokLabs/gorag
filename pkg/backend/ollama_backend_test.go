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

const contentTypeJSON = "application/json"
const testEmbeddingText = "Test embedding text."

func TestOllamaGenerate(t *testing.T) {
	t.Parallel()
	// Mock response from Ollama API
	mockResponse := OllamaResponse{
		Model:     "test-model",
		CreatedAt: time.Now().Format(time.RFC3339),
		Response:  "This is a test response from Ollama.",
		Done:      true,
	}

	// Create a mock server to simulate the Ollama API
	mockServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Validate the request
		if r.Method != http.MethodPost || r.URL.Path != generateEndpoint {
			t.Errorf("Unexpected request: %s %s", r.Method, r.URL.Path)
		}

		// Check Content-Type header
		if r.Header.Get("Content-Type") != contentTypeJSON {
			t.Errorf("Expected Content-Type application/json, got %s", r.Header.Get("Content-Type"))
		}

		// Decode the request body
		var reqBody map[string]interface{}
		if err := json.NewDecoder(r.Body).Decode(&reqBody); err != nil {
			t.Errorf("Failed to decode request body: %v", err)
		}

		// Check that the "prompt" field is correctly passed
		promptText, ok := reqBody["prompt"].(string)
		if !ok || promptText == "" {
			t.Errorf("Expected a valid prompt, got: %v", reqBody["prompt"])
		}

		// Write the mock response
		w.Header().Set("Content-Type", contentTypeJSON)
		if err := json.NewEncoder(w).Encode(mockResponse); err != nil {
			t.Errorf("Failed to encode mock response: %v", err)
		}
	}))
	defer mockServer.Close()

	// Create an instance of OllamaBackend with the mock server
	backend := &OllamaBackend{
		Model:   "test-model",
		Client:  mockServer.Client(),
		BaseURL: mockServer.URL,
	}

	ctx := context.Background()
	promptMsg := "Hello, Ollama!"

	// Construct the prompt
	prompt := NewPrompt().
		AddMessage("system", "You are an AI assistant.").
		AddMessage("user", promptMsg).
		SetParameters(Parameters{
			MaxTokens:        150,
			Temperature:      0.7,
			TopP:             0.9,
			FrequencyPenalty: 0.5,
			PresencePenalty:  0.6,
		})

	// Call the Generate method
	response, err := backend.Generate(ctx, prompt)
	if err != nil {
		t.Fatalf("Generate returned error: %v", err)
	}

	// Validate the response
	if response != mockResponse.Response {
		t.Errorf("Expected response '%s', got '%s'", mockResponse.Response, response)
	}
}

func TestOllamaEmbed(t *testing.T) {
	t.Parallel()
	// Mock response from Ollama API
	mockResponse := OllamaEmbeddingResponse{
		Embedding: []float32{0.1, 0.2, 0.3},
	}

	// Create a mock server to simulate the Ollama API
	mockServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Validate the request
		if r.Method != http.MethodPost || r.URL.Path != embedEndpoint {
			t.Errorf("Unexpected request: %s %s", r.Method, r.URL.Path)
		}

		// Check Content-Type header
		if r.Header.Get("Content-Type") != contentTypeJSON {
			t.Errorf("Expected Content-Type application/json, got %s", r.Header.Get("Content-Type"))
		}

		// Decode the request body
		var reqBody map[string]interface{}
		if err := json.NewDecoder(r.Body).Decode(&reqBody); err != nil {
			t.Errorf("Failed to decode request body: %v", err)
		}

		// Optional: Validate request body contents
		if reqBody["model"] != "test-model" {
			t.Errorf("Expected model 'test-model', got '%v'", reqBody["model"])
		}
		if reqBody["prompt"] != testEmbeddingText {
			t.Errorf("Expected prompt 'Test embedding text.', got '%v'", reqBody["prompt"])
		}

		// Write the mock response
		w.Header().Set("Content-Type", contentTypeJSON)
		if err := json.NewEncoder(w).Encode(mockResponse); err != nil {
			t.Errorf("Failed to encode mock response: %v", err)
		}
	}))
	defer mockServer.Close()

	// Create an instance of OllamaBackend with the mock server
	backend := &OllamaBackend{
		Model:   "test-model",
		Client:  mockServer.Client(),
		BaseURL: mockServer.URL,
	}

	ctx := context.Background()
	input := testEmbeddingText

	embedding, err := backend.Embed(ctx, input)
	if err != nil {
		t.Fatalf("Embed returned error: %v", err)
	}

	// Validate the response
	if len(embedding) != len(mockResponse.Embedding) {
		t.Errorf("Expected embedding length %d, got %d", len(mockResponse.Embedding), len(embedding))
	}
	for i, v := range embedding {
		if v != mockResponse.Embedding[i] {
			t.Errorf("Expected embedding[%d] = %f, got %f", i, mockResponse.Embedding[i], v)
		}
	}
}
