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
)

// Backend defines the interface for interacting with various LLM backends.
type Backend interface {
	Converse(ctx context.Context, prompt *Prompt) (PromptResponse, error)
	Generate(ctx context.Context, prompt *Prompt) (string, error)
	Embed(ctx context.Context, input string) ([]float32, error)
}

// Message represents a single role-based message in the conversation.
type Message struct {
	Role    string         `json:"role"`
	Content string         `json:"content"`
	Fields  map[string]any `json:"fields,omitempty"`
}

// Parameters defines generation settings for LLM completions.
type Parameters struct {
	MaxTokens        int     `json:"max_tokens"`
	Temperature      float64 `json:"temperature"`
	TopP             float64 `json:"top_p"`
	FrequencyPenalty float64 `json:"frequency_penalty"`
	PresencePenalty  float64 `json:"presence_penalty"`
}

// Prompt represents a structured prompt with role-based messages and parameters.
type Prompt struct {
	Messages   []Message  `json:"messages"`
	Parameters Parameters `json:"parameters"`
	// ToolRegistry is a map of tool names to their corresponding wrapper functions.
	Tools *ToolRegistry
}

// NewPrompt creates and returns a new Prompt.
func NewPrompt() *Prompt {
	return &Prompt{
		Messages:   make([]Message, 0),
		Parameters: Parameters{},
		Tools:      NewToolRegistry(),
	}
}

// AddMessage adds a message with a specific role to the prompt.
func (p *Prompt) AddMessage(role, content string) *Prompt {
	p.Messages = append(p.Messages, Message{Role: role, Content: content})
	return p
}

// AppendMessage adds a message with a specific role to the prompt.
func (p *Prompt) AppendMessage(msg Message) *Prompt {
	p.Messages = append(p.Messages, msg)
	return p
}

// SetParameters sets the generation parameters for the prompt.
func (p *Prompt) SetParameters(params Parameters) *Prompt {
	p.Parameters = params
	return p
}

// AsMap returns the conversation's messages as a list of maps.
func (p *Prompt) AsMap() ([]map[string]any, error) {
	messageList := make([]map[string]any, 0, len(p.Messages))
	for _, message := range p.Messages {
		msgMap := map[string]any{
			"role":    message.Role,
			"content": message.Content,
		}
		for k, v := range message.Fields {
			msgMap[k] = v
		}
		messageList = append(messageList, msgMap)
	}

	return messageList, nil
}

// FunctionCall represents a call to a function.
type FunctionCall struct {
	Name      string         `json:"name"`
	Arguments map[string]any `json:"arguments"`
	Result    any            `json:"result"`
}

// ToolCall represents a call to a tool.
type ToolCall struct {
	Function FunctionCall `json:"function"`
}

// PromptResponse represents a response from the AI in a conversation.
type PromptResponse struct {
	Role      string     `json:"role"`
	Content   string     `json:"content"`
	ToolCalls []ToolCall `json:"tool_calls"`
}
