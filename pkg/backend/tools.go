// Copyright 2024 Stacklok, Inc
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package backend

import (
	"fmt"
	"sync"
)

// ErrToolNotFound is returned when a tool is not found in the registry.
var ErrToolNotFound = fmt.Errorf("tool not found")

// ToolWrapper is a function type that wraps a tool's functionality.
type ToolWrapper func(args map[string]any) (string, error)

// Tool represents a tool that can be executed.
type Tool struct {
	Type     string       `json:"type"`
	Function ToolFunction `json:"function"`
}

// ToolFunction represents the function signature of a tool.
type ToolFunction struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	Parameters  map[string]any `json:"parameters"`
	Wrapper     ToolWrapper    `json:"-"`
}

// ToolRegistry manages the registration of tools and their corresponding wrapper functions.
type ToolRegistry struct {
	tools map[string]Tool
	m     sync.Mutex
}

// NewToolRegistry initializes a new ToolRegistry.
func NewToolRegistry() *ToolRegistry {
	return &ToolRegistry{
		tools: make(map[string]Tool),
	}
}

// RegisterTool allows the registration of a tool by name, expected parameters, and a wrapper function.
func (r *ToolRegistry) RegisterTool(t Tool) {
	r.m.Lock()
	r.tools[t.Function.Name] = t
	r.m.Unlock()
}

// ToolsMap returns a list of tools as a map of string to any. This is the format that both Ollama and OpenAI expect.
func (r *ToolRegistry) ToolsMap() ([]map[string]any, error) {
	toolList := make([]map[string]any, 0, len(r.tools))
	r.m.Lock()
	for _, tool := range r.tools {
		tMap, err := ToMap(tool)
		if err != nil {
			return nil, fmt.Errorf("failed to convert tool list to map: %w", err)
		}
		toolList = append(toolList, tMap)
	}
	r.m.Unlock()

	return toolList, nil
}

// ExecuteTool looks up a tool by name, checks the provided arguments, and calls the registered wrapper function.
func (r *ToolRegistry) ExecuteTool(toolName string, args map[string]any) (string, error) {
	r.m.Lock()
	defer r.m.Unlock()

	toolEntry, exists := r.tools[toolName]
	if !exists {
		return "", fmt.Errorf("%w: %s", ErrToolNotFound, toolName)
	}

	// Call the tool's wrapper function with the provided arguments
	return toolEntry.Function.Wrapper(args)
}
