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

package main

import (
	"context"
	"log"
	"os"
	"strings"
	"time"

	"github.com/stackloklabs/gollm/examples/tools/weather"
	"github.com/stackloklabs/gollm/pkg/backend"
)

var (
	ollamaHost     = "http://localhost:11434"
	ollamaGenModel = "qwen2.5"
	openaiModel    = "gpt-4o-mini"
)

const (
	systemMessage = `
You are an AI assistant that can retrieve weather forecasts by calling a tool
that returns weather data in JSON format. You will be provided with a city
name, and you will use a tool to call out to a weather forecast API that
provides the weather for that city. The tool returns a JSON object with three
fields: city, temperature, and conditions.
`
	summarizeMessage = `
Summarize the tool's forecast of the weather in clear, plain language for the user. 
`
)

func main() {
	var generationBackend backend.Backend

	beSelection := os.Getenv("BACKEND")
	if beSelection == "" {
		log.Println("No backend selected with the BACKEND env variable. Defaulting to Ollama.")
		beSelection = "ollama"
	}
	modelSelection := os.Getenv("MODEL")
	if modelSelection == "" {
		switch beSelection {
		case "ollama":
			modelSelection = ollamaGenModel
		case "openai":
			modelSelection = openaiModel
		}
		log.Println("No model selected with the MODEL env variable. Defaulting to ", modelSelection)
	}

	switch beSelection {
	case "ollama":
		generationBackend = backend.NewOllamaBackend(ollamaHost, ollamaGenModel, 30*time.Second)
		log.Println("Using Ollama backend: ", ollamaGenModel)
	case "openai":
		openaiKey := os.Getenv("OPENAI_API_KEY")
		if openaiKey == "" {
			log.Fatalf("OPENAI_API_KEY is required for OpenAI backend")
		}
		generationBackend = backend.NewOpenAIBackend(openaiKey, openaiModel, 30*time.Second)
		log.Println("Using OpenAI backend: ", openaiModel)
	default:
		log.Fatalf("Unknown backend: %s", beSelection)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	userPrompt := os.Args[1:]
	if len(userPrompt) == 0 {
		log.Fatalf("Please provide a prompt")
	}

	convo := backend.NewPrompt()
	convo.Tools.RegisterTool(weather.Tool())
	// start the conversation. We add a system message to tune the output
	// and add the weather tool to the conversation so that the model knows to call it.
	convo.AddMessage("system", systemMessage)
	convo.AddMessage("user", strings.Join(userPrompt, " "))

	// generate the response
	resp, err := generationBackend.Converse(ctx, convo)
	if err != nil {
		log.Fatalf("Error generating response: %v", err)
	}

	if len(resp.ToolCalls) == 0 {
		log.Println("No tool calls in response.")
		log.Println("Response:", convo.Messages[len(convo.Messages)-1].Content)
		return
	}

	log.Println("Tool called")

	// if there was a tool response, first just feed it back to the model so it makes sense of it
	_, err = generationBackend.Converse(ctx, convo)
	if err != nil {
		log.Fatalf("Error generating response: %v", err)
	}

	log.Println("Response:")
	log.Println(convo.Messages[len(convo.Messages)-1].Content)
}
