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

package weather

import (
	"encoding/json"
	"errors"
	"fmt"
	"github.com/stackloklabs/gollm/pkg/backend"
)

// Tool returns a backend.Tool object that can be used to interact with the weather tool.
func Tool() backend.Tool {
	return backend.Tool{
		Type: "function",
		Function: backend.ToolFunction{
			Name:        "weather",
			Description: "Get weather report for a city",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"city": map[string]any{
						"type":        "string",
						"description": "The city for which to get the weather report",
					},
				},
				"required": []string{"city"},
			},
			Wrapper: weatherReportWrapper,
		},
	}
}

func weatherReportWrapper(params map[string]any) (string, error) {
	city, ok := params["city"].(string)
	if !ok {
		return "", fmt.Errorf("city must be a string")
	}
	return weatherReport(city)
}

// WeatherReport defines the structure of the JSON response
type WeatherReport struct {
	City        string `json:"city"`
	Temperature string `json:"temperature"`
	Conditions  string `json:"conditions"`
}

// weatherReport returns a dummy weather report for the specified city in JSON format.
func weatherReport(city string) (string, error) {
	// in a real application, this data would be fetched from an external API
	weatherData := map[string]WeatherReport{
		"London":    {City: "London", Temperature: "15°C", Conditions: "Rainy"},
		"Stockholm": {City: "Stockholm", Temperature: "10°C", Conditions: "Sunny"},
		"Brno":      {City: "Brno", Temperature: "18°C", Conditions: "Clear skies"},
	}

	if report, ok := weatherData[city]; ok {
		jsonReport, err := json.Marshal(report)
		if err != nil {
			return "", err
		}
		return string(jsonReport), nil
	}

	return "", errors.New("city not found")
}
