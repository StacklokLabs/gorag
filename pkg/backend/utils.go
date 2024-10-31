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
	"encoding/json"
	"fmt"
)

// ToMap converts the given value to a map[string]any. This is useful for working with JSON data.
func ToMap(v any) (map[string]any, error) {
	data, err := json.Marshal(v)
	if err != nil {
		return nil, fmt.Errorf("error marshaling to JSON: %v", err)
	}

	var mapResult map[string]any
	err = json.Unmarshal(data, &mapResult)
	if err != nil {
		return nil, fmt.Errorf("error unmarshaling JSON to map: %v", err)
	}

	return mapResult, nil
}

// PrintJSON prints the given value as a JSON string. Useful for debugging.
func PrintJSON(v any) {
	data, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		fmt.Printf("error marshaling to JSON: %v", err)
		return
	}

	fmt.Println(string(data))
}
