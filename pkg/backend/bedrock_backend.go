package backend

import (
	"context"
	"encoding/json"
	"errors"
	"strings"

	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
)

type bedrockInvoker interface {
	InvokeModel(ctx context.Context, params *bedrockruntime.InvokeModelInput, optFns ...func(*bedrockruntime.Options)) (*bedrockruntime.InvokeModelOutput, error)
}

type BedrockBackend struct {
	client     bedrockInvoker
	textModel  string
	embedModel string
}

func NewBedrockBackend(ctx context.Context, region, textModel, embedModel string) (*BedrockBackend, error) {
	cfg, err := config.LoadDefaultConfig(ctx, config.WithRegion(region))
	if err != nil {
		return nil, err
	}
	return &BedrockBackend{
		client:     bedrockruntime.NewFromConfig(cfg),
		textModel:  textModel,
		embedModel: embedModel,
	}, nil
}

func newBedrockBackendWithClient(textModel, embedModel string, c bedrockInvoker) *BedrockBackend {
	return &BedrockBackend{
		client:     c,
		textModel:  textModel,
		embedModel: embedModel,
	}
}

func (b *BedrockBackend) Generate(ctx context.Context, prompt string, params map[string]any) (string, error) {
	if b.textModel == "" {
		return "", errors.New("text model is not set")
	}
	body := map[string]any{
		"inputText": prompt,
	}
	for k, v := range params {
		body[k] = v
	}
	req, err := json.Marshal(body)
	if err != nil {
		return "", err
	}
	out, err := b.client.InvokeModel(ctx, &bedrockruntime.InvokeModelInput{
		ModelId:     awsString(b.textModel),
		ContentType: awsString("application/json"),
		Body:        req,
	})
	if err != nil {
		return "", err
	}
	return parseText(out.Body)
}

func (b *BedrockBackend) GenerateStream(ctx context.Context, prompt string, params map[string]any, onToken func(string) error) error {
	txt, err := b.Generate(ctx, prompt, params)
	if err != nil {
		return err
	}
	for _, t := range strings.Split(txt, " ") {
		if err := onToken(t + " "); err != nil {
			return err
		}
	}
	return nil
}

func (b *BedrockBackend) Embed(ctx context.Context, texts []string, params map[string]any) ([][]float32, error) {
	if b.embedModel == "" {
		return nil, errors.New("embed model is not set")
	}
	outVecs := make([][]float32, 0, len(texts))
	for _, t := range texts {
		body := map[string]any{
			"inputText": t,
		}
		for k, v := range params {
			body[k] = v
		}
		req, err := json.Marshal(body)
		if err != nil {
			return nil, err
		}
		out, err := b.client.InvokeModel(ctx, &bedrockruntime.InvokeModelInput{
			ModelId:     awsString(b.embedModel),
			ContentType: awsString("application/json"),
			Body:        req,
		})
		if err != nil {
			return nil, err
		}
		vec, err := parseEmbedding(out.Body)
		if err != nil {
			return nil, err
		}
		outVecs = append(outVecs, vec)
	}
	return outVecs, nil
}
func parseText(b []byte) (string, error) {
	var m map[string]any
	if err := json.Unmarshal(b, &m); err != nil {
		return "", err
	}

	// Titan: {"results":[{"outputText":"..."}]}
	if r, ok := m["results"].([]any); ok && len(r) > 0 {
		for _, it := range r {
			if mm, ok := it.(map[string]any); ok {
				if s, ok := mm["outputText"].(string); ok {
					return s, nil
				}
				if s, ok := mm["text"].(string); ok {
					return s, nil
				}

				if msg, ok := mm["message"].(map[string]any); ok {
					if content, ok := msg["content"].([]any); ok && len(content) > 0 {
						if c0, ok := content[0].(map[string]any); ok {
							if s, ok := c0["text"].(string); ok {
								return s, nil
							}
						}
					}
				}
			}
		}
	}

	if s, ok := m["outputText"].(string); ok {
		return s, nil
	}
	if s, ok := m["completion"].(string); ok {
		return s, nil
	}
	if s, ok := m["generation"].(string); ok {
		return s, nil
	}
	if arr, ok := m["content"].([]any); ok && len(arr) > 0 {
		if mm, ok := arr[0].(map[string]any); ok {
			if s, ok := mm["text"].(string); ok {
				return s, nil
			}
		}
	}

	return "", errors.New("unexpected response schema")
}

func parseEmbedding(b []byte) ([]float32, error) {
	var m map[string]any
	if err := json.Unmarshal(b, &m); err != nil {
		return nil, err
	}
	if v, ok := m["embedding"].([]any); ok {
		return toFloat32Slice(v)
	}
	if r, ok := m["results"].([]any); ok && len(r) > 0 {
		if mm, ok := r[0].(map[string]any); ok {
			if v, ok := mm["embedding"].([]any); ok {
				return toFloat32Slice(v)
			}
		}
	}
	return nil, errors.New("embedding not found")
}

func toFloat32Slice(v []any) ([]float32, error) {
	res := make([]float32, len(v))
	for i, x := range v {
		switch n := x.(type) {
		case float64:
			res[i] = float32(n)
		case float32:
			res[i] = n
		default:
			return nil, errors.New("invalid embedding element type")
		}
	}
	return res, nil
}

func awsString(s string) *string {
	return &s
}
