package backend

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"testing"

	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
)

type fakeClient struct {
	responses [][]byte
	err       error
	calls     int
	inputs    []*bedrockruntime.InvokeModelInput
}

func (f *fakeClient) InvokeModel(ctx context.Context, in *bedrockruntime.InvokeModelInput, _ ...func(*bedrockruntime.Options)) (*bedrockruntime.InvokeModelOutput, error) {
	f.calls++
	f.inputs = append(f.inputs, in)
	if f.err != nil {
		return nil, f.err
	}
	body := f.responses[f.calls-1]
	return &bedrockruntime.InvokeModelOutput{
		Body:        body,
		ContentType: awsString("application/json"),
	}, nil
}

func TestAwsString(t *testing.T) {
	v := "x"
	if got := awsString(v); got == nil || *got != v {
		t.Fatalf("awsString failed")
	}
}

func TestToFloat32Slice(t *testing.T) {
	got, err := toFloat32Slice([]any{1.2, float32(3.4)})
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	want := []float32{1.2, 3.4}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %v want %v", got, want)
	}
	_, err = toFloat32Slice([]any{"nope"})
	if err == nil {
		t.Fatalf("expected error")
	}
}

func TestParseEmbedding(t *testing.T) {
	b1, _ := json.Marshal(map[string]any{"embedding": []any{0.1, 0.2}})
	v, err := parseEmbedding(b1)
	if err != nil || !reflect.DeepEqual(v, []float32{0.1, 0.2}) {
		t.Fatalf("direct embedding failed: %v %v", v, err)
	}

	b2, _ := json.Marshal(map[string]any{"results": []any{
		map[string]any{"embedding": []any{1.0, 2.0, 3.0}},
	}})
	v, err = parseEmbedding(b2)
	if err != nil || !reflect.DeepEqual(v, []float32{1, 2, 3}) {
		t.Fatalf("results embedding failed: %v %v", v, err)
	}

	_, err = parseEmbedding([]byte(`{"foo":"bar"}`))
	if err == nil || err.Error() != "embedding not found" {
		t.Fatalf("want embedding not found, got %v", err)
	}
}

func TestGenerate_ErrorWhenNoModel(t *testing.T) {
	b := newBedrockBackendWithClient("", "", &fakeClient{})
	_, err := b.Generate(context.Background(), "p", nil)
	if err == nil || !strings.Contains(err.Error(), "text model is not set") {
		t.Fatalf("expected model not set error")
	}
}

func TestEmbed_ErrorWhenNoModel(t *testing.T) {
	b := newBedrockBackendWithClient("m", "", &fakeClient{})
	_, err := b.Embed(context.Background(), []string{"a"}, nil)
	if err == nil || !strings.Contains(err.Error(), "embed model is not set") {
		t.Fatalf("expected embed model not set error")
	}
}

func TestGenerate_SuccessBranches(t *testing.T) {
	makeResp := func(m map[string]any) []byte { b, _ := json.Marshal(m); return b }
	tests := []struct {
		name     string
		payload  map[string]any
		expected string
	}{
		{"outputText", map[string]any{"outputText": "ok1"}, "ok1"},
		{"completion", map[string]any{"completion": "ok2"}, "ok2"},
		{"generation", map[string]any{"generation": "ok3"}, "ok3"},
		{"content[0].text", map[string]any{"content": []any{map[string]any{"text": "ok4"}}}, "ok4"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			fc := &fakeClient{responses: [][]byte{makeResp(tt.payload)}}
			b := newBedrockBackendWithClient("text-model", "embed-model", fc)
			got, err := b.Generate(context.Background(), "hello", map[string]any{"temp": 0.1})
			if err != nil {
				t.Fatalf("err: %v", err)
			}
			if got != tt.expected {
				t.Fatalf("got %q want %q", got, tt.expected)
			}
			if fc.calls != 1 {
				t.Fatalf("expected 1 call")
			}
			var in map[string]any
			_ = json.Unmarshal(fc.inputs[0].Body, &in)
			if in["inputText"] != "hello" {
				t.Fatalf("prompt not forwarded")
			}
			if in["temp"] != 0.1 {
				t.Fatalf("params not forwarded")
			}
		})
	}
}

func TestGenerate_UnexpectedSchema(t *testing.T) {
	resp, _ := json.Marshal(map[string]any{"foo": "bar"})
	fc := &fakeClient{responses: [][]byte{resp}}
	b := newBedrockBackendWithClient("text-model", "embed-model", fc)
	_, err := b.Generate(context.Background(), "p", nil)
	if err == nil || !strings.Contains(err.Error(), "unexpected response schema") {
		t.Fatalf("expected unexpected response schema, got %v", err)
	}
}

func TestGenerateStream(t *testing.T) {
	resp, _ := json.Marshal(map[string]any{"outputText": "a b c"})
	fc := &fakeClient{responses: [][]byte{resp}}
	b := newBedrockBackendWithClient("text-model", "embed-model", fc)
	var got []string
	err := b.GenerateStream(context.Background(), "p", nil, func(s string) error {
		got = append(got, s)
		return nil
	})
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if strings.Join(got, "") != "a b c " {
		t.Fatalf("bad stream: %v", got)
	}
}

func TestEmbed_SuccessAndInvalidElement(t *testing.T) {
	okBody, _ := json.Marshal(map[string]any{"embedding": []any{0.1, 0.2}})
	badBody, _ := json.Marshal(map[string]any{"embedding": []any{"x"}})
	fc := &fakeClient{responses: [][]byte{okBody, badBody}}
	b := newBedrockBackendWithClient("t", "e", fc)

	vecs, err := b.Embed(context.Background(), []string{"a"}, nil)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if len(vecs) != 1 || !reflect.DeepEqual(vecs[0], []float32{0.1, 0.2}) {
		t.Fatalf("unexpected vecs: %v", vecs)
	}

	_, err = b.Embed(context.Background(), []string{"b"}, nil)
	if err == nil {
		t.Fatalf("expected error on invalid element")
	}
}

func TestFakeClient_PropagatesError(t *testing.T) {
	fc := &fakeClient{err: errors.New("boom")}
	b := newBedrockBackendWithClient("t", "e", fc)
	_, err := b.Generate(context.Background(), "p", nil)
	if err == nil || !strings.Contains(err.Error(), "boom") {
		t.Fatalf("expected boom")
	}
}
