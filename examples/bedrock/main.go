package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/stackloklabs/gorag/pkg/backend"
)

func getenv(key string) string {
	v := os.Getenv(key)
	if v == "" {
		log.Fatalf("missing env %s", key)
	}
	return v
}

func main() {
	region := getenv("AWS_REGION")
	textModel := getenv("BEDROCK_TEXT_MODEL")
	embedModel := getenv("BEDROCK_EMBED_MODEL")

	prompt := "Explain Retrieval-Augmented Generation (RAG) in 3 sentences."
	if len(os.Args) > 1 {
		prompt = os.Args[1]
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	br, err := backend.NewBedrockBackend(ctx, region, textModel, embedModel)
	if err != nil {
		log.Fatalf("init bedrock backend: %v", err)
	}

	out, err := br.Generate(ctx, prompt, map[string]any{
		"textGenerationConfig": map[string]any{
			"maxTokenCount": 512,
			"temperature":   0.2,
			"topP":          0.9,
			"stopSequences": []string{},
		},
	})

	if err != nil {
		log.Fatalf("generate: %v", err)
	}
	fmt.Println("=== Completion ===")
	fmt.Println(out)

	vecs, err := br.Embed(ctx, []string{
		"RAG augments LLMs with external knowledge.",
		"Vector databases enable efficient similarity search.",
	}, nil)
	if err != nil {
		log.Fatalf("embed: %v", err)
	}
	fmt.Println("=== Embeddings ===")
	fmt.Printf("count=%d dim=%d\n", len(vecs), len(vecs[0]))
}
