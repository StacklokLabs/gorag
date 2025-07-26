# Bedrock Example

Minimal example that uses **AWS Bedrock** through the `BedrockBackend` to:
1) generate text
2) create embeddings

## Prerequisites

- Go 1.22+
- AWS credentials configured (env/SharedConfig)
- Environment variables:
  - `AWS_REGION` (e.g. `us-east-1`)
  - `BEDROCK_TEXT_MODEL` (e.g. `anthropic.claude-3-haiku-20240307-v1:0`)
  - `BEDROCK_EMBED_MODEL` (e.g. `amazon.titan-embed-text-v1`)

## Run

```bash
export AWS_REGION=us-east-1
export BEDROCK_TEXT_MODEL=anthropic.claude-3-haiku-20240307-v1:0
export BEDROCK_EMBED_MODEL=amazon.titan-embed-text-v1

go run ./examples/bedrock "Explain vector databases in 2 sentences."
