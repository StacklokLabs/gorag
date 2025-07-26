-- Enable the pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create the table to store embeddings for OpenAI
CREATE TABLE openai_embeddings (
    id SERIAL PRIMARY KEY,
    doc_id TEXT NOT NULL,
    embedding VECTOR(1536), -- Replace 1536 with the actual size of your embeddings
    metadata JSONB
);

-- Create the table to store embeddings for Ollama
CREATE TABLE ollama_embeddings (
    id SERIAL PRIMARY KEY,
    doc_id TEXT NOT NULL,
    embedding VECTOR(1024), -- Replace 1024 with the actual size of your embeddings
    metadata JSONB
);

CREATE TABLE bedrock_embeddings_1024 (
    id BIGSERIAL PRIMARY KEY,
    doc_id TEXT NOT NULL,
    model TEXT NOT NULL,
    embedding VECTOR(1024) NOT NULL,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX bedrock_embeddings_1024_ivf_cos ON bedrock_embeddings_1024
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Index for efficient vector similarity search for OpenAI embeddings
CREATE INDEX openai_embeddings_idx ON openai_embeddings
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100); -- Tune 'lists' for better performance

-- Index for efficient vector similarity search for Ollama embeddings
CREATE INDEX ollama_embeddings_idx ON ollama_embeddings
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100); -- Tune 'lists' for better performance
