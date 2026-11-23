"""Central configuration. Everything runs locally."""

# Ollama (local LLM server) used for document summaries.
OLLAMA_URL = "http://localhost:11434"
SUMMARY_MODEL = "llama3.2:3b"  # the single place the summarizer model is chosen

# GLiNER NER model for PII detection.
GLINER_MODEL = "urchade/gliner_multi_pii-v1"

# Confidence given to company / letterhead / registered-office addresses. They are
# not personal data, so they surface as low-confidence review candidates only.
ORG_ADDRESS_CONFIDENCE = 0.5
