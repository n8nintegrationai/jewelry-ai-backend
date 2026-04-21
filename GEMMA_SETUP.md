# Gemma 4 E2B Setup

## Install Ollama

1. Download and install Ollama from https://ollama.com/download
2. Verify the install:

```powershell
ollama --version
```

## Pull the Gemma model

Use the Gemma 4 E2B Ollama model before starting the app:

```powershell
ollama pull gemma4:e2b
```

This Ollama model is distributed as a GGUF quantized build suitable for low-memory local inference. Keep the app pointed at `gemma4:e2b` to stay within the intended under-3GB footprint.

## Start the Ollama server

Start Ollama before running the backend so the API is available at `http://localhost:11434`:

```powershell
ollama serve
```

Then start the FastAPI app in a separate terminal using your normal project command.

## Revert to Phi-3

If you want to switch back to the previous Phi-3 implementation:

```powershell
git checkout main
```
