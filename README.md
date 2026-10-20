# dhire-dhire

Slow, resumable, offline History study helper for local AI. Dhire Dhire turns notes into exam Q&A using a small Gemma model on a low-memory laptop.

## What it does

- Reads plain-text or markdown notes
- Splits content into manageable study chunks
- Uses Ollama's local Gemma 3 1B model to generate exam-style questions and answers
- Saves a checkpoint after each chunk so work can resume later
- Works even if Ollama is unavailable by falling back to deterministic local Q&A extraction

## Quick start

1. Install Ollama and pull a lightweight model:
   ```bash
   ollama pull gemma3:1b
   ```
2. Run the generator:
   ```bash
   python dhire_dhire.py --notes sample_notes/history_notes.txt --output output/qa.json --checkpoint output/checkpoint.json
   ```
3. To resume a partial run:
   ```bash
   python dhire_dhire.py --notes sample_notes/history_notes.txt --output output/qa.json --checkpoint output/checkpoint.json --resume
   ```

## Notes

This project targets the "slow but steady" workflow: one note chunk at a time, saved checkpoint by checkpoint, so it works on a 3.6 GB RAM laptop without a large offline model server.

## Project structure

- `dhire_dhire.py` – CLI entrypoint
- `sample_notes/history_notes.txt` – example study notes
- `output/` – generated Q&A and progress checkpoints

## License

MIT
