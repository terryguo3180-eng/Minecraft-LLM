# Minecraft-LLM
A project that runs the Llama 2 large language model entirely inside Minecraft using only vanilla commands

This project is inspired by [Andrej Karpathy/llama2.c](https://github.com/karpathy/llama2.c) and [Xiaodou's Math Lib](https://github.com/xiaodou8593/math3.0/). It implements full forward inference & tokenizer & sampler of a Llama 2 model inside vanilla Minecraft using only commands.

## Features
- **Pure Vanilla**: Works entirely with Minecraft commands, scoreboards, and storage; no mods or plugins.
- **Full LLaMA Architecture**: Supports GQA (Grouped Query Attention), RoPE (Rotary Position Embedding), RMSNorm, SiLU activation, and multiple Transformer layers.
- **KV Caching**: Efficient attention over a sliding context window.
- **Tokenization**: BPE encoding directly inside Minecraft using the provided tokenizer file.
- **Sampling**: Argmax or temperature‑based (0–100 scale).
- **Progress bar**: Shows initialization, inference, and sampling steps.
- **Resume / stop**: Interrupt generation and resume later.

## How to Use

If you don't have a python environment or just want to get it running, you can download the pre-generated datapack `stories260K.zip` from this repo and go to step 3. (generated from [this](https://huggingface.co/karpathy/tinyllamas/tree/main/stories260K) model)

### 1. Prepare the Model
Obtain a Llama 2 checkpoint and tokenizer in the `.bin` format (using the `export` script from `llama2.c`).

### 2. Generate Minecraft Datapack
```bash
python generate.py <checkpoint.bin> <tokenizer.bin> <output.zip>
```
The script will generate a datapack named `<output.zip>`.

### 3. Import into Your World
Place the generated .zip file into your world’s datapacks/ folder (no need to unzip).

### 4. Launch Minecraft
Version 1.21.11 is recommended, other versions are not guarenteed to work.

### 5. Run the Model
Run `/reload` after entering the world.

Use the `/function` command with the following syntax:
```
/function llm:generate {s:<steps>,t:<temperature>,i:"<prompt>"}
```
- `s`: number of tokens to generate (positive integer)
- `t`: temperature (0 = argmax, 1–100 = scaled temperature, e.g., 50 ≈ 1.0)
- `i`: input prompt (string, supports any Unicode characters the tokenizer knows)

#### Example:
```
/function llm:generate {s:50,t:50,i:"Once upon a time"}
```
The model will start generating token by token, printing the output in chat (clearing the screen with newlines each step). A bossbar shows progress.

#### Additional commands:
- `/function llm:stop`: immediately stop generation
- `/function llm:resume {s:<additional_steps>}`: resume generation from the current state (e.g., after a server restart)

## Performance and Limitations
Inference speed: Extremely slow. A 15M parameter model takes ~20 minutes per token on my laptop.
Context window: Limited by sequence length from the checkpoint (typically 512, 1024, or 2048). The KV cache is allocated statically.
Model size: The datapack stores all weights in storage, larger models may crash the game.
Command limit – For larger models, even with tick splitting, some operations may still surpass the 32 bit `max_command_sequence_length` limit and truncate the execution.
Recommended model size: Use tiny models that fit within a few hundred MB of datapack size and run (slowly) on a single player world.

## Acknowledgements

- [Andrej Karpathy](https://github.com/karpathy) for `llama2.c` – the minimal LLaMA inference implementation and weight format.
- [Xiaodou's Math Lib](https://github.com/xiaodou8593/math2.0/) for the ingenious scoreboard‑based floating‑point emulation.
- The Minecraft command community for pushing the limits of vanilla functionality.

## License

This project is for educational and entertainment purposes. Use at your own risk – it may cause lag, crashes, or spontaneous combustion of your Minecraft server.

---
