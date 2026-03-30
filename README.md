# Minecraft-LLM
A project that runs the Llama 2 large language model entirely inside Minecraft using only vanilla commands

This project is inspired by [Andrej Karpathy/llama2.c](https://github.com/karpathy/llama2.c) and [Xiaodou's Math Lib](https://github.com/xiaodou8593/math2.0/). It implements full forward inference of a Llama 2 model inside vanilla Minecraft using only commands.

## Features
- **Pure Vanilla**: Works entirely with Minecraft commands, scoreboards, and storage; no mods or plugins.
- **Full LLaMA Architecture**: Supports GQA (Grouped Query Attention), RoPE (Rotary Position Embedding), RMSNorm, SiLU activation, and multiple Transformer layers.
- **Floating‑Point Emulation**: Encodes floats as `(base, exponent, sign)` triplets and implements addition, multiplication, division, exponentiation, trigonometric functions, etc., using scoreboard arithmetic.
- **Automatic Generation**: Python script reads `.bin` checkpoint & tokenizer files and produces a zip file that contains all required `.mcfunction` files; simply import them into your world.

## How to Use

### 1. Prepare the Model
Obtain a Llama 2 checkpoint and tokenizer in the `.bin` format (e.g., using the `export` script from `llama2.c`).

### 2. Generate Minecraft Datapack
```bash
python generate.py <checkpoint.bin> <tokenizer.bin> <output>
```
The script will generate the datapack with the name of `<output>.zip`.

### 3. Import into Your World
Copy the generated `.zip` file into your Minecraft world’s `datapacks` directory.

### 4. Run the Model
In‑game, execute:
```
/reload
/function llm:setup
```
This initialises scoreboard objectives, constants, parameters, and clears the KV cache.

Set the starting token & position, and clear the token buffer:
```
/scoreboard players set tok llm 1
/scoreboard players set pos llm 0
/data modify storage llm args.tokens set value []
```
Then run inference:
```
/function llm:forward
```
The process may take a few minutes to hours depending on model size. After it finishes, run the following commands to sample:
```
/function llm:argmax
/function llm:sample
```
The generated token(s) are stored in `storage llm args.tokens`. View them with:
```
/tellraw @a {storage:"llm",nbt:"args.tokens[]",separator:""}
```

To generate the next token, simply copy the score in `max_i llm` to `tok llm` and increment `pos llm` by 1, and run inference again:
```
/scoreboard players operation tok llm = max_i llm
/scoreboard players add pos llm 1

/function llm:forward
```

## Limitations

- **Performance**: Inference is really, really slow, a 15M model would take ~20 minutes to complete a forward pass on my laptop.
- **Sampling**: Currently only argmax is implemented; no stochastic sampling.

## Acknowledgements

- [Andrej Karpathy](https://github.com/karpathy) for `llama2.c` – the minimal LLaMA inference implementation and weight format.
- [Xiaodou's Math Lib](https://github.com/xiaodou8593/math2.0/) for the ingenious scoreboard‑based floating‑point emulation.
- The Minecraft command community for pushing the limits of vanilla functionality.

## License

MIT

---
