# Minecraft-LLM
A project that runs the Llama 2 large language model entirely inside Minecraft using only vanilla commands

This project is inspired by [Andrej Karpathy/llama2.c](https://github.com/karpathy/llama2.c) and [Xiaodou's Math Lib](https://github.com/xiaodou8593/math2.0/). It implements full forward inference of a Llama 2 model inside vanilla Minecraft using only commands.

## Features
- **Pure Vanilla**: Works entirely with Minecraft commands, scoreboards, and storage; no mods or plugins.
- **Full LLaMA Architecture**: Supports GQA (Grouped Query Attention), RoPE (Rotary Position Embedding), RMSNorm, SiLU activation, and multiple Transformer layers.
- **Floating‑Point Emulation**: Encodes floats as `(base, exponent, sign)` triplets and implements addition, multiplication, division, exponentiation, trigonometric functions, etc., using scoreboard arithmetic.
- **Automatic Generation**: Python script reads `.bin` checkpoint & tokenizer files and produces a zip file that contains all required `.mcfunction` files; simply import them into your world.

## How to Use

Totally understand if you don't have a python environment or just want to feel the magic. You can download the pre-generated datapack `stories260K.zip` from this repo and jump to step 3. (generated from [this](https://huggingface.co/karpathy/tinyllamas/tree/main/stories260K) model)

### 1. Prepare the Model
Obtain a Llama 2 checkpoint and tokenizer in the `.bin` format (using the `export` script from `llama2.c`).

### 2. Generate Minecraft Datapack
```bash
python generate.py <checkpoint.bin> <tokenizer.bin> <output>
```
The script will generate a datapack named `<output>.zip`.

### 3. Import into Your World
Copy the generated `.zip` file into your Minecraft world’s `datapacks` directory.

### 4. Launch Minecraft
Version 1.21.11 is recommended, other versions are not guarenteed to work.

### 5. Run the Model
Run `/reload` after entering the world.

For auto-regressive generating, execute:
```
/function llm:generate {s:<steps>}
```
Where `<steps>` is the number of tokens you want the model to generate.

If you want to continue the generation process after it stops, simply run:
```
/function llm:generate_continue {s:<steps>}
```
Where `<steps>` is the number of additional tokens to generate.

## Running the model manually

You can also run the model manually if you want to use this as part of your other project. To initiallize the model, run:
```
/function llm:setup
```
This initialises scoreboard objectives, constants, parameters and clears the KV cache.

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
