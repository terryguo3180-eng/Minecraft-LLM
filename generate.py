# Complete llama2 implementation in pure Minecraft commands
# TerryGuo 4/16/2026
# An mcfunction port of llama2.c

# v0.1: Added basic transformer forward logic, can generate text from scratch
# v0.2: Added temperature sampling & sentencepiece tokenizer, can generate text that follows a given
#       prompt, also with a bit of uncertainty
# v0.3: Added chat function, fixed a few bugs

import argparse
import json
import math
import shutil
import struct
import os
import zipfile


# Get command line arguments
parser = argparse.ArgumentParser(description="Generate Minecraft datapack for Llama 2")
parser.add_argument("checkpoint", help="Path to model checkpoint (.bin)")
parser.add_argument("tokenizer", help="Path to tokenizer (.bin)")
parser.add_argument("pack", help="Path to output datapack")
args = parser.parse_args()

checkpoint = args.checkpoint
tokenizer = args.tokenizer
pack = args.pack

if not pack.endswith('.zip'):
    print("error: datapack must be in .zip format")
    raise SystemExit(1)

pack = pack.rstrip('.zip')

if not os.path.exists(checkpoint):
    print(f"error: checkpoint file {checkpoint!r} does not exist")

if not os.path.exists(tokenizer):
    print(f"error: tokenizer file {tokenizer!r} does not exist")


# Read parameters from .bin file

def read_floats(count, file):
    return struct.unpack(str(count) + 'f', file.read(count * 4 if count > 0 else count))

with (
    open(checkpoint, "rb") as cfile,
    open(tokenizer, "rb") as tfile,
):
    _config = cfile.read(struct.calcsize('7i'))

    # Hyper-parameters for the model
    dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len = struct.unpack('7i', _config)
    vocab, vocab_scores = [], []

    struct.unpack('i', tfile.read(4))[0]  # Skip max_token_length

    # Get the vocabulary list
    for _ in range(0, vocab_size):
        vocab_scores.append(int(struct.unpack('f', tfile.read(4))[0]))
        length = struct.unpack('i', tfile.read(4))[0]
        bstr = tfile.read(length)
        if type(bstr) is not str:
            bstr = bstr.decode('utf8')
        
        # Weird llama feature
        if bstr.startswith('<0x') and bstr.endswith('>') and len(bstr) == 6:
            for c in bstr[3:4]:
                if c not in "1234567890ABCDEF":
                    break
            else:
                bstr = chr(int(bstr[1:5], 16))

        vocab.append(bstr)
    
    head_size = dim // n_heads

    # Read all the weights
    token_embedding_table = read_floats(vocab_size * dim, cfile)
    rms_att_weight = read_floats(n_layers * dim, cfile)
    wq = read_floats(n_layers * dim * dim, cfile)
    wk = read_floats(n_layers * dim * n_kv_heads * head_size, cfile)
    wv = read_floats(n_layers * dim * n_kv_heads * head_size, cfile)
    wo = read_floats(n_layers * dim * dim, cfile)
    rms_ffn_weight = read_floats(n_layers * dim, cfile)
    w1 = read_floats(n_layers * dim * hidden_dim, cfile)
    w2 = read_floats(n_layers * hidden_dim * dim, cfile)
    w3 = read_floats(n_layers * dim * hidden_dim, cfile)
    rms_final_weight = read_floats(dim, cfile)


base_dir = os.path.join(pack, "data", "llm", "function")
os.makedirs(base_dir, exist_ok=True)

print("Generating datapack...")


# Common constants for convenience
kv_dim = (dim * n_kv_heads) // n_heads
kv_mul = n_heads // n_kv_heads
head_size = dim // n_heads

# Set of constants for scoreboard operations
consts = {
    -1, 2, 10, 25, 40, 100, 500, 1000, 4750, 10000, 24703,
    79249, 100000, 1000000, 10000000, 100000000, 1000000000
}  # Constants used in the math lib
consts.add(dim)
consts.add(seq_len)


# Helper context class for writing .mcfunction files
class FunctionWritter:
    def __init__(self, name: str, max_linecount: int = 10000):
        self.name = name
        self.part_index = 0
        # Ensure the directory exists
        os.makedirs(base_dir, exist_ok=True)
        # Current file
        self.cfile = open(os.path.join(base_dir, f'{self.name}.mcfunction'), 'w', encoding='utf-8')
        self.line_count = 0
        self.max_linecount = max_linecount

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc, tb):
        self.close()

    def split(self, schedule: bool = False, schedule_ticks: int = 1) -> str:
        funcname = f'{self.name}_{self.part_index}'
        filename = os.path.join(base_dir, f'{funcname}.mcfunction')

        if self.cfile:
            if schedule:
                self.cfile.write(f'schedule function llm:{funcname} {schedule_ticks}t\n')
            else:
                self.cfile.write(f'function llm:{funcname}\n')
            self.cfile.close()

        self.cfile = open(filename, 'w', encoding='utf-8')
        self.line_count = 0
        self.part_index += 1

        return funcname

    def write(self, s: str):
        if not s.endswith('\n'):
            s += '\n'

        if self.line_count > self.max_linecount:
            self.split()

        self.cfile.write(s)
        self.line_count += 1
    
    def close(self):
        if self.cfile:
            self.cfile.close()


# Array element getters & setters & inplace operations
# ----------------------------------------------------

with FunctionWritter('get_kc') as f:
    f.write(f'$scoreboard players operation xb llm = $(i)b kc')
    f.write(f'$scoreboard players operation xe llm = $(i)e kc')
    f.write(f'$scoreboard players operation xs llm = $(i)s kc')

with FunctionWritter('get_vc') as f:
    f.write(f'$scoreboard players operation xb llm = $(i)b vc')
    f.write(f'$scoreboard players operation xe llm = $(i)e vc')
    f.write(f'$scoreboard players operation xs llm = $(i)s vc')

with FunctionWritter('get_av') as f:
    f.write(f'$scoreboard players operation xb llm = $(i)b av')
    f.write(f'$scoreboard players operation xe llm = $(i)e av')
    f.write(f'$scoreboard players operation xs llm = $(i)s av')

with FunctionWritter('set_kc') as f:
    f.write(f'$scoreboard players operation $(i)b kc = xb llm')
    f.write(f'$scoreboard players operation $(i)e kc = xe llm')
    f.write(f'$scoreboard players operation $(i)s kc = xs llm')

with FunctionWritter('set_vc') as f:
    f.write(f'$scoreboard players operation $(i)b vc = xb llm')
    f.write(f'$scoreboard players operation $(i)e vc = xe llm')
    f.write(f'$scoreboard players operation $(i)s vc = xs llm')

with FunctionWritter('set_av') as f:
    f.write(f'$scoreboard players operation $(i)b av = xb llm')
    f.write(f'$scoreboard players operation $(i)e av = xe llm')
    f.write(f'$scoreboard players operation $(i)s av = xs llm')


# Float operations with scoreboard
# Every function except exp are adapted from XiaoDou's Math Lib
# -------------------------------------------------------------

float_funcs = {
    "add": [
        "execute if score xb llm matches 0 run scoreboard players operation xe llm = ye llm",
        "execute if score yb llm matches 0 run scoreboard players operation ye llm = xe llm",
        "scoreboard players operation t0 llm = ys llm",
        "scoreboard players operation t0 llm *= xs llm",
        "scoreboard players operation t2 llm = yb llm",
        "function llm:float_0",
        "scoreboard players operation t3 llm = xe llm",
        "scoreboard players operation t3 llm -= ye llm",
        "execute if score t1 llm matches 1 run function llm:float_1",
        "scoreboard players operation t2 llm *= t0 llm",
        "execute unless score t3 llm matches 0 run function llm:float_2",
        "scoreboard players operation xb llm += t2 llm",
        "function llm:float_5",
        "execute if score xb llm matches 100000000.. run function llm:float_15",
    ], "cmp": [
        "execute if score xs llm > ys llm run return run scoreboard players set r llm 1",
        "execute if score xs llm < ys llm run return run scoreboard players set r llm -1",
        "function llm:float_14",
        "execute if score xs llm matches -1 run scoreboard players operation r llm *= -1 llm",
    ], "cos": [
        "data modify storage llm args.s set value \"\"",
        "execute if score xs llm matches -1 run data modify storage llm args.s set value \"-\"",
        "execute store result storage llm args.e int 1 run scoreboard players remove xe llm 7",
        "execute store result storage llm args.b int 1 run scoreboard players get xb llm",
        "execute summon marker run function llm:float_22 with storage llm args",
        "scoreboard players set xe llm 0",
        "scoreboard players set xs llm 1",
        "execute if score xb llm matches ..-1 run function llm:float_21",
        "function llm:float_5",
        "execute if score xb llm matches 100000000.. run function llm:float_15",
    ], "div": [
        "scoreboard players operation t0 llm = xb llm",
        "scoreboard players operation t0 llm %= 10000 llm",
        "scoreboard players operation xb llm *= 10 llm",
        "scoreboard players operation t1 llm = yb llm",
        "scoreboard players operation t2 llm = t1 llm",
        "scoreboard players operation t1 llm /= 25 llm",
        "scoreboard players operation t2 llm %= 25 llm",
        "scoreboard players operation t3 llm = xb llm",
        "scoreboard players operation xb llm /= t1 llm",
        "scoreboard players operation xb llm *= 40 llm",
        "scoreboard players operation t3 llm %= t1 llm",
        "scoreboard players operation t3 llm *= 40 llm",
        "scoreboard players operation t0 llm = t3 llm",
        "scoreboard players operation t3 llm /= t1 llm",
        "scoreboard players operation xb llm += t3 llm",
        "scoreboard players operation t0 llm %= t1 llm",
        "scoreboard players operation t0 llm *= 25 llm",
        "scoreboard players operation t4 llm = t2 llm",
        "scoreboard players operation t4 llm *= xb llm",
        "scoreboard players operation t0 llm -= t4 llm",
        "scoreboard players operation t0 llm *= 10 llm",
        "scoreboard players operation t3 llm = t0 llm",
        "scoreboard players operation t0 llm /= t1 llm",
        "scoreboard players operation t0 llm *= 40 llm",
        "scoreboard players operation t3 llm %= t1 llm",
        "scoreboard players operation t3 llm *= 40 llm",
        "scoreboard players operation t3 llm /= t1 llm",
        "scoreboard players operation t0 llm += t3 llm",
        "scoreboard players operation xs llm *= ys llm",
        "scoreboard players operation xe llm -= ye llm",
        "scoreboard players remove xe llm 1",
        "execute if score xs llm matches 0 run scoreboard players set xe llm 0",
        "execute if score xb llm matches 100000.. run return 0",
        "scoreboard players operation xb llm *= 10000 llm",
        "scoreboard players operation xb llm += t0 llm",
        "execute if score xb llm matches 100000000.. run function llm:float_15",
    ], "exp": [
        # exp(x) = 10^n * exp(f * log(10)), where n = floor(x), f = x - n
        "execute store result storage llm args.t int 4.3429448190325176 run scoreboard players get xb llm",
        "execute store result score xb llm run data get storage llm args.t",
        "execute if score xb llm matches 100000000.. run function llm:float_15",
        "scoreboard players remove xe llm 1",
        "scoreboard players operation t4 llm = xb llm",
        "scoreboard players operation t5 llm = xe llm",
        "scoreboard players operation t6 llm = xs llm",
        "function llm:floor",
        "execute if score t6 llm matches -1 run function llm:float_30",
        "scoreboard players operation yb llm = t4 llm",
        "scoreboard players operation ye llm = t5 llm",
        "scoreboard players operation ys llm = t6 llm",
        "scoreboard players operation ys llm *= -1 llm",
        "function llm:add",
        "scoreboard players operation xs llm *= -1 llm",
        "execute store result storage llm args.t int 2.3025850929940456840 run scoreboard players get xb llm",
        "execute store result score xb llm run data get storage llm args.t",
        "execute if score xb llm matches 100000000.. run function llm:float_15",

        # Chebyshev approximation for exp(x) when x is between (0, log(10))
        "scoreboard players operation t4 llm = xb llm",
        "scoreboard players operation t5 llm = xe llm",
        "scoreboard players operation t6 llm = xs llm",
        "execute store result storage llm args.t int 3.2546203518824833519 run scoreboard players get t4 llm",
        "execute store result score xb llm run data get storage llm args.t",
        "scoreboard players remove xe llm 4",
        "execute if score xb llm matches 100000000.. run function llm:float_15",
        "scoreboard players set yb llm 19813115",
        "scoreboard players set ye llm -3",
        "scoreboard players set ys llm 1",
        "scoreboard players operation t0 llm = ys llm",
        "scoreboard players operation t0 llm *= xs llm",
        "scoreboard players operation t2 llm = yb llm",
        "function llm:float_0",
        "scoreboard players operation t3 llm = xe llm",
        "scoreboard players operation t3 llm -= ye llm",
        "execute if score t1 llm matches 1 run function llm:float_1",
        "scoreboard players operation t2 llm *= t0 llm",
        "execute unless score t3 llm matches 0 run function llm:float_2",
        "scoreboard players operation xb llm += t2 llm",
        "function llm:float_5",
        "execute if score xb llm matches 100000000.. run function llm:float_15",
        "scoreboard players operation yb llm = t4 llm",
        "scoreboard players operation t0 llm = xb llm",
        "scoreboard players operation t0 llm %= 10000 llm",
        "scoreboard players operation xb llm /= 10000 llm",
        "scoreboard players operation t1 llm = xb llm",
        "scoreboard players operation t2 llm = yb llm",
        "scoreboard players operation t2 llm %= 10000 llm",
        "scoreboard players operation yb llm /= 10000 llm",
        "scoreboard players operation t1 llm *= t2 llm",
        "scoreboard players operation t3 llm = t0 llm",
        "scoreboard players operation t3 llm *= yb llm",
        "scoreboard players operation t3 llm /= 10000 llm",
        "scoreboard players operation t0 llm *= yb llm",
        "scoreboard players operation t0 llm += t1 llm",
        "scoreboard players operation t0 llm += t3 llm",
        "scoreboard players operation xb llm *= yb llm",
        "scoreboard players operation xb llm *= 10 llm",
        "scoreboard players operation t0 llm /= 1000 llm",
        "scoreboard players operation xb llm += t0 llm",
        "scoreboard players operation xs llm *= t6 llm",
        "scoreboard players operation xe llm += t5 llm",
        "execute if score xb llm matches 100000000.. run function llm:float_15",
        "scoreboard players set yb llm 35871171",
        "scoreboard players set ye llm -3",
        "scoreboard players set ys llm 1",
        "scoreboard players operation t0 llm = ys llm",
        "scoreboard players operation t0 llm *= xs llm",
        "scoreboard players operation t2 llm = yb llm",
        "function llm:float_0",
        "scoreboard players operation t3 llm = xe llm",
        "scoreboard players operation t3 llm -= ye llm",
        "execute if score t1 llm matches 1 run function llm:float_1",
        "scoreboard players operation t2 llm *= t0 llm",
        "execute unless score t3 llm matches 0 run function llm:float_2",
        "scoreboard players operation xb llm += t2 llm",
        "function llm:float_5",
        "execute if score xb llm matches 100000000.. run function llm:float_15",
        "scoreboard players operation yb llm = t4 llm",
        "scoreboard players operation t0 llm = xb llm",
        "scoreboard players operation t0 llm %= 10000 llm",
        "scoreboard players operation xb llm /= 10000 llm",
        "scoreboard players operation t1 llm = xb llm",
        "scoreboard players operation t2 llm = yb llm",
        "scoreboard players operation t2 llm %= 10000 llm",
        "scoreboard players operation yb llm /= 10000 llm",
        "scoreboard players operation t1 llm *= t2 llm",
        "scoreboard players operation t3 llm = t0 llm",
        "scoreboard players operation t3 llm *= yb llm",
        "scoreboard players operation t3 llm /= 10000 llm",
        "scoreboard players operation t0 llm *= yb llm",
        "scoreboard players operation t0 llm += t1 llm",
        "scoreboard players operation t0 llm += t3 llm",
        "scoreboard players operation xb llm *= yb llm",
        "scoreboard players operation xb llm *= 10 llm",
        "scoreboard players operation t0 llm /= 1000 llm",
        "scoreboard players operation xb llm += t0 llm",
        "scoreboard players operation xs llm *= t6 llm",
        "scoreboard players operation xe llm += t5 llm",
        "execute if score xb llm matches 100000000.. run function llm:float_15",
        "scoreboard players set yb llm 54159899",
        "scoreboard players set ye llm -2",
        "scoreboard players set ys llm 1",
        "scoreboard players operation t0 llm = ys llm",
        "scoreboard players operation t0 llm *= xs llm",
        "scoreboard players operation t2 llm = yb llm",
        "function llm:float_0",
        "scoreboard players operation t3 llm = xe llm",
        "scoreboard players operation t3 llm -= ye llm",
        "execute if score t1 llm matches 1 run function llm:float_1",
        "scoreboard players operation t2 llm *= t0 llm",
        "execute unless score t3 llm matches 0 run function llm:float_2",
        "scoreboard players operation xb llm += t2 llm",
        "function llm:float_5",
        "execute if score xb llm matches 100000000.. run function llm:float_15",
        "scoreboard players operation yb llm = t4 llm",
        "scoreboard players operation t0 llm = xb llm",
        "scoreboard players operation t0 llm %= 10000 llm",
        "scoreboard players operation xb llm /= 10000 llm",
        "scoreboard players operation t1 llm = xb llm",
        "scoreboard players operation t2 llm = yb llm",
        "scoreboard players operation t2 llm %= 10000 llm",
        "scoreboard players operation yb llm /= 10000 llm",
        "scoreboard players operation t1 llm *= t2 llm",
        "scoreboard players operation t3 llm = t0 llm",
        "scoreboard players operation t3 llm *= yb llm",
        "scoreboard players operation t3 llm /= 10000 llm",
        "scoreboard players operation t0 llm *= yb llm",
        "scoreboard players operation t0 llm += t1 llm",
        "scoreboard players operation t0 llm += t3 llm",
        "scoreboard players operation xb llm *= yb llm",
        "scoreboard players operation xb llm *= 10 llm",
        "scoreboard players operation t0 llm /= 1000 llm",
        "scoreboard players operation xb llm += t0 llm",
        "scoreboard players operation xs llm *= t6 llm",
        "scoreboard players operation xe llm += t5 llm",
        "execute if score xb llm matches 100000000.. run function llm:float_15",
        "scoreboard players set yb llm 14960666",
        "scoreboard players set ye llm -1",
        "scoreboard players set ys llm 1",
        "scoreboard players operation t0 llm = ys llm",
        "scoreboard players operation t0 llm *= xs llm",
        "scoreboard players operation t2 llm = yb llm",
        "function llm:float_0",
        "scoreboard players operation t3 llm = xe llm",
        "scoreboard players operation t3 llm -= ye llm",
        "execute if score t1 llm matches 1 run function llm:float_1",
        "scoreboard players operation t2 llm *= t0 llm",
        "execute unless score t3 llm matches 0 run function llm:float_2",
        "scoreboard players operation xb llm += t2 llm",
        "function llm:float_5",
        "execute if score xb llm matches 100000000.. run function llm:float_15",
        "scoreboard players operation yb llm = t4 llm",
        "scoreboard players operation t0 llm = xb llm",
        "scoreboard players operation t0 llm %= 10000 llm",
        "scoreboard players operation xb llm /= 10000 llm",
        "scoreboard players operation t1 llm = xb llm",
        "scoreboard players operation t2 llm = yb llm",
        "scoreboard players operation t2 llm %= 10000 llm",
        "scoreboard players operation yb llm /= 10000 llm",
        "scoreboard players operation t1 llm *= t2 llm",
        "scoreboard players operation t3 llm = t0 llm",
        "scoreboard players operation t3 llm *= yb llm",
        "scoreboard players operation t3 llm /= 10000 llm",
        "scoreboard players operation t0 llm *= yb llm",
        "scoreboard players operation t0 llm += t1 llm",
        "scoreboard players operation t0 llm += t3 llm",
        "scoreboard players operation xb llm *= yb llm",
        "scoreboard players operation xb llm *= 10 llm",
        "scoreboard players operation t0 llm /= 1000 llm",
        "scoreboard players operation xb llm += t0 llm",
        "scoreboard players operation xs llm *= t6 llm",
        "scoreboard players operation xe llm += t5 llm",
        "execute if score xb llm matches 100000000.. run function llm:float_15",
        "scoreboard players set yb llm 51295916",
        "scoreboard players set ye llm -1",
        "scoreboard players set ys llm 1",
        "scoreboard players operation t0 llm = ys llm",
        "scoreboard players operation t0 llm *= xs llm",
        "scoreboard players operation t2 llm = yb llm",
        "function llm:float_0",
        "scoreboard players operation t3 llm = xe llm",
        "scoreboard players operation t3 llm -= ye llm",
        "execute if score t1 llm matches 1 run function llm:float_1",
        "scoreboard players operation t2 llm *= t0 llm",
        "execute unless score t3 llm matches 0 run function llm:float_2",
        "scoreboard players operation xb llm += t2 llm",
        "function llm:float_5",
        "execute if score xb llm matches 100000000.. run function llm:float_15",
        "scoreboard players operation yb llm = t4 llm",
        "scoreboard players operation t0 llm = xb llm",
        "scoreboard players operation t0 llm %= 10000 llm",
        "scoreboard players operation xb llm /= 10000 llm",
        "scoreboard players operation t1 llm = xb llm",
        "scoreboard players operation t2 llm = yb llm",
        "scoreboard players operation t2 llm %= 10000 llm",
        "scoreboard players operation yb llm /= 10000 llm",
        "scoreboard players operation t1 llm *= t2 llm",
        "scoreboard players operation t3 llm = t0 llm",
        "scoreboard players operation t3 llm *= yb llm",
        "scoreboard players operation t3 llm /= 10000 llm",
        "scoreboard players operation t0 llm *= yb llm",
        "scoreboard players operation t0 llm += t1 llm",
        "scoreboard players operation t0 llm += t3 llm",
        "scoreboard players operation xb llm *= yb llm",
        "scoreboard players operation xb llm *= 10 llm",
        "scoreboard players operation t0 llm /= 1000 llm",
        "scoreboard players operation xb llm += t0 llm",
        "scoreboard players operation xs llm *= t6 llm",
        "scoreboard players operation xe llm += t5 llm",
        "execute if score xb llm matches 100000000.. run function llm:float_15",
        "scoreboard players set yb llm 99479042",
        "scoreboard players set ye llm -1",
        "scoreboard players set ys llm 1",
        "scoreboard players operation t0 llm = ys llm",
        "scoreboard players operation t0 llm *= xs llm",
        "scoreboard players operation t2 llm = yb llm",
        "function llm:float_0",
        "scoreboard players operation t3 llm = xe llm",
        "scoreboard players operation t3 llm -= ye llm",
        "execute if score t1 llm matches 1 run function llm:float_1",
        "scoreboard players operation t2 llm *= t0 llm",
        "execute unless score t3 llm matches 0 run function llm:float_2",
        "scoreboard players operation xb llm += t2 llm",
        "function llm:float_5",
        "execute if score xb llm matches 100000000.. run function llm:float_15",
        "scoreboard players operation yb llm = t4 llm",
        "scoreboard players operation t0 llm = xb llm",
        "scoreboard players operation t0 llm %= 10000 llm",
        "scoreboard players operation xb llm /= 10000 llm",
        "scoreboard players operation t1 llm = xb llm",
        "scoreboard players operation t2 llm = yb llm",
        "scoreboard players operation t2 llm %= 10000 llm",
        "scoreboard players operation yb llm /= 10000 llm",
        "scoreboard players operation t1 llm *= t2 llm",
        "scoreboard players operation t3 llm = t0 llm",
        "scoreboard players operation t3 llm *= yb llm",
        "scoreboard players operation t3 llm /= 10000 llm",
        "scoreboard players operation t0 llm *= yb llm",
        "scoreboard players operation t0 llm += t1 llm",
        "scoreboard players operation t0 llm += t3 llm",
        "scoreboard players operation xb llm *= yb llm",
        "scoreboard players operation xb llm *= 10 llm",
        "scoreboard players operation t0 llm /= 1000 llm",
        "scoreboard players operation xb llm += t0 llm",
        "scoreboard players operation xs llm *= t6 llm",
        "scoreboard players operation xe llm += t5 llm",
        "execute if score xb llm matches 100000000.. run function llm:float_15",
        "scoreboard players set yb llm 10008708",
        "scoreboard players set ye llm 0",
        "scoreboard players set ys llm 1",
        "scoreboard players operation t0 llm = ys llm",
        "scoreboard players operation t0 llm *= xs llm",
        "scoreboard players operation t2 llm = yb llm",
        "function llm:float_0",
        "scoreboard players operation t3 llm = xe llm",
        "scoreboard players operation t3 llm -= ye llm",
        "execute if score t1 llm matches 1 run function llm:float_1",
        "scoreboard players operation t2 llm *= t0 llm",
        "execute unless score t3 llm matches 0 run function llm:float_2",
        "scoreboard players operation xb llm += t2 llm",
        "function llm:float_5",
        "execute if score xb llm matches 100000000.. run function llm:float_15",
        "scoreboard players operation xe llm = r llm",
    ], "float_0": [
        "execute if score xe llm > ye llm run return run scoreboard players set t1 llm 0",
        "execute if score xe llm = ye llm if score xb llm >= t2 llm run return run scoreboard players set t1 llm 0",
        "scoreboard players set t1 llm 1",
    ], "float_1": [
        "scoreboard players operation xb llm >< t2 llm",
        "scoreboard players operation xe llm = ye llm",
        "scoreboard players operation t3 llm *= -1 llm",
        "scoreboard players operation xs llm = ys llm",
    ], "float_2": [
        "execute if score t3 llm matches 1..2 run return run function llm:float_3",
        "execute if score t3 llm matches 3..5 run return run function llm:float_4",
        "execute if score t3 llm matches 6 run return run scoreboard players operation t2 llm /= 1000000 llm",
        "execute if score t3 llm matches 7 run return run scoreboard players operation t2 llm /= 10000000 llm",
        "scoreboard players operation t2 llm /= 1000000000 llm",
    ], "float_3": [
        "execute if score t3 llm matches 1 run return run scoreboard players operation t2 llm /= 10 llm",
        "scoreboard players operation t2 llm /= 100 llm",
    ], "float_4": [
        "execute if score t3 llm matches 3 run return run scoreboard players operation t2 llm /= 1000 llm",
        "execute if score t3 llm matches 4 run return run scoreboard players operation t2 llm /= 10000 llm",
        "scoreboard players operation t2 llm /= 100000 llm",
    ], "float_5": [
        "execute if score xb llm matches 100000.. run return run function llm:float_6",
        "execute if score xb llm matches 100..99999 run return run function llm:float_7",
        "execute if score xb llm matches 1..9 run return run function llm:float_8",
        "execute if score xb llm matches 10..99 run return run function llm:float_9",
        "scoreboard players set xe llm 0",
        "scoreboard players set xs llm 0",
    ], "float_6": [
        "execute if score xb llm matches 100000..999999 run return run function llm:float_10",
        "execute if score xb llm matches 1000000..9999999 run return run function llm:float_11",
    ], "float_7": [
        "execute if score xb llm matches 100..999 run return run function llm:float_12",
        "execute if score xb llm matches 1000..9999 run return run function llm:float_13",
        "scoreboard players operation xb llm *= 10000 llm",
        "scoreboard players remove xe llm 4",
    ], "float_8": [
        "scoreboard players operation xb llm *= 100000000 llm",
        "scoreboard players remove xe llm 8",
    ], "float_9": [
        "scoreboard players operation xb llm *= 10000000 llm",
        "scoreboard players remove xe llm 7",
    ], "float_10": [
        "scoreboard players operation xb llm *= 1000 llm",
        "scoreboard players remove xe llm 3",
    ], "float_11": [
        "scoreboard players operation xb llm *= 10 llm",
        "scoreboard players remove xe llm 1",
    ], "float_12": [
        "scoreboard players operation xb llm *= 1000000 llm",
        "scoreboard players remove xe llm 6",
    ], "float_13": [
        "scoreboard players operation xb llm *= 100000 llm",
        "scoreboard players remove xe llm 5",
    ], "float_14": [
        "execute if score xe llm > ye llm run return run scoreboard players set r llm 1",
        "execute if score xe llm < ye llm run return run scoreboard players set r llm -1",
        "execute if score xb llm > yb llm run return run scoreboard players set r llm 1",
        "execute if score xb llm < yb llm run return run scoreboard players set r llm -1",
        "scoreboard players set r llm 0",
    ], "float_15": [
        "scoreboard players operation xb llm /= 10 llm",
        "scoreboard players add xe llm 1",
    ], "float_16": [
        "execute if score xb llm matches 1..329475 run return run scoreboard players set t0 llm 744",
        "scoreboard players operation t0 llm = xb llm",
        "execute if score xb llm matches 329476..18688328 run return run function llm:float_17",
        "execute if score xb llm matches 18688329..533609999 run return run function llm:float_18",
        "scoreboard players operation t0 llm /= 79249 llm",
        "scoreboard players add t0 llm 19750",
    ], "float_17": [
        "scoreboard players operation t0 llm /= 4750 llm",
        "scoreboard players add t0 llm 744",
    ], "float_18": [
        "scoreboard players operation t0 llm /= 24703 llm",
        "scoreboard players add t0 llm 4750",
    ], "float_19": [
        "scoreboard players remove t0 llm 1",
        "scoreboard players operation t1 llm = t0 llm",
        "scoreboard players operation t1 llm *= t0 llm",
    ], "float_20": [
        "$data modify entity @s Rotation[0] set value $(s)$(b)E$(e)f",
        "execute at @s positioned 0. 0. 0. rotated ~ 0. run tp @s ^ ^ ^1.",
        "execute store result score xb llm run data get entity @s Pos[0] -10000000",
        "kill @s",
    ], "float_21": [
        "scoreboard players set xs llm -1",
        "scoreboard players operation xb llm *= -1 llm",
    ], "float_22": [
        "$data modify entity @s Rotation[0] set value $(s)$(b)E$(e)f",
        "execute at @s positioned 0. 0. 0. rotated ~ 0. run tp @s ^ ^ ^1.",
        "execute store result score xb llm run data get entity @s Pos[2] 10000000",
        "kill @s",
    ], "float_23": [
        "scoreboard players operation xb llm /= 10 llm",
        "scoreboard players operation r llm = xb llm",
        "scoreboard players operation xb llm *= 10 llm",
    ], "float_24": [
        "scoreboard players operation xb llm /= 100 llm",
        "scoreboard players operation r llm = xb llm",
        "scoreboard players operation xb llm *= 100 llm",
    ], "float_25": [
        "scoreboard players operation xb llm /= 1000 llm",
        "scoreboard players operation r llm = xb llm",
        "scoreboard players operation xb llm *= 1000 llm",
    ], "float_26": [
        "scoreboard players operation xb llm /= 10000 llm",
        "scoreboard players operation r llm = xb llm",
        "scoreboard players operation xb llm *= 10000 llm",
    ], "float_27": [
        "scoreboard players operation xb llm /= 100000 llm",
        "scoreboard players operation r llm = xb llm",
        "scoreboard players operation xb llm *= 100000 llm",
    ], "float_28": [
        "scoreboard players operation xb llm /= 1000000 llm",
        "scoreboard players operation r llm = xb llm",
        "scoreboard players operation xb llm *= 1000000 llm",
    ], "float_29": [
        "scoreboard players operation xb llm /= 10000000 llm",
        "scoreboard players operation r llm = xb llm",
        "scoreboard players operation xb llm *= 10000000 llm",
    ], "float_30": [
        "scoreboard players add r llm 1",
        "scoreboard players operation r llm *= -1 llm",
        "function llm:float_31",
        "execute if score xb llm matches 100000000.. run function llm:float_15",
        "execute if score xs llm matches 0 run scoreboard players set xs llm -1",
    ], "float_31": [
        "execute if score xe llm matches 7.. run return run scoreboard players add xb llm 1",
        "execute if score xe llm matches 6 run return run scoreboard players add xb llm 10",
        "execute if score xe llm matches 5 run return run scoreboard players add xb llm 100",
        "execute if score xe llm matches 4 run return run scoreboard players add xb llm 1000",
        "execute if score xe llm matches 3 run return run scoreboard players add xb llm 10000",
        "execute if score xe llm matches 2 run return run scoreboard players add xb llm 100000",
        "execute if score xe llm matches 1 run return run scoreboard players add xb llm 1000000",
        "execute if score xe llm matches 0 run return run scoreboard players add xb llm 10000000",
    ], "float_32" : [
        "$data modify storage llm args.x set value $(s)$(b)E$(e)f",
    ], "floor": [
        "execute if score xe llm matches 7.. run return run scoreboard players operation r llm = xb llm",
        "execute if score xe llm matches 6 run return run function llm:float_23",
        "execute if score xe llm matches 5 run return run function llm:float_24",
        "execute if score xe llm matches 4 run return run function llm:float_25",
        "execute if score xe llm matches 3 run return run function llm:float_26",
        "execute if score xe llm matches 2 run return run function llm:float_27",
        "execute if score xe llm matches 1 run return run function llm:float_28",
        "execute if score xe llm matches 0 run return run function llm:float_29",
        "scoreboard players set xb llm 0",
        "scoreboard players set xe llm 0",
        "scoreboard players set xs llm 0",
        "scoreboard players set r llm 0",
    ], "inv": [
        "scoreboard players operation t0 llm = xb llm",
        "scoreboard players operation t1 llm = t0 llm",
        "scoreboard players operation t1 llm %= 10000 llm",
        "scoreboard players set xb llm 100000000",
        "scoreboard players operation t2 llm = t0 llm",
        "scoreboard players operation t0 llm /= 25 llm",
        "scoreboard players operation t2 llm %= 25 llm",
        "scoreboard players operation t3 llm = xb llm",
        "scoreboard players operation xb llm /= t0 llm",
        "scoreboard players operation xb llm *= 40 llm",
        "scoreboard players operation t3 llm %= t0 llm",
        "scoreboard players operation t3 llm *= 40 llm",
        "scoreboard players operation t1 llm = t3 llm",
        "scoreboard players operation t3 llm /= t0 llm",
        "scoreboard players operation xb llm += t3 llm",
        "scoreboard players operation t1 llm %= t0 llm",
        "scoreboard players operation t1 llm *= 25 llm",
        "scoreboard players operation t4 llm = t2 llm",
        "scoreboard players operation t4 llm *= xb llm",
        "scoreboard players operation t1 llm -= t4 llm",
        "scoreboard players operation t1 llm *= 10 llm",
        "scoreboard players operation t3 llm = t1 llm",
        "scoreboard players operation t1 llm /= t0 llm",
        "scoreboard players operation t1 llm *= 40 llm",
        "scoreboard players operation t3 llm %= t0 llm",
        "scoreboard players operation t3 llm *= 40 llm",
        "scoreboard players operation t3 llm /= t0 llm",
        "scoreboard players operation t1 llm += t3 llm",
        "scoreboard players operation xe llm *= -1 llm",
        "scoreboard players remove xe llm 1",
        "execute if score xb llm matches 100000.. run return 0",
        "scoreboard players operation xb llm *= 10000 llm",
        "scoreboard players operation xb llm += t1 llm",
        "execute if score xb llm matches 100000000.. run function llm:float_15",
    ], "mul": [
        "scoreboard players operation t0 llm = xb llm",
        "scoreboard players operation t0 llm %= 10000 llm",
        "scoreboard players operation xb llm /= 10000 llm",
        "scoreboard players operation t1 llm = xb llm",
        "scoreboard players operation t2 llm = yb llm",
        "scoreboard players operation t2 llm %= 10000 llm",
        "scoreboard players operation yb llm /= 10000 llm",
        "scoreboard players operation t1 llm *= t2 llm",
        "scoreboard players operation t3 llm = t0 llm",
        "scoreboard players operation t3 llm *= yb llm",
        "scoreboard players operation t3 llm /= 10000 llm",
        "scoreboard players operation t0 llm *= yb llm",
        "scoreboard players operation t0 llm += t1 llm",
        "scoreboard players operation t0 llm += t3 llm",
        "scoreboard players operation xb llm *= yb llm",
        "scoreboard players operation xb llm *= 10 llm",
        "scoreboard players operation t0 llm /= 1000 llm",
        "scoreboard players operation xb llm += t0 llm",
        "scoreboard players operation xs llm *= ys llm",
        "scoreboard players operation xe llm += ye llm",
        "execute if score xb llm matches 0 store result score xs llm run scoreboard players set xe llm 0",
        "execute if score xb llm matches 100000000.. run function llm:float_15",
    ], "sin": [
        "data modify storage llm args.s set value \"\"",
        "execute if score xs llm matches -1 run data modify storage llm args.s set value \"-\"",
        "execute store result storage llm args.e int 1 run scoreboard players remove xe llm 7",
        "execute store result storage llm args.b int 1 run scoreboard players get xb llm",
        "execute summon marker run function llm:float_20 with storage llm args",
        "scoreboard players set xe llm 0",
        "scoreboard players set xs llm 1",
        "execute if score xb llm matches ..-1 run function llm:float_21",
        "function llm:float_5",
        "execute if score xb llm matches 100000000.. run function llm:float_15",
    ], "sq": [
        "scoreboard players operation t0 llm = xb llm",
        "scoreboard players operation xb llm /= 10000 llm",
        "scoreboard players operation t0 llm %= 10000 llm",
        "scoreboard players operation t1 llm = t0 llm",
        "scoreboard players operation t1 llm *= t0 llm",
        "scoreboard players operation t1 llm /= 10000 llm",
        "scoreboard players operation t0 llm *= xb llm",
        "scoreboard players operation t0 llm *= 2 llm",
        "scoreboard players operation t0 llm += t1 llm",
        "scoreboard players operation xb llm *= xb llm",
        "scoreboard players operation xb llm *= 10 llm",
        "scoreboard players operation t0 llm /= 1000 llm",
        "scoreboard players operation xb llm += t0 llm",
        "scoreboard players operation xs llm *= xs llm",
        "scoreboard players operation xe llm *= 2 llm",
        "execute if score xb llm matches 100000000.. run function llm:float_15",
    ], "sqrt": [
        "execute if score xb llm matches 0 run return run scoreboard players operation xe llm /= 2 llm",
        "scoreboard players operation t0 llm = xe llm",
        "scoreboard players operation t0 llm %= 2 llm",
        "execute if score t0 llm matches 0 run function llm:float_11",
        "scoreboard players operation xe llm /= 2 llm",
        "function llm:float_16",
        "scoreboard players operation t1 llm = xb llm",
        "scoreboard players operation t1 llm /= t0 llm",
        "scoreboard players operation t0 llm += t1 llm",
        "scoreboard players operation t0 llm /= 2 llm",
        "scoreboard players operation t1 llm = xb llm",
        "scoreboard players operation t1 llm /= t0 llm",
        "scoreboard players operation t0 llm += t1 llm",
        "scoreboard players operation t0 llm /= 2 llm",
        "scoreboard players operation t1 llm = t0 llm",
        "scoreboard players operation t1 llm *= t0 llm",
        "execute if score t1 llm > xb llm run function llm:float_19",
        "scoreboard players operation xb llm -= t1 llm",
        "scoreboard players operation xb llm *= 500 llm",
        "scoreboard players operation xb llm /= t0 llm",
        "scoreboard players operation t0 llm *= 1000 llm",
        "scoreboard players operation xb llm += t0 llm",
        "scoreboard players operation xb llm *= 10 llm",
        "execute if score xb llm matches 100000000.. run function llm:float_15",
    ], "float_to_nbt": [
        "data modify storage llm args.s set value \"\"",
        "execute if score xs llm matches -1 run data modify storage llm args.s set value \"-\"",
        "scoreboard players operation t0 llm = xe llm",
        "execute store result storage llm args.e int 1 run scoreboard players remove t0 llm 7",
        "execute store result storage llm args.b int 1 run scoreboard players get xb llm",
        "function llm:float_32 with storage llm args",
    ]
}

for name, cmds in float_funcs.items():
    with FunctionWritter(name) as f:
        for cmd in cmds:
            f.write(cmd)


# Parameters setup
# ----------------

# Helper function that encodes a float number into (base, exponent, sign)
# n = s*b*10^(e-7)
def encode_float(n):
    e_notation = format(n, '.7e')
    b, e = e_notation.split('e')
    b = int(b.replace('.', ''))
    e = int(e)
    s = 1 if b > 0 else -1 if b < 0 else 0
    if s == 0:
        b = e = 0
    else:
        b = abs(b)
    return b, e, s

# Get the commands that setup a list of parameters
# Automatically split the command if it reaches command character limit (2M)
def set_params_cmds(params, path: str, start: int = 0) -> list[str]:
    starter = f'data modify storage llm {path} set value {{'
    starter_len = len(starter)
    cmds = []
    current_chunk = []
    current_len = starter_len

    for i, value in enumerate(params, start=start):
        b, e, s = encode_float(value)

        for suf, elt in zip('bes', [b, e, s]):
            v = f"{i}{suf}:{str(elt)}"
            v_len = len(v)
            if not current_chunk:
                new_len = current_len + v_len
            else:
                new_len = current_len + 1 + v_len
            
            if new_len >= 2_000_000:
                cmd = starter + ','.join(current_chunk) + '}'
                cmds.append(cmd)
                if len(cmds) == 1:
                    starter = f'data modify storage llm {path} merge value {{'
                    starter_len = len(starter)

                current_chunk = [v]
                current_len = starter_len + v_len
            else:
                current_chunk.append(v)
                current_len = new_len
    if current_chunk:
        cmd = starter + ','.join(current_chunk) + '}'
        cmds.append(cmd)
    return cmds

with FunctionWritter('params') as f:
    cmds = []
    cmds.extend(set_params_cmds(token_embedding_table, 'params.emb'))
    cmds.extend(set_params_cmds(rms_att_weight, 'params.att_w'))
    cmds.extend(set_params_cmds(rms_ffn_weight, 'params.ffn_w'))

    for l in range(n_layers):
        cmds.extend(set_params_cmds(wq[l * dim * dim : (l + 1) * dim * dim], f'params.wq_{l}'))
        cmds.extend(set_params_cmds(wk[l * kv_dim * dim : (l + 1) * kv_dim * dim], f'params.wk_{l}'))
        cmds.extend(set_params_cmds(wv[l * kv_dim * dim : (l + 1) * kv_dim * dim], f'params.wv_{l}'))
        cmds.extend(set_params_cmds(wo[l * dim * dim : (l + 1) * dim * dim], f'params.wo_{l}'))
        cmds.extend(set_params_cmds(w1[l * hidden_dim * dim : (l + 1) * hidden_dim * dim], f'params.w1_{l}'))
        cmds.extend(set_params_cmds(w2[l * hidden_dim * dim : (l + 1) * hidden_dim * dim], f'params.w2_{l}'))
        cmds.extend(set_params_cmds(w3[l * hidden_dim * dim : (l + 1) * hidden_dim * dim], f'params.w3_{l}'))
    
    f.write(f'bossbar set progress max {len(cmds)}')
    f.write(f'bossbar set progress value 0')
    f.write(f'bossbar set progress name {{"text":"Initializing parameters","color":"yellow","bold":true}}')
    f.write(f'bossbar set progress players @a')

    for i, cmd in enumerate(cmds, 1):
        f.write(cmd)
        f.write(f'bossbar set progress value {i}')

    f.write(f'bossbar set progress players')

# Transformer implementation
# --------------------------

# Functions that are commonly used

with FunctionWritter('rmsnorm_0') as f:
    # ss = 0
    f.write(f'scoreboard players set ssb llm 0')
    f.write(f'scoreboard players set sse llm 0')
    f.write(f'scoreboard players set sss llm 0')
    for i in range(dim):
        # ss += x[i] ** 2
        f.write(f'scoreboard players operation xb llm = {i}b x')
        f.write(f'scoreboard players operation xe llm = {i}e x')
        f.write(f'scoreboard players operation xs llm = {i}s x')
        f.write(f'function llm:sq')
        f.write(f'scoreboard players operation yb llm = ssb llm')
        f.write(f'scoreboard players operation ye llm = sse llm')
        f.write(f'scoreboard players operation ys llm = sss llm')
        f.write(f'function llm:add')
        f.write(f'scoreboard players operation ssb llm = xb llm')
        f.write(f'scoreboard players operation sse llm = xe llm')
        f.write(f'scoreboard players operation sss llm = xs llm')

    # ss = 1 / (sqrt(ss / dim + 0.00001))
    f.write(f'scoreboard players set xb llm {dim}')
    f.write(f'scoreboard players set xe llm 7')
    f.write(f'scoreboard players set xs llm 1')
    f.write(f'function llm:float_5')
    f.write(f'execute if score xb llm matches 100000000.. run function llm:float_15')
    f.write(f'scoreboard players operation yb llm = xb llm')
    f.write(f'scoreboard players operation ye llm = xe llm')
    f.write(f'scoreboard players operation ys llm = xs llm')
    f.write(f'scoreboard players operation xb llm = ssb llm')
    f.write(f'scoreboard players operation xe llm = sse llm')
    f.write(f'scoreboard players operation xs llm = sss llm')
    f.write(f'function llm:div')
    f.write(f'scoreboard players add xb llm 1')
    f.write(f'execute if score xb llm matches 100000000.. run function llm:float_15')
    f.write(f'function llm:sqrt')
    f.write(f'function llm:inv')
    f.write(f'scoreboard players operation ssb llm = xb llm')
    f.write(f'scoreboard players operation sse llm = xe llm')
    f.write(f'scoreboard players operation sss llm = xs llm')
    
with FunctionWritter(f'rmsnorm_1') as f:
    for i in range(dim):
        # tx[i] *= ss * x[i]
        f.write(f'scoreboard players operation xb llm = {i}b tx')
        f.write(f'scoreboard players operation xe llm = {i}e tx')
        f.write(f'scoreboard players operation xs llm = {i}s tx')
        f.write(f'scoreboard players operation yb llm = ssb llm')
        f.write(f'scoreboard players operation ye llm = sse llm')
        f.write(f'scoreboard players operation ys llm = sss llm')
        f.write(f'function llm:mul')
        f.write(f'scoreboard players operation yb llm = {i}b x')
        f.write(f'scoreboard players operation ye llm = {i}e x')
        f.write(f'scoreboard players operation ys llm = {i}s x')
        f.write(f'function llm:mul')
        f.write(f'scoreboard players operation {i}b tx = xb llm')
        f.write(f'scoreboard players operation {i}e tx = xe llm')
        f.write(f'scoreboard players operation {i}s tx = xs llm')

# Matrix multiplication

with FunctionWritter(f'matmul') as f:
    # i = 0;
    f.write(f'scoreboard players set i llm 0')
    f.write(f'data modify storage llm args.i set value 0')

    # idx = 0;
    f.write(f'scoreboard players set idx llm 0')
    f.write(f'data modify storage llm args.idx set value 0')
    
    # do {
    f.write(f'function llm:matmul_i with storage llm args')
    # } while (1);

with FunctionWritter(f'matmul_i') as f:
    # out[i] = 0;
    f.write(f'$scoreboard players set $(i)b $(out) 0')
    f.write(f'$scoreboard players set $(i)e $(out) 0')
    f.write(f'$scoreboard players set $(i)s $(out) 0')

    # j = 0;
    f.write(f'scoreboard players set j llm 0')
    f.write(f'data modify storage llm args.j set value 0')

    # do {
    f.write(f'function llm:matmul_j with storage llm args')
    # } while (1);

    # i++;
    f.write(f'scoreboard players add i llm 1')

    # if (i == s1) break;
    f.write(f'execute if score i llm = s1 llm run return 1')

    f.write(f'execute store result storage llm args.i int 1 run scoreboard players get i llm')
    f.write(f'function llm:matmul_i with storage llm args')

with FunctionWritter(f'matmul_j') as f:
    # x = M[idx];
    f.write(f'$execute store result score xb llm run data get storage llm params.$(m).$(idx)b')
    f.write(f'$execute store result score xe llm run data get storage llm params.$(m).$(idx)e')
    f.write(f'$execute store result score xs llm run data get storage llm params.$(m).$(idx)s')

    # y = I[j];
    f.write(f'$scoreboard players operation yb llm = $(j)b $(in)')
    f.write(f'$scoreboard players operation ye llm = $(j)e $(in)')
    f.write(f'$scoreboard players operation ys llm = $(j)s $(in)')

    # x *= y;
    f.write(f'function llm:mul')

    # y = O[i];
    f.write(f'$scoreboard players operation yb llm = $(i)b $(out)')
    f.write(f'$scoreboard players operation ye llm = $(i)e $(out)')
    f.write(f'$scoreboard players operation ys llm = $(i)s $(out)')

    # x += y;
    f.write(f'function llm:add')

    # O[i] = x;
    f.write(f'$scoreboard players operation $(i)b $(out) = xb llm')
    f.write(f'$scoreboard players operation $(i)e $(out) = xe llm')
    f.write(f'$scoreboard players operation $(i)s $(out) = xs llm')
    
    # j++;
    f.write(f'scoreboard players add j llm 1')

    # idx++;
    f.write(f'execute store result storage llm args.idx int 1 run scoreboard players add idx llm 1')

    # if (j == s2) break;
    f.write(f'execute if score j llm = s2 llm run return 1')

    f.write(f'execute store result storage llm args.j int 1 run scoreboard players get j llm')
    f.write(f'function llm:matmul_j with storage llm args')

# Special matmul function that updates the progress bar when calculating logits = x @ emb,
# update the progress bar as computing. It's important since this process takes up most of the time
with FunctionWritter(f'matmul_logits') as f:
    # i = 0;
    f.write(f'scoreboard players set i llm 0')
    f.write(f'scoreboard players set n llm 0')
    f.write(f'data modify storage llm args.i set value 0')

    # idx = 0;
    f.write(f'scoreboard players set idx llm 0')
    f.write(f'data modify storage llm args.idx set value 0')
    
    # do {
    f.write(f'function llm:matmul_logits_i with storage llm args')
    # } while (1);

with FunctionWritter(f'matmul_logits_i') as f:
    # logits[i] = 0;
    f.write(f'$scoreboard players set $(i)b logits 0')
    f.write(f'$scoreboard players set $(i)e logits 0')
    f.write(f'$scoreboard players set $(i)s logits 0')

    # j = 0;
    f.write(f'scoreboard players set j llm 0')
    f.write(f'data modify storage llm args.j set value 0')
    
    # do {...} while (1);
    f.write(f'function llm:matmul_logits_j with storage llm args')
    # i++;
    f.write(f'scoreboard players add i llm 1')
    # if (i == s1) break;
    f.write(f'execute if score i llm matches {vocab_size} run return 1')

    # Update the progress bar every dim^2 steps
    f.write(f'scoreboard players add n llm 1')
    f.write(f'execute if score n llm matches {dim} run return run function llm:matmul_logits_update')

    # Otherwise loop back as normal
    f.write(f'execute store result storage llm args.i int 1 run scoreboard players get i llm')
    f.write(f'function llm:matmul_logits_i with storage llm args')

with FunctionWritter(f'matmul_logits_j') as f:
    # Get Emb[i, j]
    f.write(f'$execute store result score xb llm run data get storage llm params.emb.$(idx)b')
    f.write(f'$execute store result score xe llm run data get storage llm params.emb.$(idx)e')
    f.write(f'$execute store result score xs llm run data get storage llm params.emb.$(idx)s')

    # Get X[j]
    f.write(f'$scoreboard players operation yb llm = $(j)b x')
    f.write(f'$scoreboard players operation ye llm = $(j)e x')
    f.write(f'$scoreboard players operation ys llm = $(j)s x')

    # x = Emb[i, j] * X[j]
    f.write(f'function llm:mul')
    
    # Get Logits[i]
    f.write(f'$scoreboard players operation yb llm = $(i)b logits')
    f.write(f'$scoreboard players operation ye llm = $(i)e logits')
    f.write(f'$scoreboard players operation ys llm = $(i)s logits')

    # x += Logits[i]
    f.write(f'function llm:add')

    # Logits[i] = x
    f.write(f'$scoreboard players operation $(i)b logits = xb llm')
    f.write(f'$scoreboard players operation $(i)e logits = xe llm')
    f.write(f'$scoreboard players operation $(i)s logits = xs llm')

    # j += 1, idx += 1; loop back
    f.write(f'scoreboard players add j llm 1')
    f.write(f'execute store result storage llm args.idx int 1 run scoreboard players add idx llm 1')
    f.write(f'execute if score j llm matches {dim} run return 1')
    f.write(f'execute store result storage llm args.j int 1 run scoreboard players get j llm')
    f.write(f'function llm:matmul_logits_j with storage llm args')

with FunctionWritter(f'matmul_logits_schedule') as f:
    f.write(f'execute store result storage llm args.i int 1 run scoreboard players get i llm')
    f.write(f'function llm:matmul_logits_i with storage llm args')

with FunctionWritter(f'matmul_logits_update') as f:
    # Update the value in the progress bar; n = 0
    f.write(f'execute store result score progress llm run bossbar get progress value')
    f.write(f'execute store result bossbar progress value run scoreboard players add progress llm 1')
    f.write(f'scoreboard players set n llm 0')

    # Minecraft only updates the bossbar at the end of each tick unless the execution is directly
    # launched by the player. Although the inference process is initially triggered by the player,
    # the current execution environment is actually called by previous `/schedule` commands in llm:forward.
    # So we have to wait another tick for the progress bar to update visually.
    f.write(f'schedule function llm:matmul_logits_schedule 1t')

# Copy a row of token embedding table into x
with FunctionWritter('copy_embedding_row') as f:
    f.write(f'$execute store result score $(i)b x run data get storage llm params.emb.$(j)b')
    f.write(f'$execute store result score $(i)e x run data get storage llm params.emb.$(j)e')
    f.write(f'$execute store result score $(i)s x run data get storage llm params.emb.$(j)s')

    f.write(f'execute if score i llm matches {dim} run return 0')
    f.write(f'execute store result storage llm args.i int 1 run scoreboard players add i llm 1')
    f.write(f'execute store result storage llm args.j int 1 run scoreboard players add j llm 1')
    f.write(f'function llm:copy_embedding_row with storage llm args')

# SiLU activation function (silu(x) = x * sigmoid(x)) & Elementwise multiply with w3(x)
with FunctionWritter(f'silu') as f:
    for i in range(hidden_dim):
        # th[i] *= th2[i] / (1 + exp(-th[i]))
        
        # Get th[i]
        f.write(f'scoreboard players operation xb llm = {i}b th')
        f.write(f'scoreboard players operation xe llm = {i}e th')
        f.write(f'scoreboard players operation xs llm = {i}s th')

        # Compute x = exp(-th[i])
        f.write(f'scoreboard players operation xs llm *= -1 llm')

        f.write(f'function llm:exp')

        # x += 1.0
        f.write(f'scoreboard players set yb llm 10000000')
        f.write(f'scoreboard players set ye llm 0')
        f.write(f'scoreboard players set ys llm 1')
        f.write(f'function llm:add')

        # x = th2[i] / x
        f.write(f'scoreboard players operation yb llm = xb llm')
        f.write(f'scoreboard players operation ye llm = xe llm')
        f.write(f'scoreboard players operation ys llm = xs llm')
        f.write(f'scoreboard players operation xb llm = {i}b th2')
        f.write(f'scoreboard players operation xe llm = {i}e th2')
        f.write(f'scoreboard players operation xs llm = {i}s th2')
        f.write(f'function llm:div')

        # x *= th[i]
        f.write(f'scoreboard players operation yb llm = {i}b th')
        f.write(f'scoreboard players operation ye llm = {i}e th')
        f.write(f'scoreboard players operation ys llm = {i}s th')
        f.write(f'function llm:mul')

        # th[i] = x
        f.write(f'scoreboard players operation {i}b th = xb llm')
        f.write(f'scoreboard players operation {i}e th = xe llm')
        f.write(f'scoreboard players operation {i}s th = xs llm')

with FunctionWritter(f'crop_context') as f:
    # Keep a fixed context window
    # window_len = min(seq_len, pos + 1)
    # start = pos - window_len + 1
    f.write(f'scoreboard players operation window_len llm = pos llm')
    f.write(f'scoreboard players add window_len llm 1')
    f.write(f'execute if score pos llm matches {seq_len + 1}.. run scoreboard players set window_len llm {seq_len}')
    f.write(f'scoreboard players operation start llm = pos llm')
    f.write(f'scoreboard players operation start llm -= window_len llm')
    f.write(f'scoreboard players add start llm 1')

with FunctionWritter(f'rope') as f:
    # Apply RoPE (rotary position embedding) to Q and K
    # Note: Q has dim dimensions, K has kv_dim dimensions
    def rotate_v(i, v):
        # v0 = v[i]
        # v1 = v[i + 1]

        # v[i] = v0 * fcr - v1 * fci
        # v[i + 1] = v0 * fci + v1 * fcr

        f.write(f'scoreboard players operation t0b llm = {i}b {v}')
        f.write(f'scoreboard players operation t0e llm = {i}e {v}')
        f.write(f'scoreboard players operation t0s llm = {i}s {v}')

        f.write(f'scoreboard players operation xb llm = {i}b {v}')
        f.write(f'scoreboard players operation xe llm = {i}e {v}')
        f.write(f'scoreboard players operation xs llm = {i}s {v}')
        
        f.write(f'scoreboard players operation yb llm = fcrb llm')
        f.write(f'scoreboard players operation ye llm = fcre llm')
        f.write(f'scoreboard players operation ys llm = fcrs llm')

        f.write(f'function llm:mul')

        f.write(f'scoreboard players operation t1b llm = xb llm')
        f.write(f'scoreboard players operation t1e llm = xe llm')
        f.write(f'scoreboard players operation t1s llm = xs llm')

        f.write(f'scoreboard players operation xb llm = {i + 1}b {v}')
        f.write(f'scoreboard players operation xe llm = {i + 1}e {v}')
        f.write(f'scoreboard players operation xs llm = {i + 1}s {v}')
        
        f.write(f'scoreboard players operation yb llm = fcib llm')
        f.write(f'scoreboard players operation ye llm = fcie llm')
        f.write(f'scoreboard players operation ys llm = fcis llm')

        f.write(f'function llm:mul')

        f.write(f'scoreboard players operation xs llm *= -1 llm')

        f.write(f'scoreboard players operation yb llm = t1b llm')
        f.write(f'scoreboard players operation ye llm = t1e llm')
        f.write(f'scoreboard players operation ys llm = t1s llm')

        f.write(f'function llm:add')

        f.write(f'scoreboard players operation {i}b {v} = xb llm')
        f.write(f'scoreboard players operation {i}e {v} = xe llm')
        f.write(f'scoreboard players operation {i}s {v} = xs llm')

        f.write(f'scoreboard players operation xb llm = {i + 1}b {v}')
        f.write(f'scoreboard players operation xe llm = {i + 1}e {v}')
        f.write(f'scoreboard players operation xs llm = {i + 1}s {v}')

        f.write(f'scoreboard players operation yb llm = fcrb llm')
        f.write(f'scoreboard players operation ye llm = fcre llm')
        f.write(f'scoreboard players operation ys llm = fcrs llm')

        f.write(f'function llm:mul')

        f.write(f'scoreboard players operation t1b {v} = xb llm')
        f.write(f'scoreboard players operation t1e {v} = xe llm')
        f.write(f'scoreboard players operation t1s {v} = xs llm')

        f.write(f'scoreboard players operation xb llm = t0b llm')
        f.write(f'scoreboard players operation xe llm = t0e llm')
        f.write(f'scoreboard players operation xs llm = t0s llm')

        f.write(f'scoreboard players operation yb llm = fcib llm')
        f.write(f'scoreboard players operation ye llm = fcie llm')
        f.write(f'scoreboard players operation ys llm = fcis llm')

        f.write(f'function llm:mul')

        f.write(f'scoreboard players operation yb llm = t1b {v}')
        f.write(f'scoreboard players operation ye llm = t1e {v}')
        f.write(f'scoreboard players operation ys llm = t1s {v}')

        f.write(f'function llm:add')

        f.write(f'scoreboard players operation {i + 1}b {v} = xb llm')
        f.write(f'scoreboard players operation {i + 1}e {v} = xe llm')
        f.write(f'scoreboard players operation {i + 1}s {v} = xs llm')

    for i in range(0, dim, 2):
        head_dim = i % head_size
        freq = 1 / 10000.0 ** (head_dim / head_size)
        freq_b, freq_e, freq_s = encode_float(freq)

        # v = pos * freq
        f.write(f'scoreboard players set xb llm {freq_b}')
        f.write(f'scoreboard players set xe llm {freq_e}')
        f.write(f'scoreboard players set xs llm {freq_s}')

        f.write(f'scoreboard players operation yb llm = posb llm')
        f.write(f'scoreboard players operation ye llm = pose llm')
        f.write(f'scoreboard players operation ys llm = poss llm')

        f.write(f'function llm:mul')

        # v = degrees(v)
        # Took me forever to trace this bug

        f.write(f'scoreboard players set yb llm 17453293')
        f.write(f'scoreboard players set ye llm -2')
        f.write(f'scoreboard players set ys llm 1')

        f.write(f'function llm:div')
    
        f.write(f'scoreboard players operation vb llm = xb llm')
        f.write(f'scoreboard players operation ve llm = xe llm')
        f.write(f'scoreboard players operation vs llm = xs llm')

        # fci = sin(v)
        f.write(f'function llm:sin')

        f.write(f'scoreboard players operation fcib llm = xb llm')
        f.write(f'scoreboard players operation fcie llm = xe llm')
        f.write(f'scoreboard players operation fcis llm = xs llm')

        f.write(f'scoreboard players operation xb llm = vb llm')
        f.write(f'scoreboard players operation xe llm = ve llm')
        f.write(f'scoreboard players operation xs llm = vs llm')

        # fcr = cos(v)
        f.write(f'function llm:cos')

        f.write(f'scoreboard players operation fcrb llm = xb llm')
        f.write(f'scoreboard players operation fcre llm = xe llm')
        f.write(f'scoreboard players operation fcrs llm = xs llm')

        # Rotate q[i : i + 2]
        rotate_v(i, 'q')

        # Only rotate k[i : i + 2] if i < kv_dim (K has fewer dimensions than Q)
        if i < kv_dim:
            rotate_v(i, 'k')


# Forward propagation of the transformer
# It's pure torture to debug this

with FunctionWritter('forward') as f:
    # Full forward pass including final norm and classifier

    # Setup the progress bar
    # There are 8 updates in each layer, and `vocab_size // dim` updates in the final matrix multiplication
    f.write(f'bossbar set progress max {n_layers * 8 + vocab_size // dim}')
    f.write(f'bossbar set progress value 0')
    f.write(f'bossbar set progress name [{{"text":"Inferring, Step #","color":"green","bold":true}},{{"score":{{"name":"pos","objective":"llm"}}}}]')
    f.write(f'bossbar set progress players @a')

    f.write(f'function llm:forward_hidden')
    f.split(True, 8 * n_layers + 1)

    # Final rmsnorm
    f.write('function llm:rmsnorm_0')

    for i in range(dim):
        # x[i] *= rms_final_weight[i] * ss
        rms_ffn_i = rms_final_weight[i]
        pb, pe, ps = encode_float(rms_ffn_i)

        f.write(f'scoreboard players operation xb llm = {i}b x')
        f.write(f'scoreboard players operation xe llm = {i}e x')
        f.write(f'scoreboard players operation xs llm = {i}s x')

        f.write(f'scoreboard players set yb llm {pb}')
        f.write(f'scoreboard players set ye llm {pe}')
        f.write(f'scoreboard players set ys llm {ps}')

        f.write(f'function llm:mul')

        f.write(f'scoreboard players operation yb llm = ssb llm')
        f.write(f'scoreboard players operation ye llm = sse llm')
        f.write(f'scoreboard players operation ys llm = sss llm')

        f.write(f'function llm:mul')

        f.write(f'scoreboard players operation {i}b x = xb llm')
        f.write(f'scoreboard players operation {i}e x = xe llm')
        f.write(f'scoreboard players operation {i}s x = xs llm')

    # Classifier into logits
    f.write(f'function llm:matmul_logits')


with FunctionWritter('forward_hidden') as f:
    # Process a single token at a given position (prompt processing)

    # Convert pos (int) to float
    f.write(f'scoreboard players operation xb llm = pos llm')
    f.write(f'scoreboard players set xe llm 7')
    f.write(f'scoreboard players set xs llm 1')
    f.write(f'function llm:float_5')
    f.write(f'execute if score xb llm matches 100000000.. run function llm:float_15')
    f.write(f'scoreboard players operation posb llm = xb llm')
    f.write(f'scoreboard players operation pose llm = xe llm')
    f.write(f'scoreboard players operation poss llm = xs llm')

    # Copy the token embedding into x
    f.write(f'scoreboard players operation start llm = tok llm')
    f.write(f'scoreboard players operation start llm *= {dim} llm')

    f.write(f'scoreboard players set i llm 0')
    f.write(f'data modify storage llm args.i set value 0')
    f.write(f'scoreboard players operation j llm = start llm')
    f.write(f'execute store result storage llm args.j int 1 run scoreboard players get j llm')

    f.write(f'function llm:copy_embedding_row with storage llm args')

    # Forward all the layers
    for l in range(n_layers):
        # Attention rmsnorm:
        # tx = rmsnorm(x, rms_att_weight[l * dim : (l + 1) * dim])
        f.write(f'function llm:rmsnorm_0')
        
        for j in range(dim):
            # tx[j] = rms_att_weight[l * dim + j]
            idx = l * dim + j
            f.write(f'execute store result score {j}b tx run data get storage llm params.att_w.{idx}b')
            f.write(f'execute store result score {j}e tx run data get storage llm params.att_w.{idx}e')
            f.write(f'execute store result score {j}s tx run data get storage llm params.att_w.{idx}s')

        f.write(f'function llm:rmsnorm_1')
        
        # QKV matrix multiplications for this position
        f.write(f'data modify storage llm args.in set value "tx"')
        f.write(f'data modify storage llm args.out set value "q"')
        f.write(f'data modify storage llm args.m set value "wq_{l}"')
        f.write(f'execute store result score s1 llm run scoreboard players set s2 llm {dim}')
        f.write(f'function llm:matmul')

        f.write(f'bossbar set progress value {l * 8 + 1}')
        # The inference for larger models can easily surpass the 32-bit command sequence limit,
        # So it has to be scheduled into multiple ticks.
        f.split(True)

        f.write(f'data modify storage llm args.in set value "tx"')
        f.write(f'data modify storage llm args.out set value "k"')
        f.write(f'data modify storage llm args.m set value "wk_{l}"')
        f.write(f'scoreboard players set s1 llm {kv_dim}')
        f.write(f'scoreboard players set s2 llm {dim}')
        f.write(f'function llm:matmul')

        f.write(f'bossbar set progress value {l * 8 + 2}')
        f.split(True)

        f.write(f'data modify storage llm args.in set value "tx"')
        f.write(f'data modify storage llm args.out set value "v"')
        f.write(f'data modify storage llm args.m set value "wv_{l}"')
        f.write(f'scoreboard players set s1 llm {kv_dim}')
        f.write(f'scoreboard players set s2 llm {dim}')
        f.write(f'function llm:matmul')
        
        f.write(f'bossbar set progress value {l * 8 + 3}')
        f.split(True)
        
        f.write(f'function llm:rope')

        # Delete the earliest context if pos > seq_len
        # cache_idx = pos % seq_len
        f.write(f'scoreboard players operation cache_idx llm = pos llm')
        f.write(f'scoreboard players operation cache_idx llm %= {seq_len} llm')
        
        # Save key, value at this time step (pos) to our KV cache
        loff = l * seq_len * dim  # KV cache layer offset for convenience
        # Note: We store K and V in the cache at full dim size, but only fill the first kv_dim elements
        for i in range(kv_dim):
            f.write(f'scoreboard players operation i llm = cache_idx llm')
            f.write(f'scoreboard players operation i llm *= {dim} llm')
            f.write(f'scoreboard players add i llm {loff + i}')
            f.write(f'execute store result storage llm args.i int 1 run scoreboard players get i llm')

            # kc[loff + i + cache_idx * dim] = k[i]
            f.write(f'scoreboard players operation xb llm = {i}b k')
            f.write(f'scoreboard players operation xe llm = {i}e k')
            f.write(f'scoreboard players operation xs llm = {i}s k')
            f.write(f'function llm:set_kc with storage llm args')
            
            # vc[loff + i + cache_idx * dim] = v[i]
            f.write(f'scoreboard players operation xb llm = {i}b v')
            f.write(f'scoreboard players operation xe llm = {i}e v')
            f.write(f'scoreboard players operation xs llm = {i}s v')
            f.write(f'function llm:set_vc with storage llm args')

        f.write(f'function llm:crop_context')

        # Multihead attention. Iterate over all heads
        for h in range(n_heads):
            kv_head = h // kv_mul

            # Iterate over all timesteps, including the current one
            # for t in range(start, pos + 1):
            att_funcname = f'attention_l{l}_h{h}_score'

            f.write(f'scoreboard players operation t llm = start llm')
            f.write(f'scoreboard players set i llm 0')
            f.write(f'execute if score t llm <= pos llm run function llm:{att_funcname}')

            with FunctionWritter(att_funcname) as f1:
                # Get the key vector for this head and at this timestep
                # Each query head maps to a specific KV head

                # cache_idx = t % seq_len * dim + loff + kv_head * head_size
                # score = 0.0
                f1.write(f'scoreboard players operation cache_idx llm = t llm')
                f1.write(f'scoreboard players operation cache_idx llm %= {seq_len} llm')
                f1.write(f'scoreboard players operation cache_idx llm *= {dim} llm')
                f1.write(f'scoreboard players add cache_idx llm {loff + kv_head * head_size}')

                f1.write(f'scoreboard players set scoreb llm 0')
                f1.write(f'scoreboard players set scoree llm 0')
                f1.write(f'scoreboard players set scores llm 0')

                # Calculate the attention score as the dot product of q and k
                for i in range(head_size):
                    # score += q[h * head_size + i] * kc[cache_idx]
                    f1.write(f'scoreboard players operation yb llm = {h * head_size + i}b q')
                    f1.write(f'scoreboard players operation ye llm = {h * head_size + i}e q')
                    f1.write(f'scoreboard players operation ys llm = {h * head_size + i}s q')

                    f1.write(f'execute store result storage llm args.i int 1 run scoreboard players get cache_idx llm')
                    f1.write(f'function llm:get_kc with storage llm args')
                    f1.write(f'scoreboard players add cache_idx llm 1')

                    f1.write(f'function llm:mul')

                    f1.write(f'scoreboard players operation yb llm = scoreb llm')
                    f1.write(f'scoreboard players operation ye llm = scoree llm')
                    f1.write(f'scoreboard players operation ys llm = scores llm')
                    
                    f1.write(f'function llm:add')

                    f1.write(f'scoreboard players operation scoreb llm = xb llm')
                    f1.write(f'scoreboard players operation scoree llm = xe llm')
                    f1.write(f'scoreboard players operation scores llm = xs llm')
                
                # score /= sqrt(head_size)
                sqrt_head_size = math.sqrt(head_size)
                shs_b, shs_e, shs_s = encode_float(sqrt_head_size)
                f1.write(f'scoreboard players operation xb llm = scoreb llm')
                f1.write(f'scoreboard players operation xe llm = scoree llm')
                f1.write(f'scoreboard players operation xs llm = scores llm')

                f1.write(f'scoreboard players set yb llm {shs_b}')
                f1.write(f'scoreboard players set ye llm {shs_e}')
                f1.write(f'scoreboard players set ys llm {shs_s}')

                f1.write(f'function llm:div')

                # Save the score to the attention buffer
                # av[i] = score
                f1.write(f'execute store result storage llm args.i int 1 run scoreboard players get i llm')
                f1.write(f'function llm:set_av with storage llm args')
                
                # t += 1, i += 1, loop back
                f1.write(f'scoreboard players add t llm 1')
                f1.write(f'scoreboard players add i llm 1')
                f1.write(f'execute if score t llm <= pos llm run function llm:{att_funcname}')

            # Softmax the scores to get attention weights, from 0..pos inclusively

            # max = av[0]
            f.write(f'scoreboard players operation maxb llm = 0b av')
            f.write(f'scoreboard players operation maxe llm = 0e av')
            f.write(f'scoreboard players operation maxs llm = 0s av')

            # for i in range(1, window_len):
            max_funcname = f'attention_l{l}_h{h}_max'
            f.write(f'scoreboard players set i llm 1')
            f.write(f'execute if score i llm < window_len llm run function llm:{max_funcname}')

            with FunctionWritter(max_funcname) as f1:
                # if av[i] > max: max = av[i]
                f1.write(f'execute store result storage llm args.i int 1 run scoreboard players get i llm')
                f1.write(f'function llm:get_av with storage llm args')

                f1.write(f'scoreboard players operation yb llm = maxb llm')
                f1.write(f'scoreboard players operation ye llm = maxe llm')
                f1.write(f'scoreboard players operation ys llm = maxs llm')

                f1.write(f'function llm:cmp')
                
                f1.write(f'execute if score r llm matches 1 run scoreboard players operation maxb llm = xb llm')
                f1.write(f'execute if score r llm matches 1 run scoreboard players operation maxe llm = xe llm')
                f1.write(f'execute if score r llm matches 1 run scoreboard players operation maxs llm = xs llm')

                # i += 1, loop back
                f1.write(f'scoreboard players add i llm 1')
                f1.write(f'execute if score i llm < window_len llm run function llm:{max_funcname}')
                
            f.write(f'scoreboard players set expsumb llm 0')
            f.write(f'scoreboard players set expsume llm 0')
            f.write(f'scoreboard players set expsums llm 0')

            # for i in range(window_len):
            esum_funcname = f'attention_l{l}_h{h}_expsum'
            f.write(f'scoreboard players set i llm 0')
            f.write(f'execute if score i llm < window_len llm run function llm:{esum_funcname}')

            with FunctionWritter(esum_funcname) as f1:
                # av[i] = exp(av[i] - max)
                f1.write(f'execute store result storage llm args.i int 1 run scoreboard players get i llm')
                f1.write(f'function llm:get_av with storage llm args')

                f1.write(f'scoreboard players operation yb llm = maxb llm')
                f1.write(f'scoreboard players operation ye llm = maxe llm')
                f1.write(f'scoreboard players operation ys llm = maxs llm')
                f1.write(f'scoreboard players operation ys llm *= -1 llm')
                
                f1.write(f'function llm:add')

                f1.write(f'function llm:exp')
                f1.write(f'function llm:set_av with storage llm args')

                # expsum += av[i]
                f1.write(f'scoreboard players operation yb llm = expsumb llm')
                f1.write(f'scoreboard players operation ye llm = expsume llm')
                f1.write(f'scoreboard players operation ys llm = expsums llm')
                f1.write(f'function llm:add')
                f1.write(f'scoreboard players operation expsumb llm = xb llm')
                f1.write(f'scoreboard players operation expsume llm = xe llm')
                f1.write(f'scoreboard players operation expsums llm = xs llm')

                # i += 1, loop back
                f1.write(f'scoreboard players add i llm 1')
                f1.write(f'execute if score i llm < window_len llm run function llm:{esum_funcname}')

            # for i in range(window_len):
            norm_funcname = f'attention_l{l}_h{h}_norm'
            f.write(f'scoreboard players set i llm 0')
            f.write(f'execute if score i llm < window_len llm run function llm:{norm_funcname}')

            with FunctionWritter(norm_funcname) as f1:
                # av[i] /= expsum
                f1.write(f'execute store result storage llm args.i int 1 run scoreboard players get i llm')
                f1.write(f'function llm:get_av with storage llm args')

                f1.write(f'scoreboard players operation yb llm = expsumb llm')
                f1.write(f'scoreboard players operation ye llm = expsume llm')
                f1.write(f'scoreboard players operation ys llm = expsums llm')

                f1.write(f'function llm:div')
                f1.write(f'function llm:set_av with storage llm args')

                # i += 1, loop back
                f1.write(f'scoreboard players add i llm 1')
                f1.write(f'execute if score i llm < window_len llm run function llm:{norm_funcname}')

            tx_ptr = h * head_size
            # Weighted sum of the values, store back into tx
            for i in range(tx_ptr, tx_ptr + head_size):
                f.write(f'scoreboard players set {i}b tx 0')
                f.write(f'scoreboard players set {i}e tx 0')
                f.write(f'scoreboard players set {i}s tx 0')
            
            # for t in range(start, pos + 1):
            val_funcname = f'attention_l{l}_l{h}_val'
            f.write(f'scoreboard players operation t llm = start llm')
            f.write(f'scoreboard players set i llm 0')
            f.write(f'execute if score t llm <= pos llm run function llm:{val_funcname}')

            with FunctionWritter(val_funcname) as f1:
                # cache_idx = t % seq_len * dim + loff + kv_head * head_size
                f1.write(f'scoreboard players operation cache_idx llm = t llm')
                f1.write(f'scoreboard players operation cache_idx llm %= {seq_len} llm')
                f1.write(f'scoreboard players operation cache_idx llm *= {dim} llm')
                offset = loff + kv_head * head_size
                if offset != 0:
                    f1.write(f'scoreboard players add cache_idx llm {offset}')

                # a = av[i]
                f1.write(f'execute store result storage llm args.i int 1 run scoreboard players get i llm')
                f1.write(f'function llm:get_av with storage llm args')

                f1.write(f'scoreboard players operation ab llm = xb llm')
                f1.write(f'scoreboard players operation ae llm = xe llm')
                f1.write(f'scoreboard players operation as llm = xs llm')

                for i in range(head_size):
                    # xb[tx_ptr + i] += av[t - start] * vc[cache_idx + i]

                    # Get the attention weight for this timestep
                    # Get the value vector for this head and at this timestep
                    
                    f1.write(f'scoreboard players operation yb llm = ab llm')
                    f1.write(f'scoreboard players operation ye llm = ae llm')
                    f1.write(f'scoreboard players operation ys llm = as llm')
                    f1.write(f'execute store result storage llm args.i int 1 run scoreboard players get cache_idx llm')
                    f1.write(f'function llm:get_vc with storage llm args')
                    f1.write(f'function llm:mul')

                    f1.write(f'scoreboard players add cache_idx llm 1')

                    # Accumulate the weighted value into tx

                    f1.write(f'scoreboard players operation yb llm = {tx_ptr + i}b tx')
                    f1.write(f'scoreboard players operation ye llm = {tx_ptr + i}e tx')
                    f1.write(f'scoreboard players operation ys llm = {tx_ptr + i}s tx')
                    
                    f1.write(f'function llm:add')

                    f1.write(f'scoreboard players operation {tx_ptr + i}b tx = xb llm')
                    f1.write(f'scoreboard players operation {tx_ptr + i}e tx = xe llm')
                    f1.write(f'scoreboard players operation {tx_ptr + i}s tx = xs llm')

                # t += 1, i += 1, loop back
                f1.write(f'scoreboard players add t llm 1')
                f1.write(f'scoreboard players add i llm 1')
                f1.write(f'execute if score t llm <= pos llm run function llm:{val_funcname}')
        
        f.write(f'bossbar set progress value {l * 8 + 4}')
        f.split(True)

        # Final matrix multiplication to get the output of the attention
        f.write(f'data modify storage llm args.in set value "tx"')
        f.write(f'data modify storage llm args.out set value "tx2"')
        f.write(f'data modify storage llm args.m set value "wo_{l}"')
        f.write(f'execute store result score s1 llm run scoreboard players set s2 llm {dim}')
        f.write(f'function llm:matmul')

        f.write(f'bossbar set progress value {l * 8 + 5}')
        f.split(True)

        # Residual connection back into x
        for i in range(dim):
            # x[i] += tx2[i]
            f.write(f'scoreboard players operation xb llm = {i}b tx2')
            f.write(f'scoreboard players operation xe llm = {i}e tx2')
            f.write(f'scoreboard players operation xs llm = {i}s tx2')

            f.write(f'scoreboard players operation yb llm = {i}b x')
            f.write(f'scoreboard players operation ye llm = {i}e x')
            f.write(f'scoreboard players operation ys llm = {i}s x')

            f.write(f'function llm:add')
            
            f.write(f'scoreboard players operation {i}b x = xb llm')
            f.write(f'scoreboard players operation {i}e x = xe llm')
            f.write(f'scoreboard players operation {i}s x = xs llm')

        # FFN rmsnorm
        f.write(f'function llm:rmsnorm_0')
        for j in range(dim):
            # tx[j] = rms_ffn_weight[l * dim + j] * ss * x[j]
            idx = l * dim + j
            f.write(f'execute store result score {j}b tx run data get storage llm params.ffn_w.{idx}b')
            f.write(f'execute store result score {j}e tx run data get storage llm params.ffn_w.{idx}e')
            f.write(f'execute store result score {j}s tx run data get storage llm params.ffn_w.{idx}s')
        f.write(f'function llm:rmsnorm_1')

        # Calculate w1(x) and w3(x) for FFN
        f.write(f'data modify storage llm args.in set value "tx"')
        f.write(f'data modify storage llm args.out set value "th"')
        f.write(f'data modify storage llm args.m set value "w1_{l}"')
        f.write(f'scoreboard players set s1 llm {hidden_dim}')
        f.write(f'scoreboard players set s2 llm {dim}')
        f.write(f'function llm:matmul')
        
        f.write(f'bossbar set progress value {l * 8 + 6}')
        f.split(True)

        f.write(f'data modify storage llm args.in set value "tx"')
        f.write(f'data modify storage llm args.out set value "th2"')
        f.write(f'data modify storage llm args.m set value "w3_{l}"')
        f.write(f'scoreboard players set s1 llm {hidden_dim}')
        f.write(f'scoreboard players set s2 llm {dim}')
        f.write(f'function llm:matmul')
        
        f.write(f'bossbar set progress value {l * 8 + 7}')
        f.split(True)
        
        f.write(f'function llm:silu')

        # Final matrix multiplication to get the output of the FFN
        f.write(f'data modify storage llm args.in set value "th"')
        f.write(f'data modify storage llm args.out set value "tx"')
        f.write(f'data modify storage llm args.m set value "w2_{l}"')
        f.write(f'scoreboard players set s1 llm {dim}')
        f.write(f'scoreboard players set s2 llm {hidden_dim}')
        f.write(f'function llm:matmul')

        f.write(f'bossbar set progress value {l * 8 + 8}')
        f.split(True)

        # Residual connection
        for i in range(dim):
            # x[i] += tx[i]
            f.write(f'scoreboard players operation xb llm = {i}b tx')
            f.write(f'scoreboard players operation xe llm = {i}e tx')
            f.write(f'scoreboard players operation xs llm = {i}s tx')

            f.write(f'scoreboard players operation yb llm = {i}b x')
            f.write(f'scoreboard players operation ye llm = {i}e x')
            f.write(f'scoreboard players operation ys llm = {i}s x')

            f.write(f'function llm:add')
            
            f.write(f'scoreboard players operation {i}b x = xb llm')
            f.write(f'scoreboard players operation {i}e x = xe llm')
            f.write(f'scoreboard players operation {i}s x = xs llm')


# Tokenize function
# -----------------

with FunctionWritter('encode') as f:
    # Byte-pair encoding

    whitespace_tok = len(vocab) - vocab[::-1].index(" ") - 1
    # Set the first token to a whitespace
    f.write(f'data modify storage llm args.prompt_tokens set value [{whitespace_tok}]')

    # Enumerate every characters in the prompt string
    f.write(f'scoreboard players set i llm 0')
    f.write(f'scoreboard players set j llm 1')
    f.write(f'execute store result score len llm run data get storage llm args.prompt')
    f.write(f'function llm:enumerate_prompt {{i:0,j:1}}')

    with FunctionWritter('enumerate_prompt') as f1:
        # c = prompt[i]
        f1.write(f'$data modify storage llm args.c set string storage llm args.prompt $(i) $(j)')

        # tok = lookup_vocab(c)
        f1.write(f'function llm:lookup_vocab')
        f1.write(f'execute if score tok llm matches -1 run return run tellraw @a [{{"text":"Not a good prompt at position ","color":"red"}},{{"score":{{"name":"i","objective":"llm"}}}}]')

        # tokens[i] = tok
        f1.write(f'data modify storage llm args.prompt_tokens append value 0')
        f1.write(f'execute store result storage llm args.prompt_tokens[-1] int 1 run scoreboard players get tok llm')

        f1.write(f'execute store result storage llm args.i int 1 run scoreboard players add i llm 1')
        f1.write(f'execute store result storage llm args.j int 1 run scoreboard players add j llm 1')
        f1.write(f'execute if score i llm < len llm run function llm:enumerate_prompt with storage llm args')
    
    # while True
    f.write(f'function llm:encode_iterate')

    with FunctionWritter('encode_iterate') as f1:
        # best_score = -inf
        f1.write(f'scoreboard players set bs llm -2147483648')
        # best_token = -1
        f1.write(f'scoreboard players set bt llm -1')
        # best_idx = -1
        f1.write(f'scoreboard players set bi llm -1')

        f1.write(f'scoreboard players set i llm 0')
        f1.write(f'scoreboard players set j llm 1')
        f1.write(f'execute store result score len llm run data get storage llm args.prompt_tokens')
        f1.write(f'data merge storage llm {{args:{{i:0,j:1}}}}')
        f1.write(f'function llm:get_token_pair with storage llm args')
        f1.write(f'function llm:check_token_pair with storage llm args')

        with FunctionWritter('check_token_pair') as f2:
            # tok = lookup_vocab(tokens[i] + tokens[i + 1])
            f2.write(f'$data modify storage llm args.c set value "$(tok0)$(tok1)"')
            f2.write(f'function llm:lookup_vocab with storage llm args')

            # if tok != -1: merge_tokens()
            f2.write(f'execute unless score tok llm matches -1 run function llm:merge_tokens')

            # loop back
            f2.write(f'execute store result storage llm args.i int 1 run scoreboard players add i llm 1')
            f2.write(f'execute store result storage llm args.j int 1 run scoreboard players add j llm 1')
            f2.write(f'function llm:get_token_pair with storage llm args')
            f2.write(f'execute if score i llm < len llm run function llm:check_token_pair with storage llm args')

        with FunctionWritter('get_token_pair') as f2:
            f2.write(f'$execute store result score tok llm run data get storage llm args.prompt_tokens[$(i)]')
            f2.write(f'function llm:get_token')
            f2.write(f'data modify storage llm args.tok0 set from storage llm args.tok')
            f2.write(f'$execute store result score tok llm run data get storage llm args.prompt_tokens[$(j)]')
            f2.write(f'function llm:get_token')
            f2.write(f'data modify storage llm args.tok1 set from storage llm args.tok')

        with FunctionWritter('merge_tokens') as f2:
            # score = vocab_scores[tok]
            f2.write(f'function llm:get_vocab_score')
            # if best_score > score: return
            f2.write(f'execute if score bs llm >= score llm run return 0')

            f2.write(f'scoreboard players operation bs llm = score llm')
            f2.write(f'scoreboard players operation bt llm = tok llm')
            f2.write(f'scoreboard players operation bi llm = i llm')
        
        # We couldn't find any more pairs to merge, so we're done
        f1.write(f'execute if score bi llm matches -1 run return 0')

        # tokens[best_idx] = best_tok; del tokens[best_idx + 1]
        f1.write(f'execute store result storage llm args.i int 1 run scoreboard players get bi llm')
        f1.write(f'execute store result storage llm args.j int 1 run scoreboard players add bi llm 1')
        f1.write(f'scoreboard players remove bi llm 1')
        f1.write(f'function llm:apply_token_merge with storage llm args')

        with FunctionWritter('apply_token_merge') as f2:
            f2.write(f'$execute store result storage llm args.prompt_tokens[$(i)] int 1 run scoreboard players get bt llm')
            f2.write(f'$data remove storage llm args.prompt_tokens[$(j)]')
        
        # loop back
        f1.write(f'function llm:encode_iterate')

with FunctionWritter('get_vocab_score') as f:
    for i, score in enumerate(vocab_scores):
        f.write(f'execute if score tok llm matches {i} run return run scoreboard players set score llm {score}')

with FunctionWritter('lookup_vocab') as f:
    # Find the last perfect match for string in vocab, return its index or -1 if not found
    for i, tok in enumerate(vocab[::-1]):
        rtok = repr(tok)
        if r"\x" in rtok or r"\r" in rtok or r"\u" in rtok:
            continue
        
        f.write(f'execute if data storage llm args{{c:{tok!r}}} run return run scoreboard players set tok llm {len(vocab) - i - 1}')
    f.write(f'scoreboard players set tok llm -1')


# Sampling functions
# ------------------

# Argmax sampling
with FunctionWritter('argmax') as f:
    # y = -inf
    f.write(f'scoreboard players set yb llm 99999999')
    f.write(f'scoreboard players set ye llm 99999999')
    f.write(f'scoreboard players set ys llm -1')

    # set_token = () => { y = x; tok = i; }
    with FunctionWritter('set_token') as f1:
        f1.write(f'scoreboard players operation yb llm = xb llm')
        f1.write(f'scoreboard players operation ye llm = xe llm')
        f1.write(f'scoreboard players operation ys llm = xs llm')
        f1.write(f'scoreboard players operation tok llm = i llm')

    f.write(f'scoreboard players set i llm 0')
    f.write(f'function llm:argmax_loop {{i:0}}')

    # Setup progress bar
    f.write(f'bossbar set progress max {vocab_size // 100 + 1}')
    f.write(f'bossbar set progress value 0')
    f.write(f'bossbar set progress name [{{"text":"Sampling","color":"gray","bold":true}}]')
    f.write(f'bossbar set progress players @a')
    
    with FunctionWritter('argmax_loop') as f1:
        # x = logits[i]
        f1.write(f'$scoreboard players operation xb llm = $(i)b logits')
        f1.write(f'$scoreboard players operation xe llm = $(i)e logits')
        f1.write(f'$scoreboard players operation xs llm = $(i)s logits')
        
        # if x > y: set_token()
        f1.write(f'function llm:cmp')
        f1.write(f'execute if score r llm matches 1 run function llm:set_token')

        # i += 1; Update progress bar; loop back
        f1.write(f'execute store result score t0 llm store result score t1 llm store result storage llm args.i int 1 run scoreboard players add i llm 1')
        f1.write(f'scoreboard players operation t0 llm %= 100 llm')
        f1.write(f'execute if score t0 llm matches 99 store result bossbar progress value run scoreboard players operation t1 llm /= 100 llm')

        f1.write(f'execute if score i llm matches ..{vocab_size - 1} run function llm:argmax_loop with storage llm args')

# Temperature sampling
with FunctionWritter('temperature_sample') as f:
    # y = t (temperature)
    f.write(f'scoreboard players operation yb llm = tb llm')
    f.write(f'scoreboard players operation ye llm = te llm')
    f.write(f'scoreboard players operation ys llm = ts llm')
    
    # Setup progress bar
    f.write(f'bossbar set progress max {5 * vocab_size // 100 + 1}')
    f.write(f'bossbar set progress value 0')
    f.write(f'bossbar set progress name [{{"text":"Sampling","color":"gray","bold":true}}]')
    f.write(f'bossbar set progress players @a')

    # for i in range(vocab_size):
    f.write(f'scoreboard players set i llm 0')
    f.write(f'function llm:temperature_apply_logits {{i:0}}')

    with FunctionWritter('temperature_apply_logits') as f1:
        # logits[i] /= y
        f1.write(f'$scoreboard players operation xb llm = $(i)b logits')
        f1.write(f'$scoreboard players operation xe llm = $(i)e logits')
        f1.write(f'$scoreboard players operation xs llm = $(i)s logits')
        f1.write(f'function llm:div')
        f1.write(f'$scoreboard players operation $(i)b logits = xb llm')
        f1.write(f'$scoreboard players operation $(i)e logits = xe llm')
        f1.write(f'$scoreboard players operation $(i)s logits = xs llm')

        # i += 1; Update progress bar; loop back
        f1.write(f'execute store result score t0 llm store result score t1 llm store result storage llm args.i int 1 run scoreboard players add i llm 1')
        f1.write(f'scoreboard players operation t0 llm %= 100 llm')
        f1.write(f'execute if score t0 llm matches 99 store result bossbar progress value run scoreboard players operation t1 llm /= 100 llm')
        f1.write(f'execute if score i llm matches ..{vocab_size - 1} run function llm:temperature_apply_logits with storage llm args')

    # Apply softmax to the logits

    # y = -inf
    f.write(f'scoreboard players set yb llm 99999999')
    f.write(f'scoreboard players set ye llm 99999999')
    f.write(f'scoreboard players set ys llm -1')

    # setmax = () => { y = x; tok = i; }
    with FunctionWritter('temperature_setmax') as f1:
        f1.write(f'scoreboard players operation yb llm = xb llm')
        f1.write(f'scoreboard players operation ye llm = xe llm')
        f1.write(f'scoreboard players operation ys llm = xs llm')

    # for i in range(vocab_size):
    f.write(f'scoreboard players set i llm 0')
    f.write(f'function llm:temperature_findmax {{i:0}}')

    with FunctionWritter('temperature_findmax') as f1:
        # x = logits[i]
        f1.write(f'$scoreboard players operation xb llm = $(i)b logits')
        f1.write(f'$scoreboard players operation xe llm = $(i)e logits')
        f1.write(f'$scoreboard players operation xs llm = $(i)s logits')
        
        # if x > y: setmax()
        f1.write(f'function llm:cmp')
        f1.write(f'execute if score r llm matches 1 run function llm:temperature_setmax')
        
        # i += 1; Update progress bar; loop back
        f1.write(f'execute store result score t0 llm store result score t1 llm store result storage llm args.i int 1 run scoreboard players add i llm 1')
        f1.write(f'scoreboard players operation t0 llm %= 100 llm')
        f1.write(f'scoreboard players add t1 llm {vocab_size}')
        f1.write(f'execute if score t0 llm matches 99 store result bossbar progress value run scoreboard players operation t1 llm /= 100 llm')
        f1.write(f'execute if score i llm matches ..{vocab_size - 1} run function llm:temperature_findmax with storage llm args')
    
    # expsum = 0.0
    f.write(f'scoreboard players set expsumb llm 0')
    f.write(f'scoreboard players set expsume llm 0')
    f.write(f'scoreboard players set expsums llm 0')

    # nmax = -y
    f.write(f'scoreboard players operation nmb llm = yb llm')
    f.write(f'scoreboard players operation nme llm = ye llm')
    f.write(f'scoreboard players operation nms llm = ys llm')
    f.write(f'scoreboard players operation nms llm *= -1 llm')

    # for i in range(vocab_size):
    f.write(f'scoreboard players set i llm 0')
    f.write(f'function llm:temperature_expsum {{i:0}}')

    with FunctionWritter('temperature_expsum') as f1:
        # logits[i] = exp(logits[i] + nmax)
        f1.write(f'$scoreboard players operation xb llm = $(i)b logits')
        f1.write(f'$scoreboard players operation xe llm = $(i)e logits')
        f1.write(f'$scoreboard players operation xs llm = $(i)s logits')

        f1.write(f'scoreboard players operation yb llm = nmb llm')
        f1.write(f'scoreboard players operation ye llm = nme llm')
        f1.write(f'scoreboard players operation ys llm = nms llm')

        f1.write(f'function llm:add')
        f1.write(f'function llm:exp')

        f1.write(f'$scoreboard players operation $(i)b logits = xb llm')
        f1.write(f'$scoreboard players operation $(i)e logits = xe llm')
        f1.write(f'$scoreboard players operation $(i)s logits = xs llm')

        # expsum += av[i]
        f1.write(f'scoreboard players operation yb llm = expsumb llm')
        f1.write(f'scoreboard players operation ye llm = expsume llm')
        f1.write(f'scoreboard players operation ys llm = expsums llm')

        f1.write(f'function llm:add')

        f1.write(f'scoreboard players operation expsumb llm = xb llm')
        f1.write(f'scoreboard players operation expsume llm = xe llm')
        f1.write(f'scoreboard players operation expsums llm = xs llm')

        # i += 1; Update progress bar; loop back
        f1.write(f'execute store result score t0 llm store result score t1 llm store result storage llm args.i int 1 run scoreboard players add i llm 1')
        f1.write(f'scoreboard players operation t0 llm %= 100 llm')
        f1.write(f'scoreboard players add t1 llm {2 * vocab_size}')
        f1.write(f'execute if score t0 llm matches 99 store result bossbar progress value run scoreboard players operation t1 llm /= 100 llm')
        f1.write(f'execute if score i llm matches ..{vocab_size - 1} run function llm:temperature_expsum with storage llm args')

    # Normalize

    # for i in range(vocab_size):
    f.write(f'scoreboard players set i llm 0')
    f.write(f'function llm:temperature_norm {{i:0}}')

    with FunctionWritter('temperature_norm') as f1:
        # logits[i] /= expsum
        f1.write(f'$scoreboard players operation xb llm = $(i)b logits')
        f1.write(f'$scoreboard players operation xe llm = $(i)e logits')
        f1.write(f'$scoreboard players operation xs llm = $(i)s logits')

        f1.write(f'scoreboard players operation yb llm = expsumb llm')
        f1.write(f'scoreboard players operation ye llm = expsume llm')
        f1.write(f'scoreboard players operation ys llm = expsums llm')

        f1.write(f'function llm:div')

        f1.write(f'$scoreboard players operation $(i)b logits = xb llm')
        f1.write(f'$scoreboard players operation $(i)e logits = xe llm')
        f1.write(f'$scoreboard players operation $(i)s logits = xs llm')

        # i += 1; Update progress bar; loop back
        f1.write(f'execute store result score t0 llm store result score t1 llm store result storage llm args.i int 1 run scoreboard players add i llm 1')
        f1.write(f'scoreboard players operation t0 llm %= 100 llm')
        f1.write(f'scoreboard players add t1 llm {3 * vocab_size}')
        f1.write(f'execute if score t0 llm matches 99 store result bossbar progress value run scoreboard players operation t1 llm /= 100 llm')
        f1.write(f'execute if score i llm matches ..{vocab_size - 1} run function llm:temperature_norm with storage llm args')

    # Sample from this distribution to get the next token

    # r = random(0, 1)
    f.write(f'execute store result score xb llm run random value 0..99999999')
    f.write(f'scoreboard players set xe llm -1')
    f.write(f'scoreboard players set xs llm 1')
    f.write(f'function llm:float_5')
    f.write(f'execute if score xb llm matches 100000000.. run function llm:float_15')
    f.write(f'scoreboard players operation rb llm = xb llm')
    f.write(f'scoreboard players operation re llm = xe llm')
    f.write(f'scoreboard players operation rs llm = xs llm')

    # cdf = 0.0
    f.write(f'scoreboard players set cdfb llm 0')
    f.write(f'scoreboard players set cdfe llm 0')
    f.write(f'scoreboard players set cdfs llm 0')

    # for i in range(vocab_size):
    f.write(f'scoreboard players set i llm 0')
    f.write(f'function llm:temperature_cdf {{i:0}}')

    with FunctionWritter('temperature_cdf') as f1:
        # cdf += logits[i]; if (cdf > r) return (tok = i)
        f1.write(f'$scoreboard players operation xb llm = $(i)b logits')
        f1.write(f'$scoreboard players operation xe llm = $(i)e logits')
        f1.write(f'$scoreboard players operation xs llm = $(i)s logits')

        f1.write(f'scoreboard players operation yb llm = cdfb llm')
        f1.write(f'scoreboard players operation ye llm = cdfe llm')
        f1.write(f'scoreboard players operation ys llm = cdfs llm')

        f1.write(f'function llm:add')

        f1.write(f'scoreboard players operation cdfb llm = xb llm')
        f1.write(f'scoreboard players operation cdfe llm = xe llm')
        f1.write(f'scoreboard players operation cdfs llm = xs llm')

        f1.write(f'scoreboard players operation yb llm = rb llm')
        f1.write(f'scoreboard players operation ye llm = re llm')
        f1.write(f'scoreboard players operation ys llm = rs llm')

        f1.write(f'function llm:cmp')
        f1.write(f'execute if score r llm matches 1 run return run scoreboard players operation tok llm = i llm')

        # i += 1; Update progress bar; loop back
        f1.write(f'execute store result score t0 llm store result score t1 llm store result storage llm args.i int 1 run scoreboard players add i llm 1')
        f1.write(f'scoreboard players operation t0 llm %= 100 llm')
        f1.write(f'scoreboard players add t1 llm {4 * vocab_size}')
        f1.write(f'execute if score t0 llm matches 99 store result bossbar progress value run scoreboard players operation t1 llm /= 100 llm')
        f1.write(f'execute if score i llm matches ..{vocab_size - 1} run function llm:temperature_cdf with storage llm args')

    # In case of rounding errors
    f.write(f'execute if score i llm matches {vocab_size} run scoreboard players set tok llm {vocab_size - 1}')

with FunctionWritter('get_token') as f:
    for i, tok in enumerate(vocab):
        rtok = repr(tok)
        if r"\x" in rtok or r"\r" in rtok or r"\u" in rtok:
            continue

        f.write(f'execute if score tok llm matches {i} run return run data modify storage llm args.tok set value {tok!r}')


# Generation loop
# ---------------

with FunctionWritter('generate') as f:
    # Get command prompt arguments
    # /function llm:generate {s:<steps>,t:<temperature>,i:<prompt>}
    f.write(f'$scoreboard players set steps llm $(s)')
    f.write(f'$scoreboard players set temperature llm $(t)')

    # if steps < 0: error
    f.write(f'execute if score steps llm matches ..0 run return run tellraw @a {{"text":"Steps must be a positive integer","color":"red"}}')

    # if not 0 <= temperature <= 100: error
    f.write(f'execute if score temperature llm matches ..-1 run return run tellraw @a {{"text":"Temperature must be an integer between 0 and 100 inclusive","color":"red"}}')
    f.write(f'execute if score temperature llm matches 101.. run return run tellraw @a {{"text":"Temperature must be an integer between 0 and 100 inclusive","color":"red"}}')

    # Clear the token buffer
    f.write(f'data modify storage llm args.prompt_tokens set value []')

    # If there is a prompt, tokenize it
    f.write(f'$data modify storage llm args.prompt set value "$(i)"')
    f.write(f'execute unless data storage llm args{{prompt:""}} run function llm:encode')

    # Initiallize
    f.write(f'function llm:setup')
    f.write(f'scoreboard players set tok llm 1')  # BOS token
    f.write(f'scoreboard players set pos llm 0')

    # t = float(temperature / 50)
    f.write(f'scoreboard players operation xb llm = temperature llm')
    f.write(f'scoreboard players operation xb llm *= 2 llm')
    f.write(f'scoreboard players set xe llm 5')
    f.write(f'scoreboard players set xs llm 1')
    f.write(f'function llm:float_5')
    f.write(f'execute if score xb llm matches 100000000.. run function llm:float_15')
    f.write(f'scoreboard players operation tb llm = xb llm')
    f.write(f'scoreboard players operation te llm = xe llm')
    f.write(f'scoreboard players operation ts llm = xs llm')

    # Start the main loop
    f.write(f'function llm:autoregressive')

with FunctionWritter('autoregressive') as f:
    # Forward the transformer to get logits for the next token
    f.write(f'function llm:autoregressive_forward')
    # Make sure the next inference is executed after all the scheduled ticks in llm:forward
    f.write(f'execute if data storage llm args{{prompt_tokens:[]}} run schedule function llm:autoregressive_next {8 * n_layers + 4 + vocab_size // dim}t')
    f.write(f'execute unless data storage llm args{{prompt_tokens:[]}} run schedule function llm:autoregressive_next {8 * n_layers + 1}t')
    
with FunctionWritter('autoregressive_forward') as f:
    f.write(f'execute unless data storage llm args{{prompt_tokens:[]}} run return run function llm:autoregressive_forward_prompt')
    # Setup the progress bar
    # There are 8 updates in each layer, and `vocab_size // dim` updates in the final matrix multiplication
    f.write(f'bossbar set progress max {n_layers * 8 + vocab_size // dim}')
    f.write(f'bossbar set progress value 0')
    f.write(f'bossbar set progress name [{{"text":"Inferring, Step #","color":"green","bold":true}},{{"score":{{"name":"pos","objective":"llm"}}}}]')
    f.write(f'bossbar set progress players @a')
    f.write(f'function llm:forward')

with FunctionWritter('autoregressive_forward_prompt') as f:
    # Setup the progress bar
    # There are 8 updates in each layer
    f.write(f'bossbar set progress max {n_layers * 8}')
    f.write(f'bossbar set progress value 0')
    f.write(f'bossbar set progress name [{{"text":"Processing prompt token #","color":"gold","bold":true}},{{"score":{{"name":"pos","objective":"llm"}}}}]')
    f.write(f'bossbar set progress players @a')
    f.write(f'function llm:forward_hidden')


clear_screen = "\\n" * 100

with FunctionWritter('autoregressive_next') as f:
    # If we are still processing the input prompt, force the next prompt token
    f.write(f'execute unless data storage llm args{{prompt_tokens:[]}} run return run function llm:next_prompt_token')
    
    next_forward_cmds = [
        f'bossbar set progress players',  # Clear progress bar
        f'function llm:get_token',  # Get token string, store into `storage llm args.tok`
        f'data modify storage llm args.output append from storage llm args.tok',  # Append to output buffer
        f'tellraw @a ["{clear_screen}",{{"storage":"llm","nbt":"args.output[]","separator":""}}]',  # Print the output
        f'scoreboard players add pos llm 1',  # Increment position
        f'execute if score pos llm = steps llm run return 1',
        f'function llm:autoregressive',
    ]

    with FunctionWritter('next_prompt_token') as f1:
        f1.write(f'execute store result score tok llm run data get storage llm args.prompt_tokens[0]')
        f1.write(f'data remove storage llm args.prompt_tokens[0]')

        for cmd in next_forward_cmds:
            f1.write(cmd)

    # Otherwise, sample the next token
    f.write(f'execute if score temperature llm matches 0 run function llm:argmax')
    f.write(f'execute unless score temperature llm matches 0 run function llm:temperature_sample')

    for cmd in next_forward_cmds:
        f.write(cmd)

# Resume from generation loop
with FunctionWritter('resume') as f:
    f.write(f'$scoreboard players set steps llm $(s)')
    f.write(f'execute if score steps llm matches ..0 run return run tellraw @a {{"text":"Steps must be a positive integer!","color":"red"}}')
    f.write(f'tellraw @a [{{"text":"Resuming from step #","color":"green"}},{{"score":{{"name":"pos","objective":"llm"}}}}]')
    f.write(f'scoreboard players operation steps llm += pos llm')
    f.write(f'function llm:autoregressive')

# Terminate the generation loop
with FunctionWritter('stop') as f:
    f.write(f'bossbar set progress players')
    f.write(f'schedule clear llm:forward_0')
    for i in range(0, 8 * n_layers):
        f.write(f'schedule clear llm:forward_hidden_{i}')
    f.write(f'schedule clear llm:chat_process_input_0')
    f.write(f'schedule clear llm:respond_0')
    f.write(f'schedule clear llm:autoregressive_next')
    f.write(f'schedule clear llm:matmul_logits_schedule')
    f.write(f'tellraw @a {{"text":"Stopped inferring","color":"red"}}')


# Chat function
with FunctionWritter('chat') as f:
    # Get command prompt arguments
    # /function llm:chat {t:<temperature>,i:<message>}
    f.write(f'$scoreboard players set temperature llm $(t)')

    # if not 0 <= temperature <= 100: error
    f.write(f'execute if score temperature llm matches ..-1 run return run tellraw @a {{"text":"Temperature must be an integer between 0 and 100 inclusive","color":"red"}}')
    f.write(f'execute if score temperature llm matches 101.. run return run tellraw @a {{"text":"Temperature must be an integer between 0 and 100 inclusive","color":"red"}}')

    # Make sure that the message is not empty
    f.write(f'$data modify storage llm args.prompt set value "$(i)"')
    f.write(f'execute if data storage llm args{{prompt:""}} run return run tellraw @a {{"text":"Message must be non-empty","color":"red"}}')
    # Render user/system prompts into the Llama 2 Chat schema
    f.write(f'$data modify storage llm args.prompt set value "[INST] $(i) [/INST]"')

    # Clear the token buffer
    f.write(f'$data modify storage llm args.output append value "\\nUser: $(i)\\nAssistant:"')
    f.write(f'tellraw @a ["{clear_screen}",{{"storage":"llm","nbt":"args.output[]","separator":""}}]')
    f.write(f'data modify storage llm args.prompt_tokens set value []')

    # Tokenize the rendered message
    f.write(f'function llm:encode')
    # Add EOS(1) token at the beginning
    f.write(f'data modify storage llm args.prompt_tokens prepend value 1')
    # Pop the final token into final_tok
    f.write(f'execute store result score final_tok llm run data get storage llm args.prompt_tokens[-1]')
    f.write(f'data remove storage llm args.prompt_tokens[0]')

    # Initiallize
    f.write(f'function llm:setup')

    # t = float(temperature / 50)
    f.write(f'scoreboard players operation xb llm = temperature llm')
    f.write(f'scoreboard players operation xb llm *= 2 llm')
    f.write(f'scoreboard players set xe llm 5')
    f.write(f'scoreboard players set xs llm 1')
    f.write(f'function llm:float_5')
    f.write(f'execute if score xb llm matches 100000000.. run function llm:float_15')
    f.write(f'scoreboard players operation tb llm = xb llm')
    f.write(f'scoreboard players operation te llm = xe llm')
    f.write(f'scoreboard players operation ts llm = xs llm')

    # Process the user input
    f.write(f'scoreboard players set pos llm 0')
    f.write(f'execute store result score len llm run data get storage llm args.prompt_tokens')
    f.write(f'function llm:chat_process_input')

with FunctionWritter('chat_process_input') as f:
    f.write(f'execute store result score tok llm run data get storage llm args.prompt_tokens[0]')
    f.write(f'data remove storage llm args.prompt_tokens[0]')
    # Setup the progress bar
    # There are 8 updates in each layer
    f.write(f'bossbar set progress max {n_layers * 8}')
    f.write(f'bossbar set progress value 0')
    f.write(f'bossbar set progress name [{{"text":"Processing prompt token #","color":"gold","bold":true}},{{"score":{{"name":"pos","objective":"llm"}}}}]')
    f.write(f'bossbar set progress players @a')
    f.write(f'function llm:forward_hidden')
    f.split(True, 8 * n_layers + 1)
    f.write(f'scoreboard players add pos llm 1')

    f.write(f'execute if score pos llm = len llm run scoreboard players operation tok llm = final_tok llm')  # Restore to final_tok
    f.write(f'execute if score pos llm = len llm run return run schedule function llm:respond 1t')

    f.write(f'function llm:chat_process_input')

with FunctionWritter('respond') as f:
    # Setup the progress bar
    # There are 8 updates in each layer, and `vocab_size // dim` updates in the final matrix multiplication
    f.write(f'bossbar set progress max {n_layers * 8 + vocab_size // dim}')
    f.write(f'bossbar set progress value 0')
    f.write(f'bossbar set progress name [{{"text":"Inferring, Step #","color":"green","bold":true}},{{"score":{{"name":"pos","objective":"llm"}}}}]')
    f.write(f'bossbar set progress players @a')
    f.write(f'function llm:forward')
    
    f.split(True, 8 * n_layers + 4 + vocab_size // dim)

    f.write(f'execute if score temperature llm matches 0 run function llm:argmax')
    f.write(f'execute unless score temperature llm matches 0 run function llm:temperature_sample')
    f.write(f'bossbar set progress players')  # Clear progress bar
    f.write(f'execute if score tok llm matches 2 run return 1')  # Break the loop if it generates EOS token
    
    f.write(f'function llm:get_token')
    f.write(f'data modify storage llm args.output append from storage llm args.tok')
    f.write(f'tellraw @a ["{clear_screen}",{{"storage":"llm","nbt":"args.output[]","separator":""}}]')
    f.write(f'scoreboard players add pos llm 1')
    f.write(f'function llm:respond')

# Initialization function
# -----------------------
with FunctionWritter('setup') as f:
    f.write(f'gamerule max_command_sequence_length 2147483647')
    f.write(f'scoreboard objectives add llm dummy')  # Global objective for temp vars
    f.write(f'scoreboard objectives add x dummy')  # Activation at current time stamp (dim,)
    f.write(f'scoreboard objectives add tx dummy')  # Same, but inside a residual branch (dim,)
    f.write(f'scoreboard objectives add tx2 dummy')  # An additional buffer just for convenience (dim,)
    f.write(f'scoreboard objectives add th dummy')  # Buffer for hidden dimension in the ffn (hidden_dim,)
    f.write(f'scoreboard objectives add th2 dummy')  # Buffer for hidden dimension in the ffn (hidden_dim,)
    f.write(f'scoreboard objectives add q dummy')  # Query (dim,)
    f.write(f'scoreboard objectives add k dummy')  # Key (dim,)
    f.write(f'scoreboard objectives add v dummy')  # Value (dim,)
    f.write(f'scoreboard objectives add av dummy')  # Buffer for scores/attention values (n_heads, seq_len)
    f.write(f'scoreboard objectives add kc dummy')  # Key Cache (layer, seq_len, dim)
    f.write(f'scoreboard objectives add vc dummy')  # Value Cache (layer, seq_len, dim)
    f.write(f'scoreboard objectives add logits dummy')  # Output logits

    # Initialize constants
    for n in sorted(consts):
        f.write(f'scoreboard players set {n} llm {n}')
        
    # Create progress bar
    f.write(f'bossbar add progress "Progress"')

    # Initialize parameters
    f.write(f'function llm:params')

    # Clear KV cache
    f.write(f'scoreboard players reset * kc')
    f.write(f'scoreboard players reset * vc')


# Compress into .zip file
pack_meta = {
    "pack": {
        "min_format": 88,
        "max_format": 88,
        "description": f"Llama 2 model (dim={dim}, layers={n_layers})"
    }
}
with open(os.path.join(pack, "pack.mcmeta"), "w") as f:
    json.dump(pack_meta, f, indent=2)

zip_name = f"{pack}.zip"
with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(pack):
        for file in files:
            full_path = os.path.join(root, file)
            arcname = os.path.relpath(full_path, pack)
            zipf.write(full_path, arcname)

# Delete the temporary folder
shutil.rmtree(pack)
print(f"Datapack generated: {zip_name}")
