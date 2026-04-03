import argparse
import json
import math
import shutil
import struct
import os
import zipfile


# Get command line arguments
parser = argparse.ArgumentParser(description="Generate Minecraft functions for Llama 2")
parser.add_argument("checkpoint", help="Path to model checkpoint (.bin)")
parser.add_argument("tokenizer", help="Path to tokenizer (.bin)")
parser.add_argument("pack", help="Create a datapack zip file with this name (without .zip)")
args = parser.parse_args()

checkpoint = args.checkpoint
tokenizer = args.tokenizer
pack = args.pack

base_dir = os.path.join(pack, "data", "llm", "function")
os.makedirs(base_dir, exist_ok=True)

print("Generating datapack...")

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
        vocab_scores.append(struct.unpack('f', tfile.read(4))[0])
        length = struct.unpack('i', tfile.read(4))[0]
        bstr = tfile.read(length)
        if type(bstr) is not str:
            bstr = bstr.decode('utf8')
        
        # Weird llama feature...
        if bstr.startswith('<0x') and bstr.endswith('>') and len(bstr) == 6:
            for c in bstr[3:4]:
                if c not in "1234567890QWERTYUIOPASDFGHJKLZXCVBNM":
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
    MAX_LINECOUNT = 20000
    base_dir = base_dir

    def __init__(self, name: str, progress_bar: bool = False, progress_bar_name: str | None = None):
        self.name = name
        self.progress_bar = progress_bar
        self.part_index = 0
        self._ensure_dir()
        self.current_file = open(os.path.join(self.base_dir, f'{self.name}.mcfunction'), 'w', encoding='utf-8')
        self.line_count = 0
        
        if self.progress_bar:
            if not progress_bar_name:
                progress_bar_name = name
            # Setup the progress bar
            self.current_file.write(f'bossbar set progress name "{progress_bar_name}"\n')
            self.current_file.write(f'bossbar set progress value 0\n')
            self.current_file.write(f'bossbar set progress players @a\n')

    def _ensure_dir(self):
        os.makedirs(self.base_dir, exist_ok=True)

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc, tb):
        self.close()

    def write(self, s: str):
        if not s.endswith('\n'):
            s += '\n'

        if self.line_count > self.MAX_LINECOUNT:
            filename = os.path.join(self.base_dir, f'{self.name}_{self.part_index}.mcfunction')
            if self.current_file:
                # Update progress bar
                if self.progress_bar:
                    self.current_file.write(f'bossbar set progress value {self.part_index}\n')
                    
                self.current_file.write(f'function llm:{self.name}_{self.part_index}\n')
                self.current_file.close()
            self.current_file = open(filename, 'w', encoding='utf-8')
            self.line_count = 0
            self.part_index += 1

        self.current_file.write(s)
        self.line_count += 1
    
    def close(self):
        if self.progress_bar:
            # Remove the progress bar
            self.current_file.write(f'bossbar set progress players\n')

            # Prepend a command in the first file that sets the max value of the progress bar
            with open(f'function/{self.name}.mcfunction', 'r', encoding='utf-8') as f:
                first_file_content = f.read()
            with open(f'function/{self.name}.mcfunction', 'w', encoding='utf-8') as f:
                f.write(f'bossbar set progress max {self.part_index}\n' + first_file_content)

        if self.current_file:
            self.current_file.close()


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
# --------------------------------

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
    ], "float_21": [
        "scoreboard players set xs llm -1",
        "scoreboard players operation xb llm *= -1 llm",
    ], "float_22": [
        "$data modify entity @s Rotation[0] set value $(s)$(b)E$(e)f",
        "execute at @s positioned 0. 0. 0. rotated ~ 0. run tp @s ^ ^ ^1.",
        "execute store result score xb llm run data get entity @s Pos[2] 10000000",
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
    f.write(f'bossbar set progress name "Setting up parameters..."')
    f.write(f'bossbar set progress players @a')

    for i, cmd in enumerate(cmds, 1):
        f.write(cmd)
        f.write(f'bossbar set progress value {i}')

    f.write(f'bossbar set progress players')
        
# Debugging functions
# -------------------

DEBUG = False

def tellraw_float_array(start: str, arr: str, a: int, b: int, write):
    if not DEBUG:
        return
    
    for i in range(a, b):
        write(f'scoreboard players operation xb llm = {i}b {arr}')
        write(f'scoreboard players operation xe llm = {i}e {arr}')
        write(f'scoreboard players operation xs llm = {i}s {arr}')
        write(f'function llm:float_to_nbt')
        write(f'data modify storage llm args.{i} set from storage llm args.x')

    cmd = f'tellraw @a ["{start}[",'
    for i in range(a, b):
        cmd += f'{{"storage":"llm","nbt":"args.{i}"}},", ",'
    cmd = cmd[:-5] + '"]"]\n'
    write(cmd)
    
def tellraw_float_var(start: str, var: str, objective: str, write):
    if not DEBUG:
        return
    
    if var == 'x' and objective == 'llm':
        pass
    else:
        write(f'scoreboard players operation xb llm = {var}b {objective}')
        write(f'scoreboard players operation xe llm = {var}e {objective}')
        write(f'scoreboard players operation xs llm = {var}s {objective}')

    write(f'function llm:float_to_nbt')
    write(f'tellraw @a ["{start}",{{"storage":"llm","nbt":"args.x"}}]')

# Transformer implementation
# --------------------------

# Functions that are commonly used

with FunctionWritter('rmsnorm_0') as f1:
    # ss = 0
    f1.write(f'scoreboard players set ssb llm 0')
    f1.write(f'scoreboard players set sse llm 0')
    f1.write(f'scoreboard players set sss llm 0')
    for i in range(dim):
        # ss += x[i] ** 2
        f1.write(f'scoreboard players operation xb llm = {i}b x')
        f1.write(f'scoreboard players operation xe llm = {i}e x')
        f1.write(f'scoreboard players operation xs llm = {i}s x')
        f1.write(f'function llm:sq')
        f1.write(f'scoreboard players operation yb llm = ssb llm')
        f1.write(f'scoreboard players operation ye llm = sse llm')
        f1.write(f'scoreboard players operation ys llm = sss llm')
        f1.write(f'function llm:add')
        f1.write(f'scoreboard players operation ssb llm = xb llm')
        f1.write(f'scoreboard players operation sse llm = xe llm')
        f1.write(f'scoreboard players operation sss llm = xs llm')

    # ss = 1 / (sqrt(ss / dim + 0.00001))
    f1.write(f'scoreboard players set xb llm {dim}')
    f1.write(f'scoreboard players set xe llm 7')
    f1.write(f'scoreboard players set xs llm 1')
    f1.write(f'function llm:float_5')
    f1.write(f'execute if score xb llm matches 100000000.. run function llm:float_15')
    f1.write(f'scoreboard players operation yb llm = xb llm')
    f1.write(f'scoreboard players operation ye llm = xe llm')
    f1.write(f'scoreboard players operation ys llm = xs llm')
    f1.write(f'scoreboard players operation xb llm = ssb llm')
    f1.write(f'scoreboard players operation xe llm = sse llm')
    f1.write(f'scoreboard players operation xs llm = sss llm')
    f1.write(f'function llm:div')
    f1.write(f'scoreboard players add xb llm 1')
    f1.write(f'execute if score xb llm matches 100000000.. run function llm:float_15')
    f1.write(f'function llm:sqrt')
    f1.write(f'function llm:inv')
    f1.write(f'scoreboard players operation ssb llm = xb llm')
    f1.write(f'scoreboard players operation sse llm = xe llm')
    f1.write(f'scoreboard players operation sss llm = xs llm')
    
with FunctionWritter(f'rmsnorm_1') as f1:
    for i in range(dim):
        # tx[i] *= ss * x[i]
        f1.write(f'scoreboard players operation xb llm = {i}b tx')
        f1.write(f'scoreboard players operation xe llm = {i}e tx')
        f1.write(f'scoreboard players operation xs llm = {i}s tx')
        f1.write(f'scoreboard players operation yb llm = ssb llm')
        f1.write(f'scoreboard players operation ye llm = sse llm')
        f1.write(f'scoreboard players operation ys llm = sss llm')
        f1.write(f'function llm:mul')
        f1.write(f'scoreboard players operation yb llm = {i}b x')
        f1.write(f'scoreboard players operation ye llm = {i}e x')
        f1.write(f'scoreboard players operation ys llm = {i}s x')
        f1.write(f'function llm:mul')
        f1.write(f'scoreboard players operation {i}b tx = xb llm')
        f1.write(f'scoreboard players operation {i}e tx = xe llm')
        f1.write(f'scoreboard players operation {i}s tx = xs llm')

# Matrix multiplication

with FunctionWritter(f'matmul') as f1:
    # i = 0;
    f1.write(f'scoreboard players set i llm 0')
    f1.write(f'data modify storage llm args.i set value 0')

    # idx = 0;
    f1.write(f'scoreboard players set idx llm 0')
    f1.write(f'data modify storage llm args.idx set value 0')
    
    # do {
    f1.write(f'function llm:matmul_1 with storage llm args')
    # } while (1);

with FunctionWritter(f'matmul_1') as f1:
    # out[i] = 0;
    f1.write(f'$scoreboard players set $(i)b $(out) 0')
    f1.write(f'$scoreboard players set $(i)e $(out) 0')
    f1.write(f'$scoreboard players set $(i)s $(out) 0')

    # j = 0;
    f1.write(f'scoreboard players set j llm 0')
    f1.write(f'data modify storage llm args.j set value 0')

    # do {
    f1.write(f'function llm:matmul_2 with storage llm args')
    # } while (1);

    # i++;
    f1.write(f'scoreboard players add i llm 1')

    # if (i == s1) break;
    f1.write(f'execute if score i llm = s1 llm run return 1')

    f1.write(f'execute store result storage llm args.i int 1 run scoreboard players get i llm')
    f1.write(f'function llm:matmul_1 with storage llm args')

with FunctionWritter(f'matmul_2') as f1:
    # x = M[idx];
    f1.write(f'$execute store result score xb llm run data get storage llm params.$(m).$(idx)b')
    f1.write(f'$execute store result score xe llm run data get storage llm params.$(m).$(idx)e')
    f1.write(f'$execute store result score xs llm run data get storage llm params.$(m).$(idx)s')

    # y = I[j];
    f1.write(f'$scoreboard players operation yb llm = $(j)b $(in)')
    f1.write(f'$scoreboard players operation ye llm = $(j)e $(in)')
    f1.write(f'$scoreboard players operation ys llm = $(j)s $(in)')

    # x *= y;
    f1.write(f'function llm:mul')

    # y = O[i];
    f1.write(f'$scoreboard players operation yb llm = $(i)b $(out)')
    f1.write(f'$scoreboard players operation ye llm = $(i)e $(out)')
    f1.write(f'$scoreboard players operation ys llm = $(i)s $(out)')

    # x += y;
    f1.write(f'function llm:add')

    # O[i] = x;
    f1.write(f'$scoreboard players operation $(i)b $(out) = xb llm')
    f1.write(f'$scoreboard players operation $(i)e $(out) = xe llm')
    f1.write(f'$scoreboard players operation $(i)s $(out) = xs llm')
    
    # j++;
    f1.write(f'scoreboard players add j llm 1')

    # idx++;
    f1.write(f'execute store result storage llm args.idx int 1 run scoreboard players add idx llm 1')

    # if (j == s2) break;
    f1.write(f'execute if score j llm = s2 llm run return 1')

    f1.write(f'execute store result storage llm args.j int 1 run scoreboard players get j llm')
    f1.write(f'function llm:matmul_2 with storage llm args')

# Special matmul function that updates the progress bar when caculating logits = x @ emb,
# update the progress bar as computing. It's important since this process takes up most of the time

with FunctionWritter(f'matmul_logits') as f1:
    # i = 0;
    f1.write(f'scoreboard players set i llm 0')
    f1.write(f'data modify storage llm args.i set value 0')

    # idx = 0;
    f1.write(f'scoreboard players set idx llm 0')
    f1.write(f'data modify storage llm args.idx set value 0')
    
    # do {
    f1.write(f'function llm:matmul_logits_1 with storage llm args')
    # } while (1);

with FunctionWritter(f'matmul_logits_1') as f1:
    # logits[i] = 0;
    f1.write(f'$scoreboard players set $(i)b logits 0')
    f1.write(f'$scoreboard players set $(i)e logits 0')
    f1.write(f'$scoreboard players set $(i)s logits 0')

    # j = 0;
    f1.write(f'scoreboard players set j llm 0')
    f1.write(f'data modify storage llm args.j set value 0')
    
    # do {
    f1.write(f'function llm:matmul_logits_2 with storage llm args')
    # } while (1);
    
    # i++;
    f1.write(f'scoreboard players add i llm 1')
    
    # if (i == s1) break;
    f1.write(f'execute if score i llm matches {vocab_size} run return 1')

    f1.write(f'execute store result storage llm args.i int 1 run scoreboard players get i llm')
    f1.write(f'function llm:matmul_logits_1 with storage llm args')

with FunctionWritter(f'matmul_logits_2') as f1:
    # Get Emb[i, j]
    f1.write(f'$execute store result score xb llm run data get storage llm params.emb.$(idx)b')
    f1.write(f'$execute store result score xe llm run data get storage llm params.emb.$(idx)e')
    f1.write(f'$execute store result score xs llm run data get storage llm params.emb.$(idx)s')

    # Get X[j]
    f1.write(f'$scoreboard players operation yb llm = $(j)b x')
    f1.write(f'$scoreboard players operation ye llm = $(j)e x')
    f1.write(f'$scoreboard players operation ys llm = $(j)s x')

    # x = Emb[i, j] * X[j]
    f1.write(f'function llm:mul')
    
    # Get Logits[i]
    f1.write(f'$scoreboard players operation yb llm = $(i)b logits')
    f1.write(f'$scoreboard players operation ye llm = $(i)e logits')
    f1.write(f'$scoreboard players operation ys llm = $(i)s logits')

    # x += Logits[i]
    f1.write(f'function llm:add')

    # Logits[i] = x
    f1.write(f'$scoreboard players operation $(i)b logits = xb llm')
    f1.write(f'$scoreboard players operation $(i)e logits = xe llm')
    f1.write(f'$scoreboard players operation $(i)s logits = xs llm')

    f1.write(f'scoreboard players add j llm 1')

    # Update the progress bar every dim^2 steps
    f1.write(f'scoreboard players add n llm 1')
    f1.write(f'execute if score n llm matches {dim * dim} run function llm:update_logits_progress')
    
    f1.write(f'execute store result storage llm args.idx int 1 run scoreboard players add idx llm 1')
    f1.write(f'execute if score j llm matches {dim} run return 1')
    f1.write(f'execute store result storage llm args.j int 1 run scoreboard players get j llm')
    f1.write(f'function llm:matmul_logits_2 with storage llm args')

with FunctionWritter(f'update_logits_progress') as f1:
    f1.write(f'execute store result score progress llm run bossbar get progress value')
    f1.write(f'execute store result bossbar progress value run scoreboard players add progress llm 1')
    f1.write(f'execute run scoreboard players set n llm 0')

# Copy a row of token embedding table into x
with FunctionWritter('copy_emb') as f1:
    f1.write(f'$execute store result score $(i)b x run data get storage llm params.emb.$(j)b')
    f1.write(f'$execute store result score $(i)e x run data get storage llm params.emb.$(j)e')
    f1.write(f'$execute store result score $(i)s x run data get storage llm params.emb.$(j)s')

    f1.write(f'execute if score i llm matches {dim} run return 0')
    f1.write(f'execute store result storage llm args.i int 1 run scoreboard players add i llm 1')
    f1.write(f'execute store result storage llm args.j int 1 run scoreboard players add j llm 1')
    f1.write(f'function llm:copy_emb with storage llm args')

# SiLU activation function (silu(x) = x * sigmoid(x)) & Elementwise multiply with w3(x)
with FunctionWritter(f'silu') as f1:
    for i in range(hidden_dim):
        # th[i] *= th2[i] / (1 + exp(-th[i]))
        
        # Get th[i]
        f1.write(f'scoreboard players operation xb llm = {i}b th')
        f1.write(f'scoreboard players operation xe llm = {i}e th')
        f1.write(f'scoreboard players operation xs llm = {i}s th')

        tellraw_float_var(f'th[{i}] = ', 'x', 'llm', f1.write)

        # Compute x = exp(-th[i])
        f1.write(f'scoreboard players operation xs llm *= -1 llm')

        tellraw_float_var(f'-th[{i}] = ', 'x', 'llm', f1.write)

        f1.write(f'function llm:exp')

        tellraw_float_var(f'exp(-th[{i}]) = ', 'x', 'llm', f1.write)

        # x += 1.0
        f1.write(f'scoreboard players set yb llm 10000000')
        f1.write(f'scoreboard players set ye llm 0')
        f1.write(f'scoreboard players set ys llm 1')
        f1.write(f'function llm:add')
        
        tellraw_float_var(f'exp(-th[{i}]) + 1 = ', 'x', 'llm', f1.write)

        # x = th2[i] / x
        f1.write(f'scoreboard players operation yb llm = xb llm')
        f1.write(f'scoreboard players operation ye llm = xe llm')
        f1.write(f'scoreboard players operation ys llm = xs llm')
        f1.write(f'scoreboard players operation xb llm = {i}b th2')
        f1.write(f'scoreboard players operation xe llm = {i}e th2')
        f1.write(f'scoreboard players operation xs llm = {i}s th2')
        f1.write(f'function llm:div')

        tellraw_float_var(f'th2[{i}] / (exp(-th[{i}]) + 1) = ', 'x', 'llm', f1.write)

        # x *= th[i]
        f1.write(f'scoreboard players operation yb llm = {i}b th')
        f1.write(f'scoreboard players operation ye llm = {i}e th')
        f1.write(f'scoreboard players operation ys llm = {i}s th')
        f1.write(f'function llm:mul')

        tellraw_float_var(f'th[{i}] * th2[{i}] / (exp(-th[{i}]) + 1) = ', 'x', 'llm', f1.write)

        # th[i] = x
        f1.write(f'scoreboard players operation {i}b th = xb llm')
        f1.write(f'scoreboard players operation {i}e th = xe llm')
        f1.write(f'scoreboard players operation {i}s th = xs llm')

with FunctionWritter(f'ctxwindow') as f1:
    # Keep a fixed context window
    # window_len = min(seq_len, pos + 1)
    # start = pos - window_len + 1
    f1.write(f'scoreboard players operation window_len llm = pos llm')
    f1.write(f'scoreboard players add window_len llm 1')
    f1.write(f'execute if score pos llm matches {seq_len + 1}.. run scoreboard players set window_len llm {seq_len}')
    f1.write(f'scoreboard players operation start llm = pos llm')
    f1.write(f'scoreboard players operation start llm -= window_len llm')
    f1.write(f'scoreboard players add start llm 1')

with FunctionWritter(f'rope') as f1:
    # Apply RoPE (rotary position embedding) to Q and K
    # Note: Q has dim dimensions, K has kv_dim dimensions
    def rotate_v(i, v):
        # v0 = v[i]
        # v1 = v[i + 1]
        # v[i] = v0 * fcr - v1 * fci
        # v[i + 1] = v0 * fci + v1 * fcr

        f1.write(f'scoreboard players operation t0b llm = {i}b {v}')
        f1.write(f'scoreboard players operation t0e llm = {i}e {v}')
        f1.write(f'scoreboard players operation t0s llm = {i}s {v}')

        f1.write(f'scoreboard players operation xb llm = {i}b {v}')
        f1.write(f'scoreboard players operation xe llm = {i}e {v}')
        f1.write(f'scoreboard players operation xs llm = {i}s {v}')
        
        f1.write(f'scoreboard players operation yb llm = fcrb llm')
        f1.write(f'scoreboard players operation ye llm = fcre llm')
        f1.write(f'scoreboard players operation ys llm = fcrs llm')

        f1.write(f'function llm:mul')

        f1.write(f'scoreboard players operation xb llm = {i + 1}b {v}')
        f1.write(f'scoreboard players operation xe llm = {i + 1}e {v}')
        f1.write(f'scoreboard players operation xs llm = {i + 1}s {v}')
        
        f1.write(f'scoreboard players operation yb llm = fcib llm')
        f1.write(f'scoreboard players operation ye llm = fcie llm')
        f1.write(f'scoreboard players operation ys llm = fcis llm')

        f1.write(f'function llm:mul')

        f1.write(f'scoreboard players operation xs llm *= -1 llm')

        f1.write(f'scoreboard players operation yb llm = {i}b {v}')
        f1.write(f'scoreboard players operation ye llm = {i}e {v}')
        f1.write(f'scoreboard players operation ys llm = {i}s {v}')

        f1.write(f'function llm:add')

        f1.write(f'scoreboard players operation {i}b {v} = xb llm')
        f1.write(f'scoreboard players operation {i}e {v} = xe llm')
        f1.write(f'scoreboard players operation {i}s {v} = xs llm')

        f1.write(f'scoreboard players operation xb llm = {i + 1}b {v}')
        f1.write(f'scoreboard players operation xe llm = {i + 1}e {v}')
        f1.write(f'scoreboard players operation xs llm = {i + 1}s {v}')

        f1.write(f'scoreboard players operation yb llm = fcrb llm')
        f1.write(f'scoreboard players operation ye llm = fcre llm')
        f1.write(f'scoreboard players operation ys llm = fcrs llm')

        f1.write(f'function llm:mul')

        f1.write(f'scoreboard players operation {i + 1}b {v} = xb llm')
        f1.write(f'scoreboard players operation {i + 1}e {v} = xe llm')
        f1.write(f'scoreboard players operation {i + 1}s {v} = xs llm')

        f1.write(f'scoreboard players operation xb llm = t0b llm')
        f1.write(f'scoreboard players operation xe llm = t0e llm')
        f1.write(f'scoreboard players operation xs llm = t0s llm')

        f1.write(f'scoreboard players operation yb llm = fcib llm')
        f1.write(f'scoreboard players operation ye llm = fcie llm')
        f1.write(f'scoreboard players operation ys llm = fcis llm')

        f1.write(f'function llm:mul')

        f1.write(f'scoreboard players operation yb llm = {i + 1}b {v}')
        f1.write(f'scoreboard players operation ye llm = {i + 1}e {v}')
        f1.write(f'scoreboard players operation ys llm = {i + 1}s {v}')

        f1.write(f'function llm:add')

        f1.write(f'scoreboard players operation {i + 1}b {v} = xb llm')
        f1.write(f'scoreboard players operation {i + 1}e {v} = xe llm')
        f1.write(f'scoreboard players operation {i + 1}s {v} = xs llm')

    for i in range(0, dim, 2):
        head_dim = i % head_size
        freq = 1 / 10000.0 ** (head_dim / head_size)
        freq_b, freq_e, freq_s = encode_float(freq)

        # v = pos * freq
        f1.write(f'scoreboard players set xb llm {freq_b}')
        f1.write(f'scoreboard players set xe llm {freq_e}')
        f1.write(f'scoreboard players set xs llm {freq_s}')

        f1.write(f'scoreboard players operation yb llm = posb llm')
        f1.write(f'scoreboard players operation ye llm = pose llm')
        f1.write(f'scoreboard players operation ys llm = poss llm')

        f1.write(f'function llm:mul')
    
        f1.write(f'scoreboard players operation vb llm = xb llm')
        f1.write(f'scoreboard players operation ve llm = xe llm')
        f1.write(f'scoreboard players operation vs llm = xs llm')

        # fci = sin(val)
        f1.write(f'function llm:sin')

        f1.write(f'scoreboard players operation fcib llm = xb llm')
        f1.write(f'scoreboard players operation fcie llm = xe llm')
        f1.write(f'scoreboard players operation fcis llm = xs llm')

        f1.write(f'scoreboard players operation xb llm = vb llm')
        f1.write(f'scoreboard players operation xe llm = ve llm')
        f1.write(f'scoreboard players operation xs llm = vs llm')

        # fcr = cos(val)
        f1.write(f'function llm:cos')

        f1.write(f'scoreboard players operation fcrb llm = xb llm')
        f1.write(f'scoreboard players operation fcre llm = xe llm')
        f1.write(f'scoreboard players operation fcrs llm = xs llm')

        # Rotate q[i : i + 2]
        rotate_v(i, 'q')

        # Only rotate k[i : i + 2] if i < kv_dim (K has fewer dimensions than Q)
        if i < kv_dim:
            rotate_v(i, 'k')


# Forward propagation of the transformer
# It's pure torture to debug this
with FunctionWritter('forward') as f:
    # Convert pos (int) to float
    f.write(f'scoreboard players operation xb llm = pos llm')
    f.write(f'scoreboard players set xe llm 7')
    f.write(f'scoreboard players set xs llm 1')
    f.write(f'function llm:float_5')
    f.write(f'execute if score xb llm matches 100000000.. run function llm:float_15')
    f.write(f'scoreboard players operation posb llm = xb llm')
    f.write(f'scoreboard players operation pose llm = xe llm')
    f.write(f'scoreboard players operation poss llm = xs llm')

    # Setup the progress bar
    f.write(f'bossbar set progress max {n_layers * 8 + vocab_size // dim + 1}')
    f.write(f'bossbar set progress value 0')
    f.write(f'bossbar set progress name "Inferencing..."')
    f.write(f'bossbar set progress players @a')

    # Copy the token embedding into x
    f.write(f'scoreboard players operation start llm = tok llm')
    f.write(f'scoreboard players operation start llm *= {dim} llm')

    f.write(f'scoreboard players set i llm 0')
    f.write(f'data modify storage llm args.i set value 0')
    f.write(f'scoreboard players operation j llm = start llm')
    f.write(f'execute store result storage llm args.j int 1 run scoreboard players get j llm')

    f.write(f'function llm:copy_emb with storage llm args')

    tellraw_float_array('x[:10] = ', 'x', 0, 10, f.write)  # Debug

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
        
        tellraw_float_array('tx[:10] = ', 'tx', 0, 10, f.write)  # Debug
        
        # QKV matrix multiplications for this position
        f.write(f'data modify storage llm args.in set value "tx"')
        f.write(f'data modify storage llm args.out set value "q"')
        f.write(f'data modify storage llm args.m set value "wq_{l}"')
        f.write(f'execute store result score s1 llm run scoreboard players set s2 llm {dim}')
        f.write(f'function llm:matmul')

        f.write(f'bossbar set progress value {l * 8 + 1}')

        tellraw_float_array('q[:10] = ', 'q', 0, 10, f.write)  # Debug

        f.write(f'data modify storage llm args.in set value "tx"')
        f.write(f'data modify storage llm args.out set value "k"')
        f.write(f'data modify storage llm args.m set value "wk_{l}"')
        f.write(f'scoreboard players set s1 llm {kv_dim}')
        f.write(f'scoreboard players set s2 llm {dim}')
        f.write(f'function llm:matmul')

        f.write(f'bossbar set progress value {l * 8 + 2}')
        
        tellraw_float_array('k[:10] = ', 'k', 0, 10, f.write)  # Debug

        f.write(f'data modify storage llm args.in set value "tx"')
        f.write(f'data modify storage llm args.out set value "v"')
        f.write(f'data modify storage llm args.m set value "wv_{l}"')
        f.write(f'scoreboard players set s1 llm {kv_dim}')
        f.write(f'scoreboard players set s2 llm {dim}')
        f.write(f'function llm:matmul')
        
        f.write(f'bossbar set progress value {l * 8 + 3}')

        tellraw_float_array('v[:10] = ', 'v', 0, 10, f.write)  # Debug
        
        f.write(f'function llm:rope')

        # Delete the earliest context if pos > seq_len
        # cache_idx = pos % seq_len
        f.write(f'scoreboard players operation cache_idx llm = pos llm')
        f.write(f'scoreboard players operation cache_idx llm %= {seq_len} llm')
        
        # Save key, value at this time step (pos) to our KV cache
        loff = l * seq_len * dim  # KV cache layer offset for convenience
        # Note: We store K and V in the cache at full dim size, but only fill the first kv_dim elements
        # kc[loff + pos * dim : loff + pos * dim + kv_dim] = k[:kv_dim]
        # vc[loff + pos * dim : loff + pos * dim + kv_dim] = v[:kv_dim]
        for i in range(kv_dim):
            # kc[loff + i + cache_idx * dim] = k[i]
            # vc[loff + i + cache_idx * dim] = v[i]
            f.write(f'scoreboard players operation i llm = cache_idx llm')
            f.write(f'scoreboard players operation i llm *= {dim} llm')
            f.write(f'scoreboard players add i llm {loff + i}')

            f.write(f'execute store result storage llm args.i int 1 run scoreboard players get i llm')

            f.write(f'scoreboard players operation xb llm = {i}b k')
            f.write(f'scoreboard players operation xe llm = {i}e k')
            f.write(f'scoreboard players operation xs llm = {i}s k')
            
            f.write(f'function llm:set_kc with storage llm args')
            
            f.write(f'scoreboard players operation xb llm = {i}b v')
            f.write(f'scoreboard players operation xe llm = {i}e v')
            f.write(f'scoreboard players operation xs llm = {i}s v')

            f.write(f'function llm:set_vc with storage llm args')

        f.write(f'function llm:ctxwindow')

        # Multihead attention. Iterate over all heads
        for h in range(n_heads):
            kv_head = h // kv_mul

            # Iterate over all timesteps, including the current one
            # for t in range(start, pos + 1):
            att_funcname = f'att_{l}_{h}_0'

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

                f1.write(f'scoreboard players operation scoreb llm = xb llm')
                f1.write(f'scoreboard players operation scoree llm = xe llm')
                f1.write(f'scoreboard players operation scores llm = xs llm')

                # Save the score to the attention buffer
                # av[t - start] = score
                f1.write(f'execute store result storage llm args.i int 1 run scoreboard players get i llm')
                f1.write(f'function llm:set_av with storage llm args')
                
                # t += 1, i += 1, loop back
                f1.write(f'scoreboard players add t llm 1')
                f1.write(f'scoreboard players add i llm 1')
                f1.write(f'execute if score t llm <= pos llm run function llm:{att_funcname}')
            
            tellraw_float_array('av[:10] = ', 'av', 0, 10, f.write)  # Debug

            # Softmax the scores to get attention weights, from 0..pos inclusively

            # max = av[0]
            f.write(f'scoreboard players operation maxb llm = 0b av')
            f.write(f'scoreboard players operation maxe llm = 0e av')
            f.write(f'scoreboard players operation maxs llm = 0s av')

            # for i in range(1, window_len):
            cmp_funcname = f'att_{l}_{h}_1'
            f.write(f'scoreboard players set i llm 1')
            f.write(f'execute if score i llm < window_len llm run function llm:{cmp_funcname}')

            with FunctionWritter(cmp_funcname) as f1:
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
                f1.write(f'execute if score i llm < window_len llm run function llm:{cmp_funcname}')
                
            f.write(f'scoreboard players set expsumb llm 0')
            f.write(f'scoreboard players set expsume llm 0')
            f.write(f'scoreboard players set expsums llm 0')

            # for i in range(window_len):
            esum_funcname = f'att_{l}_{h}_2'
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

                tellraw_float_var('av[i] - max = ', 'x', 'llm', f1.write)

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

            tellraw_float_var('expsum = ', 'expsum', 'llm', f.write)  # Debug

            # for i in range(window_len):
            norm_funcname = f'att_{l}_{h}_3'
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
            val_funcname = f'att_{l}_{h}_4'
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

                # av_i = av[i]
                f1.write(f'execute store result storage llm args.i int 1 run scoreboard players get i llm')
                f1.write(f'function llm:get_av with storage llm args')

                f1.write(f'scoreboard players operation av_ib llm = xb llm')
                f1.write(f'scoreboard players operation av_ie llm = xe llm')
                f1.write(f'scoreboard players operation av_is llm = xs llm')

                tellraw_float_var('av_i = ', 'av_i', 'llm', f1.write)  # Debug

                for i in range(head_size):
                    # xb[tx_ptr + i] += av[t - start] * vc[cache_idx + i]
                    # Get the attention weight for this timestep
                    # Get the value vector for this head and at this timestep
                    
                    f1.write(f'scoreboard players operation yb llm = av_ib llm')
                    f1.write(f'scoreboard players operation ye llm = av_ie llm')
                    f1.write(f'scoreboard players operation ys llm = av_is llm')
                    f1.write(f'execute store result storage llm args.i int 1 run scoreboard players get cache_idx llm')
                    f1.write(f'function llm:get_vc with storage llm args')

                    f1.write(f'function llm:mul')

                    tellraw_float_var('vc_i = ', 'x', 'llm', f1.write)  # Debug

                    f1.write(f'scoreboard players add cache_idx llm 1')

                    # Accumulate the weighted value into tx

                    f1.write(f'scoreboard players operation yb llm = {tx_ptr + i}b tx')
                    f1.write(f'scoreboard players operation ye llm = {tx_ptr + i}e tx')
                    f1.write(f'scoreboard players operation ys llm = {tx_ptr + i}s tx')

                    # tellraw_float_var('tx[i] = ', tx_ptr + i, 'tx', f1.write)  # Debug
                    
                    f1.write(f'function llm:add')

                    f1.write(f'scoreboard players operation {tx_ptr + i}b tx = xb llm')
                    f1.write(f'scoreboard players operation {tx_ptr + i}e tx = xe llm')
                    f1.write(f'scoreboard players operation {tx_ptr + i}s tx = xs llm')

                    tellraw_float_var('tx[i] = ', tx_ptr + i, 'tx', f1.write)  # Debug

                # t += 1, i += 1, loop back
                f1.write(f'scoreboard players add t llm 1')
                f1.write(f'scoreboard players add i llm 1')
                f1.write(f'execute if score t llm <= pos llm run function llm:{val_funcname}')
        
        f.write(f'bossbar set progress value {l * 8 + 4}')

        # Final matrix multiplication to get the output of the attention
        f.write(f'data modify storage llm args.in set value "tx"')
        f.write(f'data modify storage llm args.out set value "tx2"')
        f.write(f'data modify storage llm args.m set value "wo_{l}"')
        f.write(f'execute store result score s1 llm run scoreboard players set s2 llm {dim}')
        f.write(f'function llm:matmul')

        f.write(f'bossbar set progress value {l * 8 + 5}')

        tellraw_float_array('tx[:10] = ', 'tx', 0, 10, f.write)  # Debug
        tellraw_float_array('tx2[:10] = ', 'tx2', 0, 10, f.write)  # Debug

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

        tellraw_float_array('x[:10] = ', 'x', 0, 10, f.write)  # Debug

        # FFN rmsnorm
        f.write(f'function llm:rmsnorm_0')
        for j in range(dim):
            # tx[j] = rms_ffn_weight[l * dim + j] * ss * x[j]
            idx = l * dim + j
            f.write(f'execute store result score {j}b tx run data get storage llm params.ffn_w.{idx}b')
            f.write(f'execute store result score {j}e tx run data get storage llm params.ffn_w.{idx}e')
            f.write(f'execute store result score {j}s tx run data get storage llm params.ffn_w.{idx}s')
        f.write(f'function llm:rmsnorm_1')

        tellraw_float_array('tx[:10] = ', 'tx', 0, 10, f.write)  # Debug

        # Calculate w1(x) and w3(x) for FFN
        f.write(f'data modify storage llm args.in set value "tx"')
        f.write(f'data modify storage llm args.out set value "th"')
        f.write(f'data modify storage llm args.m set value "w1_{l}"')
        f.write(f'scoreboard players set s1 llm {hidden_dim}')
        f.write(f'scoreboard players set s2 llm {dim}')
        f.write(f'function llm:matmul')
        
        f.write(f'bossbar set progress value {l * 8 + 6}')

        tellraw_float_array('th[:10] = ', 'th', 0, 10, f.write)  # Debug

        f.write(f'data modify storage llm args.in set value "tx"')
        f.write(f'data modify storage llm args.out set value "th2"')
        f.write(f'data modify storage llm args.m set value "w3_{l}"')
        f.write(f'scoreboard players set s1 llm {hidden_dim}')
        f.write(f'scoreboard players set s2 llm {dim}')
        f.write(f'function llm:matmul')
        
        f.write(f'bossbar set progress value {l * 8 + 7}')
        
        tellraw_float_array('th[:10] = ', 'th', 0, 10, f.write)  # Debug
        tellraw_float_array('th2[:10] = ', 'th2', 0, 10, f.write)  # Debug
        
        f.write(f'function llm:silu')
        
        tellraw_float_array('th[:10] = ', 'th', 0, 10, f.write)  # Debug

        # Final matrix multiplication to get the output of the FFN
        f.write(f'data modify storage llm args.in set value "th"')
        f.write(f'data modify storage llm args.out set value "tx"')
        f.write(f'data modify storage llm args.m set value "w2_{l}"')
        f.write(f'scoreboard players set s1 llm {dim}')
        f.write(f'scoreboard players set s2 llm {hidden_dim}')
        f.write(f'function llm:matmul')

        f.write(f'bossbar set progress value {l * 8 + 8}')

        tellraw_float_array('tx[:10] = ', 'tx', 0, 10, f.write)  # Debug

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
        
        tellraw_float_array('x[:10] = ', 'x', 0, 10, f.write)  # Debug

    # Final rmsnorm
    f.write('function llm:rmsnorm_0')

    # tellraw_float_var('ss = ', 'ss', 'llm', f.write)  # Debug

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

    tellraw_float_array('x[:10] = ', 'x', 0, 10, f.write)  # Debug

    # Classifier into logits
    f.write(f'function llm:matmul_logits')

    f.write(f'bossbar set progress value {n_layers * 8 + vocab_size // dim + 1}')
    f.write(f'bossbar set progress players')

    tellraw_float_array('logits[:10] = ', 'logits', 0, 10, f.write)  # Debug


# Setup scoreboard objectives & constants & progress bar & parameters, clear key/value cache
with FunctionWritter('setup') as f:
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

    for n in sorted(consts):
        f.write(f'scoreboard players set {n} llm {n}')
        
    f.write(f'bossbar add progress "Progress"')
    f.write(f'function llm:params')
    f.write(f'scoreboard players reset * kc')
    f.write(f'scoreboard players reset * vc')

# Argmax sampling, too tired to implement anything more complex
with FunctionWritter('argmax', True, 'Sampling...') as f:
    f.write(f'scoreboard players set yb llm 99999999')
    f.write(f'scoreboard players set ye llm 1000')
    f.write(f'scoreboard players set ys llm -1')

    for i in range(vocab_size):
        f.write(f'scoreboard players operation xb llm = {i}b logits')
        f.write(f'scoreboard players operation xe llm = {i}e logits')
        f.write(f'scoreboard players operation xs llm = {i}s logits')
        f.write(f'function llm:cmp')
        f.write(f'execute if score r llm matches 1 run scoreboard players operation yb llm = xb llm')
        f.write(f'execute if score r llm matches 1 run scoreboard players operation ye llm = xe llm')
        f.write(f'execute if score r llm matches 1 run scoreboard players operation ys llm = xs llm')
        f.write(f'execute if score r llm matches 1 run scoreboard players set max_i llm {i}')

with FunctionWritter('sample') as f:
    for i, vocab in enumerate(vocab):
        repr_vocab = repr(vocab)
        if "\\x" in repr_vocab or "\\r" in repr_vocab or "\\u" in repr_vocab:
            continue
        f.write(f'execute if score max_i llm matches {i} run return run data modify storage llm args.tokens append value {vocab!r}')

# Compress into .zip file
pack_meta = {
    "pack": {
        "pack_format": 15,
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
