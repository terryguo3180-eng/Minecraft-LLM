# Complete llama2 implementation in pure Minecraft commands
# TerryGuo 4/17/2026

# v0.1: Added basic transformer forward logic, can generate text from scratch
# v0.2: Added temperature sampling & sentencepiece tokenizer, can generate text that follows a given
#       prompt, also with a bit of uncertainty
# v0.3: Added chat function

from __future__ import annotations

import base64
import contextlib
import json
import math
import lzma
import shutil
import struct
import os
import zipfile


def encode_float(n):
    # Encodes a float number into (base, exponent, sign)
    # n = s*b*10^(e-7)
    # This is the format used in the math lib
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


# Helper context class for writing .mcfunction files
class FunctionWritter:
    def __init__(self, pack: str, name: str, max_lineno: int = 10000):
        self.pack = pack
        self.base_dir = os.path.join(pack, "data", pack, "function")
        os.makedirs(self.base_dir, exist_ok=True)
        self.name = name
        self.npart = 0
        # Ensure the directory exists
        os.makedirs(self.base_dir, exist_ok=True)
        # Current file
        self.cfile = open(os.path.join(self.base_dir, f'{name}.mcfunction'), 'w', encoding='utf-8')
        self.lineno = 0
        self.max_lineno = max_lineno

    def __enter__(self):
        return self
    
    def __exit__(self, a, b, c):
        self.close()

    def split(self, schedule_ticks: int = 0) -> str:
        funcname = f'{self.name}_{self.npart}'
        filename = os.path.join(self.base_dir, f'{funcname}.mcfunction')

        if self.cfile:
            pack = self.pack
            if schedule_ticks != 0:
                self.cfile.write(f'schedule function {pack}:{funcname} {schedule_ticks}t\n')
            else:
                self.cfile.write(f'function {pack}:{funcname}\n')
            self.cfile.close()

        self.cfile = open(filename, 'w', encoding='utf-8')
        self.lineno = 0
        self.npart += 1

        return funcname

    def write(self, s: str):
        if not s.endswith('\n'):
            s += '\n'

        if self.lineno > self.max_lineno:
            self.split()

        self.cfile.write(s)
        self.lineno += 1
    
    def close(self):
        if self.cfile:
            self.cfile.close()


class Llama2DatapackGenerator:
    def __init__(self, model: Llama2Model, pack: str):
        self.model = model
        
        if not pack.endswith('.zip'):
            raise ValueError("datapack must be in .zip format")
        self.pack = pack.rstrip('.zip')

        self.consts = set()  # Scoreboard constants

    def write_function(self, name: str, max_lineno: int = 100000):
        return FunctionWritter(self.pack, name, max_lineno)
    
    def generate(self):
        pack = self.pack
        try:
            print("Generating datapack...")
            for key in dir(self):
                if key.startswith('_generate_'):
                    getattr(self, key, lambda: None)()

        except Exception as e:
            print(f'Encountered unexpected exception: {e}: {e.args}')

        else:
            pack_meta = {
                "pack": {
                    "min_format": 88,
                    "max_format": 88,
                    "description": f"Llama 2 model (dim={self.model.dim}, layers={self.model.n_layers})"
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
            print(f"Datapack generated: {zip_name}")

        finally:
            shutil.rmtree(pack)

    def _generate_tensors_getter_setter(self):
        pack = self.pack
        for tensor in ['kc', 'vc', 'av']:
            with self.write_function(f'get_{tensor}') as f:
                f.write(f'$scoreboard players operation xb {pack} = $(i)b {tensor}')
                f.write(f'$scoreboard players operation xe {pack} = $(i)e {tensor}')
                f.write(f'$scoreboard players operation xs {pack} = $(i)s {tensor}')

            with self.write_function(f'set_{tensor}') as f:
                f.write(f'$scoreboard players operation $(i)b {tensor} = xb {pack}')
                f.write(f'$scoreboard players operation $(i)e {tensor} = xe {pack}')
                f.write(f'$scoreboard players operation $(i)s {tensor} = xs {pack}')

    def _generate_math_lib(self):
        # Float operations with scoreboard
        # Every function except exp are adapted from XiaoDou's Math Lib
        # I'm using compressed string here just for fun :D
        for name, cmds in json.loads(lzma.decompress(base64.b85decode(
            '{Wp48S^xk9=GL@E0stWa8~^|S5YJf5;BRmS`&|G%h=?YPofvqrFL1af7#bp6Yr7Y1S<>26W5Fd2cdUD#sTiIrE}k<o$C0xHtO$s;'
            '+pJ`3D->H`BBc<S;K{GLX83}mmug2M<EzaHP<hb%i7dKLKg%YJJ6C)vF}Nb%__PQ3@AHy`;ObnyzKdS<tdUQQ5CHo3SXAcW6X1-&'
            'j+77TrS(R6u(0T|g)yxk*67r}+#qI4Hj}l(y%=tMP*^m3(rlDww0;#>hS=|>Y8{Tx<R|Jo_#Z&5h^8ou;LZ1>;JiI(+B|M!7J$Zi'
            'T3A7M`~cX(>05bb_@4Z?4n=~lOSvfFk{qLz*qDYsD-%L{C#Xl+I0Qg{C%Nnz{?e=rvFjWY{()~KRASUi-~rXXrNGkwquVDKBAZdx'
            'fGT@Pz)%Qf_Qck2(kgNrclU#wJEvwwP4R&xVZ0Pl;=!Xv(TRTx7HB{dSXysB?3i6BzpSDp-{8XoVUII|r?UtB;fdslP#%2z*F?Qa'
            'PrU&sfuZ@Tei<il?sbOL$i!=jtC3BqqZ4Qjz?LJ4s+jw^d9V#+<!s)-Jtj1vjc#_ypY93E<Gvkjl0VFGqAd(f?isq@ms9J6Jb|gc'
            'ZC;YKdQxIMobm7+X4jHxyd8m~;=&f~JU;v%d91+CQjUB}7<7FPQMWT%9Q~D39C)QxkI794;_Qe!ANUZ@T`|qO6$EJ+F1Bha>r!Z&'
            'ED-EEJBAr;h6Ql7V(I(?&dzK%fP;IeV-MtRhE?}BsJnlLoi7kL%a%ISMQY}aZ-@0DF=zZmsgy2N)Awd#l`Uuxh<h4E%No1R)fvn('
            'yfCX*2Y{?Rf@^@w>2-+;$fN51;rxb+l*Le!%tW?or^rxWb634ujX_yMR@f$Jh`cxV)LGpaxM)8~$Uc!!cs2A?WuUdAE=y8OZpGME'
            'Mw!$xQjPYG2cy`+THX={mXOdsVZl*O?vPHNsX7NVsBpY!jKOMNbvg=ng^Jzuq$3AZi+ey`OXrhyS-L?jD<?7%%FQYTEv<TLcI9IZ'
            'LR_w)v;!bS;!)|(Phnz|+v{zARm|nVFeX+bK@1vs;cd=lC5{{2UG_ShOcO2yEmWDOj90!VVQ)?yCs=N554D0K1gQxKG8VY8JD6n0'
            'qiW8G$92F+>PP%O$n)2ONS$l%{5nZUKiEMu5KBCn)3)RBC(*A4$Vs9_me^X~Y8z@(B)BK-sWAI0^;jcnaJDRJH;}S$vH^on;1NR>'
            'vmJh(gJJaKxN{WKy@#RO)w~(P4y546b12}0cR@&h@*<$kE_thKGBCmhf4#5*gi96j%51aZIL>7on;Ls0cMT-=b*j3)x@@j$Bq(_u'
            's)PA34F8D%d6|F}n^pSj0kjYruBA{HZEHZq=nr-%Aeq!Z6)*6v_vi!ADDXXigmR(1+!h6;q_u&E(2zayY1BzZ<Je@=%sxC|<&c~y'
            '`M54l5}|7wLR6noKEYUUwP8m4Z8c2qTUaG6=VTokuX$-i-aw8iu}LF`cGovEL}<YV^0DxsGnZ}t^pSgDEif14F#|7a8mOrQjy7dT'
            'Wo-E-sFNN;Z=%!aq>Ueu05LKAK|3c!3sG$KOgb?@)59+IF-V3dZ93B6c#S_a5w*8Q`PgD0Ru^C-y7iF!2QK8@BiBM|nTavJrYoDX'
            'i9+^jo6zJ5x|?w{kk{xXgxa-u_v0I0+*edlpZ(m8TonitwO>}uiw<$biuzQuXCg5DIY<{n^ml5hgv~EI8+;sZdG!*3lOaaUDM63S'
            'QCpUCF%+Ry!+M2@a&}tju7@nE*BXmb&nyvT%Q1;fhuV#X%Fd5k1=-0aH0gN2m?jpI0N<CSA*2(-;;3bi@7mX1#d(L4Yx40#L0B&2'
            '-07Mq(I`6cj7s|T-X<(fpfh);o}Dn-#zS#&Lse?Cn?U0L*VezxyIZ}MFHwtyXg-}V$+jl{N@s(Xe8}gO<W|$AAqko`bf%~&oIOfd'
            '{71GzZzSdy*cUCn$ZP-llUL7?so_nAK-|_9Q0|dA(V}d8v_*K;)#_5lwTmk-4_}<JzIVyEU`Z;v9f&57#Sakn+=+FFwx~Xo(ZrWC'
            'k8jFv%AmaJiwftPItV(}j>L?I7C$NN08eI2;4H)GU*{eTvHpidZ6tQfoi<X6_1-e#wFPclh;Kp*+01GrWrmdZX^FC^Rv6h>(HyzD'
            '@jJNX#&2slk7IC6N6(G{t-8($!6_P7IJ+cg0(;M*uSCf67ulTkWnHZm)ucJh8MLV_KPRWn{2#Dk8NQ5cjnE>G3nEUdy3UsRRx-kz'
            '4gqG6=dr5j$P1rO>4Ix>V+j;WxVoM6kmYaS_@d5Rk=GY<kp6A@(CN9yD?qU($###7-937rDF6Tf%uJF}BIUtl00EZ{@!kOd{uI#M'
            'vBYQl0ssI200dcD'
        ))).items():
            with self.write_function(name) as f:
                for cmd in cmds: f.write(cmd.replace('%n', self.pack))
        # Constants used in the math lib
        self.consts.update({-1, 2, 25, 40, 500, 4750, 24703, 79249, *{10**n for n in range(1, 10)}})

    def _generate_params_mcfunction(self):
        cmds = []
        cmds.extend(self._get_dump_params_cmds(self.model.token_embedding_table, 'params.emb'))
        cmds.extend(self._get_dump_params_cmds(self.model.rms_att_weight, 'params.att_w'))
        cmds.extend(self._get_dump_params_cmds(self.model.rms_ffn_weight, 'params.ffn_w'))

        dim = self.model.dim
        kv_dim = (dim * self.model.n_kv_heads) // self.model.n_heads
        hidden_dim = self.model.hidden_dim
        for l in range(self.model.n_layers):
            cmds.extend(self._get_dump_params_cmds(self.model.wq[l * dim * dim : (l + 1) * dim * dim], f'params.wq_{l}'))
            cmds.extend(self._get_dump_params_cmds(self.model.wk[l * kv_dim * dim : (l + 1) * kv_dim * dim], f'params.wk_{l}'))
            cmds.extend(self._get_dump_params_cmds(self.model.wv[l * kv_dim * dim : (l + 1) * kv_dim * dim], f'params.wv_{l}'))
            cmds.extend(self._get_dump_params_cmds(self.model.wo[l * dim * dim : (l + 1) * dim * dim], f'params.wo_{l}'))
            cmds.extend(self._get_dump_params_cmds(self.model.w1[l * hidden_dim * dim : (l + 1) * hidden_dim * dim], f'params.w1_{l}'))
            cmds.extend(self._get_dump_params_cmds(self.model.w2[l * hidden_dim * dim : (l + 1) * hidden_dim * dim], f'params.w2_{l}'))
            cmds.extend(self._get_dump_params_cmds(self.model.w3[l * hidden_dim * dim : (l + 1) * hidden_dim * dim], f'params.w3_{l}'))
        
        with self.write_function('load_parameters') as f:
            # Setup the progress bar
            f.write(f'bossbar set progress max {len(cmds)}')
            f.write(f'bossbar set progress value 0')
            f.write(f'bossbar set progress name {{"text":"Initializing parameters","color":"yellow","bold":true}}')
            f.write(f'bossbar set progress players @a')

            for i, cmd in enumerate(cmds, 1):
                f.write(cmd)
                f.write(f'bossbar set progress value {i}')

            f.write(f'bossbar set progress players')

    def _get_dump_params_cmds(self, params, path: str) -> list[str]:
        # Get the commands that setup a list of parameters
        # Automatically split the command if it reaches command character limit (2M)
        pack = self.pack
        starter = f'data modify storage {pack} {path} set value {{'
        starter_len = len(starter)
        cmds = []
        current_chunk = []
        current_len = starter_len

        for i, value in enumerate(params):
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
                        starter = f'data modify storage {pack} {path} merge value {{'
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

    def _generate_rmsnorm(self):
        pack = self.pack
        with self.write_function('rmsnorm_sum_squares') as f:
            # ss = 0
            f.write(f'scoreboard players set ssb {pack} 0')
            f.write(f'scoreboard players set sse {pack} 0')
            f.write(f'scoreboard players set sss {pack} 0')
            for i in range(self.model.dim):
                # ss += x[i] ** 2
                f.write(f'scoreboard players operation xb {pack} = {i}b x')
                f.write(f'scoreboard players operation xe {pack} = {i}e x')
                f.write(f'scoreboard players operation xs {pack} = {i}s x')
                f.write(f'function {pack}:sq')
                f.write(f'scoreboard players operation yb {pack} = ssb {pack}')
                f.write(f'scoreboard players operation ye {pack} = sse {pack}')
                f.write(f'scoreboard players operation ys {pack} = sss {pack}')
                f.write(f'function {pack}:add')
                f.write(f'scoreboard players operation ssb {pack} = xb {pack}')
                f.write(f'scoreboard players operation sse {pack} = xe {pack}')
                f.write(f'scoreboard players operation sss {pack} = xs {pack}')

            # ss = 1 / (sqrt(ss / dim + 0.00001))
            # Convert dim to float
            f.write(f'scoreboard players set xb {pack} {self.model.dim}')
            f.write(f'scoreboard players set xe {pack} 7')
            f.write(f'scoreboard players set xs {pack} 1')
            f.write(f'function {pack}:float_5')
            f.write(f'execute if score xb {pack} matches 100000000.. run function {pack}:float_15')

            f.write(f'scoreboard players operation yb {pack} = xb {pack}')
            f.write(f'scoreboard players operation ye {pack} = xe {pack}')
            f.write(f'scoreboard players operation ys {pack} = xs {pack}')
            f.write(f'scoreboard players operation xb {pack} = ssb {pack}')
            f.write(f'scoreboard players operation xe {pack} = sse {pack}')
            f.write(f'scoreboard players operation xs {pack} = sss {pack}')
            f.write(f'function {pack}:div')
            f.write(f'scoreboard players add xb {pack} 1')
            f.write(f'execute if score xb {pack} matches 100000000.. run function {pack}:float_15')
            f.write(f'function {pack}:sqrt')
            f.write(f'function {pack}:inv')
            f.write(f'scoreboard players operation ssb {pack} = xb {pack}')
            f.write(f'scoreboard players operation sse {pack} = xe {pack}')
            f.write(f'scoreboard players operation sss {pack} = xs {pack}')
    
        with self.write_function(f'rmsnorm_apply') as f:
            for i in range(self.model.dim):
                # tx[i] *= ss * x[i]
                f.write(f'scoreboard players operation xb {pack} = {i}b tx')
                f.write(f'scoreboard players operation xe {pack} = {i}e tx')
                f.write(f'scoreboard players operation xs {pack} = {i}s tx')
                f.write(f'scoreboard players operation yb {pack} = ssb {pack}')
                f.write(f'scoreboard players operation ye {pack} = sse {pack}')
                f.write(f'scoreboard players operation ys {pack} = sss {pack}')
                f.write(f'function {pack}:mul')
                f.write(f'scoreboard players operation yb {pack} = {i}b x')
                f.write(f'scoreboard players operation ye {pack} = {i}e x')
                f.write(f'scoreboard players operation ys {pack} = {i}s x')
                f.write(f'function {pack}:mul')
                f.write(f'scoreboard players operation {i}b tx = xb {pack}')
                f.write(f'scoreboard players operation {i}e tx = xe {pack}')
                f.write(f'scoreboard players operation {i}s tx = xs {pack}')
    
    @contextlib.contextmanager
    def _rmsnorm_context(self, func: FunctionWritter):
        pack = self.pack
        func.write(f'function {pack}:rmsnorm_sum_squares')
        yield
        func.write(f'function {pack}:rmsnorm_apply')

    def _generate_matmul(self):
        pack = self.pack
        with self.write_function(f'matmul') as f:
            f.write(f'scoreboard players set i {pack} 0')
            f.write(f'data modify storage {pack} args.i set value 0')

            f.write(f'scoreboard players set idx {pack} 0')
            f.write(f'data modify storage {pack} args.idx set value 0')
            
            f.write(f'function {pack}:matmul_i with storage {pack} args')

        with self.write_function(f'matmul_i') as f:
            f.write(f'$scoreboard players set $(i)b $(out) 0')
            f.write(f'$scoreboard players set $(i)e $(out) 0')
            f.write(f'$scoreboard players set $(i)s $(out) 0')

            f.write(f'scoreboard players set j {pack} 0')
            f.write(f'data modify storage {pack} args.j set value 0')

            f.write(f'function {pack}:matmul_j with storage {pack} args')

            f.write(f'scoreboard players add i {pack} 1')
            f.write(f'execute if score i {pack} = s1 {pack} run return 1')

            f.write(f'execute store result storage {pack} args.i int 1 run scoreboard players get i {pack}')
            f.write(f'function {pack}:matmul_i with storage {pack} args')

        with self.write_function(f'matmul_j') as f:
            f.write(f'$execute store result score xb {pack} run data get storage {pack} params.$(m).$(idx)b')
            f.write(f'$execute store result score xe {pack} run data get storage {pack} params.$(m).$(idx)e')
            f.write(f'$execute store result score xs {pack} run data get storage {pack} params.$(m).$(idx)s')

            f.write(f'$scoreboard players operation yb {pack} = $(j)b $(in)')
            f.write(f'$scoreboard players operation ye {pack} = $(j)e $(in)')
            f.write(f'$scoreboard players operation ys {pack} = $(j)s $(in)')

            f.write(f'function {pack}:mul')

            f.write(f'$scoreboard players operation yb {pack} = $(i)b $(out)')
            f.write(f'$scoreboard players operation ye {pack} = $(i)e $(out)')
            f.write(f'$scoreboard players operation ys {pack} = $(i)s $(out)')

            f.write(f'function {pack}:add')

            f.write(f'$scoreboard players operation $(i)b $(out) = xb {pack}')
            f.write(f'$scoreboard players operation $(i)e $(out) = xe {pack}')
            f.write(f'$scoreboard players operation $(i)s $(out) = xs {pack}')
            
            f.write(f'scoreboard players add j {pack} 1')

            f.write(f'execute store result storage {pack} args.idx int 1 run scoreboard players add idx {pack} 1')
            f.write(f'execute if score j {pack} = s2 {pack} run return 1')

            f.write(f'execute store result storage {pack} args.j int 1 run scoreboard players get j {pack}')
            f.write(f'function {pack}:matmul_j with storage {pack} args')


        # Special matmul function that updates the progress bar when calculating logits = x @ emb,
        # update the progress bar as computing. It's important since this process takes up most of the time
        with self.write_function(f'matmul_logits') as f:
            f.write(f'scoreboard players set i {pack} 0')
            f.write(f'scoreboard players set n {pack} 0')
            f.write(f'data modify storage {pack} args.i set value 0')

            f.write(f'scoreboard players set idx {pack} 0')
            f.write(f'data modify storage {pack} args.idx set value 0')
            
            f.write(f'function {pack}:matmul_logits_i with storage {pack} args')

        with self.write_function(f'matmul_logits_i') as f:
            f.write(f'$scoreboard players set $(i)b logits 0')
            f.write(f'$scoreboard players set $(i)e logits 0')
            f.write(f'$scoreboard players set $(i)s logits 0')

            f.write(f'scoreboard players set j {pack} 0')
            f.write(f'data modify storage {pack} args.j set value 0')
            
            f.write(f'function {pack}:matmul_logits_j with storage {pack} args')
            f.write(f'scoreboard players add i {pack} 1')
            f.write(f'execute if score i {pack} matches {self.model.vocab_size} run return 1')

            f.write(f'scoreboard players add n {pack} 1')
            f.write(f'execute if score n {pack} matches {self.model.dim} run return run function {pack}:matmul_logits_update')

            f.write(f'execute store result storage {pack} args.i int 1 run scoreboard players get i {pack}')
            f.write(f'function {pack}:matmul_logits_i with storage {pack} args')

        with self.write_function(f'matmul_logits_j') as f:
            f.write(f'$execute store result score xb {pack} run data get storage {pack} params.emb.$(idx)b')
            f.write(f'$execute store result score xe {pack} run data get storage {pack} params.emb.$(idx)e')
            f.write(f'$execute store result score xs {pack} run data get storage {pack} params.emb.$(idx)s')

            f.write(f'$scoreboard players operation yb {pack} = $(j)b x')
            f.write(f'$scoreboard players operation ye {pack} = $(j)e x')
            f.write(f'$scoreboard players operation ys {pack} = $(j)s x')

            f.write(f'function {pack}:mul')
            
            f.write(f'$scoreboard players operation yb {pack} = $(i)b logits')
            f.write(f'$scoreboard players operation ye {pack} = $(i)e logits')
            f.write(f'$scoreboard players operation ys {pack} = $(i)s logits')

            f.write(f'function {pack}:add')

            f.write(f'$scoreboard players operation $(i)b logits = xb {pack}')
            f.write(f'$scoreboard players operation $(i)e logits = xe {pack}')
            f.write(f'$scoreboard players operation $(i)s logits = xs {pack}')

            f.write(f'scoreboard players add j {pack} 1')
            f.write(f'execute store result storage {pack} args.idx int 1 run scoreboard players add idx {pack} 1')
            f.write(f'execute if score j {pack} matches {self.model.dim} run return 1')
            f.write(f'execute store result storage {pack} args.j int 1 run scoreboard players get j {pack}')
            f.write(f'function {pack}:matmul_logits_j with storage {pack} args')

        with self.write_function(f'matmul_logits_resume') as f:
            f.write(f'execute store result storage {pack} args.i int 1 run scoreboard players get i {pack}')
            f.write(f'function {pack}:matmul_logits_i with storage {pack} args')

        with self.write_function(f'matmul_logits_update') as f:
            # Update the value in the progress bar; n = 0
            f.write(f'execute store result score progress {pack} run bossbar get progress value')
            f.write(f'execute store result bossbar progress value run scoreboard players add progress {pack} 1')
            f.write(f'scoreboard players set n {pack} 0')

            # Minecraft only updates the bossbar at the end of each tick unless the execution is directly
            # launched by the player. Although the inference process is initially triggered by the player,
            # the current execution environment is actually called by previous `/schedule` commands in forward().
            # So we have to wait another tick for the progress bar to update visually.
            f.write(f'schedule function {pack}:matmul_logits_resume 1t')

    def _generate_copy_embrow(self):
        # Copy a row of token embedding table into x
        pack = self.pack
        with self.write_function('copy_embedding_row') as f:
            f.write(f'$execute store result score $(i)b x run data get storage {pack} params.emb.$(j)b')
            f.write(f'$execute store result score $(i)e x run data get storage {pack} params.emb.$(j)e')
            f.write(f'$execute store result score $(i)s x run data get storage {pack} params.emb.$(j)s')

            f.write(f'execute if score i {pack} matches {self.model.dim} run return 0')
            f.write(f'execute store result storage {pack} args.i int 1 run scoreboard players add i {pack} 1')
            f.write(f'execute store result storage {pack} args.j int 1 run scoreboard players add j {pack} 1')
            f.write(f'function {pack}:copy_embedding_row with storage {pack} args')

    def _generate_silu(self):
        # SiLU activation function (silu(x) = x * sigmoid(x)) & Elementwise multiply with w3(x)
        pack = self.pack
        with self.write_function('silu_activation') as f:
            for i in range(self.model.hidden_dim):
                # th[i] *= th2[i] / (1 + exp(-th[i]))
                
                # Get th[i]
                f.write(f'scoreboard players operation xb {pack} = {i}b th')
                f.write(f'scoreboard players operation xe {pack} = {i}e th')
                f.write(f'scoreboard players operation xs {pack} = {i}s th')

                # Compute x = exp(-th[i])
                f.write(f'scoreboard players operation xs {pack} *= -1 {pack}')

                f.write(f'function {pack}:exp')

                # x += 1.0
                f.write(f'scoreboard players set yb {pack} 10000000')
                f.write(f'scoreboard players set ye {pack} 0')
                f.write(f'scoreboard players set ys {pack} 1')
                f.write(f'function {pack}:add')

                # x = th2[i] / x
                f.write(f'scoreboard players operation yb {pack} = xb {pack}')
                f.write(f'scoreboard players operation ye {pack} = xe {pack}')
                f.write(f'scoreboard players operation ys {pack} = xs {pack}')
                f.write(f'scoreboard players operation xb {pack} = {i}b th2')
                f.write(f'scoreboard players operation xe {pack} = {i}e th2')
                f.write(f'scoreboard players operation xs {pack} = {i}s th2')
                f.write(f'function {pack}:div')

                # x *= th[i]
                f.write(f'scoreboard players operation yb {pack} = {i}b th')
                f.write(f'scoreboard players operation ye {pack} = {i}e th')
                f.write(f'scoreboard players operation ys {pack} = {i}s th')
                f.write(f'function {pack}:mul')

                # th[i] = x
                f.write(f'scoreboard players operation {i}b th = xb {pack}')
                f.write(f'scoreboard players operation {i}e th = xe {pack}')
                f.write(f'scoreboard players operation {i}s th = xs {pack}')

    def _generate_crop_context(self):
        # Keep a fixed context window
        # window_len = min(seq_len, pos + 1)
        # start = pos - window_len + 1
        pack = self.pack
        seq_len = self.model.seq_len
        with self.write_function('crop_context') as f:
            f.write(f'scoreboard players operation window_len {pack} = pos {pack}')
            f.write(f'scoreboard players add window_len {pack} 1')
            f.write(f'execute if score pos {pack} matches {seq_len + 1}.. run scoreboard players set window_len {pack} {seq_len}')
            f.write(f'scoreboard players operation start {pack} = pos {pack}')
            f.write(f'scoreboard players operation start {pack} -= window_len {pack}')
            f.write(f'scoreboard players add start {pack} 1')

    def _generate_rope(self):
        pack = self.pack
        head_size = self.model.dim // self.model.n_heads
        kv_dim = (self.model.dim * self.model.n_kv_heads) // self.model.n_heads
        with self.write_function('rotary_embedding') as f:
            # Apply RoPE (rotary position embedding) to Q and K
            # Note: Q has dim dimensions, K has kv_dim dimensions
            def rotate_v(i, v):
                # v0 = v[i]
                # v1 = v[i + 1]

                # v[i] = v0 * fcr - v1 * fci
                # v[i + 1] = v0 * fci + v1 * fcr

                f.write(f'scoreboard players operation t0b {pack} = {i}b {v}')
                f.write(f'scoreboard players operation t0e {pack} = {i}e {v}')
                f.write(f'scoreboard players operation t0s {pack} = {i}s {v}')

                f.write(f'scoreboard players operation xb {pack} = {i}b {v}')
                f.write(f'scoreboard players operation xe {pack} = {i}e {v}')
                f.write(f'scoreboard players operation xs {pack} = {i}s {v}')
                
                f.write(f'scoreboard players operation yb {pack} = fcrb {pack}')
                f.write(f'scoreboard players operation ye {pack} = fcre {pack}')
                f.write(f'scoreboard players operation ys {pack} = fcrs {pack}')

                f.write(f'function {pack}:mul')

                f.write(f'scoreboard players operation t1b {pack} = xb {pack}')
                f.write(f'scoreboard players operation t1e {pack} = xe {pack}')
                f.write(f'scoreboard players operation t1s {pack} = xs {pack}')

                f.write(f'scoreboard players operation xb {pack} = {i + 1}b {v}')
                f.write(f'scoreboard players operation xe {pack} = {i + 1}e {v}')
                f.write(f'scoreboard players operation xs {pack} = {i + 1}s {v}')
                
                f.write(f'scoreboard players operation yb {pack} = fcib {pack}')
                f.write(f'scoreboard players operation ye {pack} = fcie {pack}')
                f.write(f'scoreboard players operation ys {pack} = fcis {pack}')

                f.write(f'function {pack}:mul')

                f.write(f'scoreboard players operation xs {pack} *= -1 {pack}')

                f.write(f'scoreboard players operation yb {pack} = t1b {pack}')
                f.write(f'scoreboard players operation ye {pack} = t1e {pack}')
                f.write(f'scoreboard players operation ys {pack} = t1s {pack}')

                f.write(f'function {pack}:add')

                f.write(f'scoreboard players operation {i}b {v} = xb {pack}')
                f.write(f'scoreboard players operation {i}e {v} = xe {pack}')
                f.write(f'scoreboard players operation {i}s {v} = xs {pack}')

                f.write(f'scoreboard players operation xb {pack} = {i + 1}b {v}')
                f.write(f'scoreboard players operation xe {pack} = {i + 1}e {v}')
                f.write(f'scoreboard players operation xs {pack} = {i + 1}s {v}')

                f.write(f'scoreboard players operation yb {pack} = fcrb {pack}')
                f.write(f'scoreboard players operation ye {pack} = fcre {pack}')
                f.write(f'scoreboard players operation ys {pack} = fcrs {pack}')

                f.write(f'function {pack}:mul')

                f.write(f'scoreboard players operation t1b {v} = xb {pack}')
                f.write(f'scoreboard players operation t1e {v} = xe {pack}')
                f.write(f'scoreboard players operation t1s {v} = xs {pack}')

                f.write(f'scoreboard players operation xb {pack} = t0b {pack}')
                f.write(f'scoreboard players operation xe {pack} = t0e {pack}')
                f.write(f'scoreboard players operation xs {pack} = t0s {pack}')

                f.write(f'scoreboard players operation yb {pack} = fcib {pack}')
                f.write(f'scoreboard players operation ye {pack} = fcie {pack}')
                f.write(f'scoreboard players operation ys {pack} = fcis {pack}')

                f.write(f'function {pack}:mul')

                f.write(f'scoreboard players operation yb {pack} = t1b {v}')
                f.write(f'scoreboard players operation ye {pack} = t1e {v}')
                f.write(f'scoreboard players operation ys {pack} = t1s {v}')

                f.write(f'function {pack}:add')

                f.write(f'scoreboard players operation {i + 1}b {v} = xb {pack}')
                f.write(f'scoreboard players operation {i + 1}e {v} = xe {pack}')
                f.write(f'scoreboard players operation {i + 1}s {v} = xs {pack}')

            for i in range(0, self.model.dim, 2):
                head_dim = i % head_size
                freq = 1 / 10000.0 ** (head_dim / head_size)
                freq_b, freq_e, freq_s = encode_float(freq)

                # v = pos * freq
                f.write(f'scoreboard players set xb {pack} {freq_b}')
                f.write(f'scoreboard players set xe {pack} {freq_e}')
                f.write(f'scoreboard players set xs {pack} {freq_s}')

                f.write(f'scoreboard players operation yb {pack} = posb {pack}')
                f.write(f'scoreboard players operation ye {pack} = pose {pack}')
                f.write(f'scoreboard players operation ys {pack} = poss {pack}')

                f.write(f'function {pack}:mul')

                # v = degrees(v)
                # Took me forever to trace this bug

                f.write(f'scoreboard players set yb {pack} 17453293')
                f.write(f'scoreboard players set ye {pack} -2')
                f.write(f'scoreboard players set ys {pack} 1')

                f.write(f'function {pack}:div')
            
                f.write(f'scoreboard players operation vb {pack} = xb {pack}')
                f.write(f'scoreboard players operation ve {pack} = xe {pack}')
                f.write(f'scoreboard players operation vs {pack} = xs {pack}')

                # fci = sin(v)
                f.write(f'function {pack}:sin')

                f.write(f'scoreboard players operation fcib {pack} = xb {pack}')
                f.write(f'scoreboard players operation fcie {pack} = xe {pack}')
                f.write(f'scoreboard players operation fcis {pack} = xs {pack}')

                f.write(f'scoreboard players operation xb {pack} = vb {pack}')
                f.write(f'scoreboard players operation xe {pack} = ve {pack}')
                f.write(f'scoreboard players operation xs {pack} = vs {pack}')

                # fcr = cos(v)
                f.write(f'function {pack}:cos')

                f.write(f'scoreboard players operation fcrb {pack} = xb {pack}')
                f.write(f'scoreboard players operation fcre {pack} = xe {pack}')
                f.write(f'scoreboard players operation fcrs {pack} = xs {pack}')

                # Rotate q[i : i + 2]
                rotate_v(i, 'q')

                # Only rotate k[i : i + 2] if i < kv_dim (K has fewer dimensions than Q)
                if i < kv_dim:
                    rotate_v(i, 'k')

    def _generate_forward(self):
        # Forward propagation of the transformer
        # It's pure torture to debug this

        pack = self.pack
        
        dim = self.model.dim
        n_heads = self.model.n_heads
        head_size = dim // n_heads
        kv_dim = (dim * self.model.n_kv_heads) // n_heads
        kv_mul = n_heads // self.model.n_kv_heads
        n_layers = self.model.n_layers
        vocab_size = self.model.vocab_size
        seq_len = self.model.seq_len
        hidden_dim = self.model.hidden_dim

        with self.write_function('forward') as f:
            # Full forward pass including final norm and classifier

            # Setup the progress bar
            # There are 8 updates in each layer, and `vocab_size // dim` updates in the final matrix multiplication
            f.write(f'bossbar set progress max {n_layers * 8 + vocab_size // dim}')
            f.write(f'bossbar set progress value 0')
            f.write(f'bossbar set progress name [{{"text":"Inferring, Step #","color":"green","bold":true}},{{"score":{{"name":"pos","objective":"{pack}"}}}}]')
            f.write(f'bossbar set progress players @a')

            f.write(f'function {pack}:forward_hidden')
            f.split(8 * n_layers + 1)

            # Final rmsnorm
            f.write(f'function {pack}:rmsnorm_sum_squares')

            for i in range(dim):
                # x[i] *= rms_final_weight[i] * ss
                rms_ffn_i = self.model.rms_final_weight[i]
                pb, pe, ps = encode_float(rms_ffn_i)

                f.write(f'scoreboard players operation xb {pack} = {i}b x')
                f.write(f'scoreboard players operation xe {pack} = {i}e x')
                f.write(f'scoreboard players operation xs {pack} = {i}s x')

                f.write(f'scoreboard players set yb {pack} {pb}')
                f.write(f'scoreboard players set ye {pack} {pe}')
                f.write(f'scoreboard players set ys {pack} {ps}')

                f.write(f'function {pack}:mul')

                f.write(f'scoreboard players operation yb {pack} = ssb {pack}')
                f.write(f'scoreboard players operation ye {pack} = sse {pack}')
                f.write(f'scoreboard players operation ys {pack} = sss {pack}')

                f.write(f'function {pack}:mul')

                f.write(f'scoreboard players operation {i}b x = xb {pack}')
                f.write(f'scoreboard players operation {i}e x = xe {pack}')
                f.write(f'scoreboard players operation {i}s x = xs {pack}')

            # Classifier into logits
            f.write(f'function {pack}:matmul_logits')

        with self.write_function('forward_hidden') as f:
            # Process a single token at a given position (prompt processing)

            # Convert pos (int) to float
            f.write(f'scoreboard players operation xb {pack} = pos {pack}')
            f.write(f'scoreboard players set xe {pack} 7')
            f.write(f'scoreboard players set xs {pack} 1')
            f.write(f'function {pack}:float_5')
            f.write(f'execute if score xb {pack} matches 100000000.. run function {pack}:float_15')
            f.write(f'scoreboard players operation posb {pack} = xb {pack}')
            f.write(f'scoreboard players operation pose {pack} = xe {pack}')
            f.write(f'scoreboard players operation poss {pack} = xs {pack}')

            # Copy the token embedding into x
            f.write(f'scoreboard players operation start {pack} = tok {pack}')
            self.consts.add(dim)
            f.write(f'scoreboard players operation start {pack} *= {dim} {pack}')

            f.write(f'scoreboard players set i {pack} 0')
            f.write(f'data modify storage {pack} args.i set value 0')
            f.write(f'scoreboard players operation j {pack} = start {pack}')
            f.write(f'execute store result storage {pack} args.j int 1 run scoreboard players get j {pack}')

            f.write(f'function {pack}:copy_embedding_row with storage {pack} args')

            # Forward all the layers
            for l in range(n_layers):
                # Attention rmsnorm:
                # tx = rmsnorm(x, rms_att_weight[l * dim : (l + 1) * dim])

                with self._rmsnorm_context(f):
                    for j in range(dim):
                        # tx[j] = rms_att_weight[l * dim + j]
                        idx = l * dim + j
                        f.write(f'execute store result score {j}b tx run data get storage {pack} params.att_w.{idx}b')
                        f.write(f'execute store result score {j}e tx run data get storage {pack} params.att_w.{idx}e')
                        f.write(f'execute store result score {j}s tx run data get storage {pack} params.att_w.{idx}s')
                
                # QKV matrix multiplications for this position
                f.write(f'data modify storage {pack} args.in set value "tx"')
                f.write(f'data modify storage {pack} args.out set value "q"')
                f.write(f'data modify storage {pack} args.m set value "wq_{l}"')
                f.write(f'execute store result score s1 {pack} run scoreboard players set s2 {pack} {dim}')
                f.write(f'function {pack}:matmul')

                f.write(f'bossbar set progress value {l * 8 + 1}')
                # The inference for larger models can easily surpass the 32-bit command sequence limit,
                # So it has to be scheduled into multiple ticks.
                f.split(1)

                f.write(f'data modify storage {pack} args.in set value "tx"')
                f.write(f'data modify storage {pack} args.out set value "k"')
                f.write(f'data modify storage {pack} args.m set value "wk_{l}"')
                f.write(f'scoreboard players set s1 {pack} {kv_dim}')
                f.write(f'scoreboard players set s2 {pack} {dim}')
                f.write(f'function {pack}:matmul')

                f.write(f'bossbar set progress value {l * 8 + 2}')
                f.split(1)

                f.write(f'data modify storage {pack} args.in set value "tx"')
                f.write(f'data modify storage {pack} args.out set value "v"')
                f.write(f'data modify storage {pack} args.m set value "wv_{l}"')
                f.write(f'scoreboard players set s1 {pack} {kv_dim}')
                f.write(f'scoreboard players set s2 {pack} {dim}')
                f.write(f'function {pack}:matmul')
                
                f.write(f'bossbar set progress value {l * 8 + 3}')
                f.split(1)
                
                f.write(f'function {pack}:rotary_embedding')

                # Delete the earliest context if pos > seq_len
                # cache_idx = pos % seq_len
                f.write(f'scoreboard players operation cache_idx {pack} = pos {pack}')

                self.consts.add(seq_len)
                f.write(f'scoreboard players operation cache_idx {pack} %= {seq_len} {pack}')
                
                # Save key, value at this time step (pos) to our KV cache
                loff = l * seq_len * dim  # KV cache layer offset for convenience
                # Note: We store K and V in the cache at full dim size, but only fill the first kv_dim elements
                for i in range(kv_dim):
                    f.write(f'scoreboard players operation i {pack} = cache_idx {pack}')
                    f.write(f'scoreboard players operation i {pack} *= {dim} {pack}')
                    f.write(f'scoreboard players add i {pack} {loff + i}')
                    f.write(f'execute store result storage {pack} args.i int 1 run scoreboard players get i {pack}')

                    # kc[loff + i + cache_idx * dim] = k[i]
                    f.write(f'scoreboard players operation xb {pack} = {i}b k')
                    f.write(f'scoreboard players operation xe {pack} = {i}e k')
                    f.write(f'scoreboard players operation xs {pack} = {i}s k')
                    f.write(f'function {pack}:set_kc with storage {pack} args')
                    
                    # vc[loff + i + cache_idx * dim] = v[i]
                    f.write(f'scoreboard players operation xb {pack} = {i}b v')
                    f.write(f'scoreboard players operation xe {pack} = {i}e v')
                    f.write(f'scoreboard players operation xs {pack} = {i}s v')
                    f.write(f'function {pack}:set_vc with storage {pack} args')

                f.write(f'function {pack}:crop_context')

                # Multihead attention. Iterate over all heads
                for h in range(n_heads):
                    kv_head = h // kv_mul

                    # Iterate over all timesteps, including the current one
                    # for t in range(start, pos + 1):
                    att_funcname = f'attention_l{l}_h{h}_compute_score'

                    f.write(f'scoreboard players operation t {pack} = start {pack}')
                    f.write(f'scoreboard players set i {pack} 0')
                    f.write(f'execute if score t {pack} <= pos {pack} run function {pack}:{att_funcname}')

                    with self.write_function(att_funcname) as f1:
                        # Get the key vector for this head and at this timestep
                        # Each query head maps to a specific KV head

                        # cache_idx = t % seq_len * dim + loff + kv_head * head_size
                        # score = 0.0
                        f1.write(f'scoreboard players operation cache_idx {pack} = t {pack}')
                        f1.write(f'scoreboard players operation cache_idx {pack} %= {seq_len} {pack}')
                        f1.write(f'scoreboard players operation cache_idx {pack} *= {dim} {pack}')
                        f1.write(f'scoreboard players add cache_idx {pack} {loff + kv_head * head_size}')

                        f1.write(f'scoreboard players set scoreb {pack} 0')
                        f1.write(f'scoreboard players set scoree {pack} 0')
                        f1.write(f'scoreboard players set scores {pack} 0')

                        # Calculate the attention score as the dot product of q and k
                        for i in range(head_size):
                            # score += q[h * head_size + i] * kc[cache_idx]
                            f1.write(f'scoreboard players operation yb {pack} = {h * head_size + i}b q')
                            f1.write(f'scoreboard players operation ye {pack} = {h * head_size + i}e q')
                            f1.write(f'scoreboard players operation ys {pack} = {h * head_size + i}s q')

                            f1.write(f'execute store result storage {pack} args.i int 1 run scoreboard players get cache_idx {pack}')
                            f1.write(f'function {pack}:get_kc with storage {pack} args')
                            f1.write(f'scoreboard players add cache_idx {pack} 1')

                            f1.write(f'function {pack}:mul')

                            f1.write(f'scoreboard players operation yb {pack} = scoreb {pack}')
                            f1.write(f'scoreboard players operation ye {pack} = scoree {pack}')
                            f1.write(f'scoreboard players operation ys {pack} = scores {pack}')
                            
                            f1.write(f'function {pack}:add')

                            f1.write(f'scoreboard players operation scoreb {pack} = xb {pack}')
                            f1.write(f'scoreboard players operation scoree {pack} = xe {pack}')
                            f1.write(f'scoreboard players operation scores {pack} = xs {pack}')
                        
                        # score /= sqrt(head_size)
                        sqrt_head_size = math.sqrt(head_size)
                        shs_b, shs_e, shs_s = encode_float(sqrt_head_size)
                        f1.write(f'scoreboard players operation xb {pack} = scoreb {pack}')
                        f1.write(f'scoreboard players operation xe {pack} = scoree {pack}')
                        f1.write(f'scoreboard players operation xs {pack} = scores {pack}')

                        f1.write(f'scoreboard players set yb {pack} {shs_b}')
                        f1.write(f'scoreboard players set ye {pack} {shs_e}')
                        f1.write(f'scoreboard players set ys {pack} {shs_s}')

                        f1.write(f'function {pack}:div')

                        # Save the score to the attention buffer
                        # av[i] = score
                        f1.write(f'execute store result storage {pack} args.i int 1 run scoreboard players get i {pack}')
                        f1.write(f'function {pack}:set_av with storage {pack} args')
                        
                        # t += 1, i += 1, loop back
                        f1.write(f'scoreboard players add t {pack} 1')
                        f1.write(f'scoreboard players add i {pack} 1')
                        f1.write(f'execute if score t {pack} <= pos {pack} run function {pack}:{att_funcname}')

                    # Softmax the scores to get attention weights, from 0..pos inclusively

                    # max = av[0]
                    f.write(f'scoreboard players operation maxb {pack} = 0b av')
                    f.write(f'scoreboard players operation maxe {pack} = 0e av')
                    f.write(f'scoreboard players operation maxs {pack} = 0s av')

                    # for i in range(1, window_len):
                    max_funcname = f'attention_l{l}_h{h}_find_max'
                    f.write(f'scoreboard players set i {pack} 1')
                    f.write(f'execute if score i {pack} < window_len {pack} run function {pack}:{max_funcname}')

                    with self.write_function(max_funcname) as f1:
                        # if av[i] > max: max = av[i]
                        f1.write(f'execute store result storage {pack} args.i int 1 run scoreboard players get i {pack}')
                        f1.write(f'function {pack}:get_av with storage {pack} args')

                        f1.write(f'scoreboard players operation yb {pack} = maxb {pack}')
                        f1.write(f'scoreboard players operation ye {pack} = maxe {pack}')
                        f1.write(f'scoreboard players operation ys {pack} = maxs {pack}')

                        f1.write(f'function {pack}:cmp')
                        
                        f1.write(f'execute if score r {pack} matches 1 run scoreboard players operation maxb {pack} = xb {pack}')
                        f1.write(f'execute if score r {pack} matches 1 run scoreboard players operation maxe {pack} = xe {pack}')
                        f1.write(f'execute if score r {pack} matches 1 run scoreboard players operation maxs {pack} = xs {pack}')

                        # i += 1, loop back
                        f1.write(f'scoreboard players add i {pack} 1')
                        f1.write(f'execute if score i {pack} < window_len {pack} run function {pack}:{max_funcname}')
                        
                    f.write(f'scoreboard players set expsumb {pack} 0')
                    f.write(f'scoreboard players set expsume {pack} 0')
                    f.write(f'scoreboard players set expsums {pack} 0')

                    # for i in range(window_len):
                    esum_funcname = f'attention_l{l}_h{h}_sum_exp'
                    f.write(f'scoreboard players set i {pack} 0')
                    f.write(f'execute if score i {pack} < window_len {pack} run function {pack}:{esum_funcname}')

                    with self.write_function(esum_funcname) as f1:
                        # av[i] = exp(av[i] - max)
                        f1.write(f'execute store result storage {pack} args.i int 1 run scoreboard players get i {pack}')
                        f1.write(f'function {pack}:get_av with storage {pack} args')

                        f1.write(f'scoreboard players operation yb {pack} = maxb {pack}')
                        f1.write(f'scoreboard players operation ye {pack} = maxe {pack}')
                        f1.write(f'scoreboard players operation ys {pack} = maxs {pack}')
                        f1.write(f'scoreboard players operation ys {pack} *= -1 {pack}')
                        
                        f1.write(f'function {pack}:add')

                        f1.write(f'function {pack}:exp')
                        f1.write(f'function {pack}:set_av with storage {pack} args')

                        # expsum += av[i]
                        f1.write(f'scoreboard players operation yb {pack} = expsumb {pack}')
                        f1.write(f'scoreboard players operation ye {pack} = expsume {pack}')
                        f1.write(f'scoreboard players operation ys {pack} = expsums {pack}')
                        f1.write(f'function {pack}:add')
                        f1.write(f'scoreboard players operation expsumb {pack} = xb {pack}')
                        f1.write(f'scoreboard players operation expsume {pack} = xe {pack}')
                        f1.write(f'scoreboard players operation expsums {pack} = xs {pack}')

                        # i += 1, loop back
                        f1.write(f'scoreboard players add i {pack} 1')
                        f1.write(f'execute if score i {pack} < window_len {pack} run function {pack}:{esum_funcname}')

                    # for i in range(window_len):
                    norm_funcname = f'attention_l{l}_h{h}_normalize'
                    f.write(f'scoreboard players set i {pack} 0')
                    f.write(f'execute if score i {pack} < window_len {pack} run function {pack}:{norm_funcname}')

                    with self.write_function(norm_funcname) as f1:
                        # av[i] /= expsum
                        f1.write(f'execute store result storage {pack} args.i int 1 run scoreboard players get i {pack}')
                        f1.write(f'function {pack}:get_av with storage {pack} args')

                        f1.write(f'scoreboard players operation yb {pack} = expsumb {pack}')
                        f1.write(f'scoreboard players operation ye {pack} = expsume {pack}')
                        f1.write(f'scoreboard players operation ys {pack} = expsums {pack}')

                        f1.write(f'function {pack}:div')
                        f1.write(f'function {pack}:set_av with storage {pack} args')

                        # i += 1, loop back
                        f1.write(f'scoreboard players add i {pack} 1')
                        f1.write(f'execute if score i {pack} < window_len {pack} run function {pack}:{norm_funcname}')

                    tx_ptr = h * head_size
                    # Weighted sum of the values, store back into tx
                    for i in range(tx_ptr, tx_ptr + head_size):
                        f.write(f'scoreboard players set {i}b tx 0')
                        f.write(f'scoreboard players set {i}e tx 0')
                        f.write(f'scoreboard players set {i}s tx 0')
                    
                    # for t in range(start, pos + 1):
                    val_funcname = f'attention_l{l}_h{h}_weighted_sum_vals'
                    f.write(f'scoreboard players operation t {pack} = start {pack}')
                    f.write(f'scoreboard players set i {pack} 0')
                    f.write(f'execute if score t {pack} <= pos {pack} run function {pack}:{val_funcname}')

                    with self.write_function(val_funcname) as f1:
                        # cache_idx = t % seq_len * dim + loff + kv_head * head_size
                        f1.write(f'scoreboard players operation cache_idx {pack} = t {pack}')
                        f1.write(f'scoreboard players operation cache_idx {pack} %= {seq_len} {pack}')
                        f1.write(f'scoreboard players operation cache_idx {pack} *= {dim} {pack}')
                        offset = loff + kv_head * head_size
                        if offset != 0:
                            f1.write(f'scoreboard players add cache_idx {pack} {offset}')

                        # a = av[i]
                        f1.write(f'execute store result storage {pack} args.i int 1 run scoreboard players get i {pack}')
                        f1.write(f'function {pack}:get_av with storage {pack} args')

                        f1.write(f'scoreboard players operation ab {pack} = xb {pack}')
                        f1.write(f'scoreboard players operation ae {pack} = xe {pack}')
                        f1.write(f'scoreboard players operation as {pack} = xs {pack}')

                        for i in range(head_size):
                            # xb[tx_ptr + i] += av[t - start] * vc[cache_idx + i]

                            # Get the attention weight for this timestep
                            # Get the value vector for this head and at this timestep
                            
                            f1.write(f'scoreboard players operation yb {pack} = ab {pack}')
                            f1.write(f'scoreboard players operation ye {pack} = ae {pack}')
                            f1.write(f'scoreboard players operation ys {pack} = as {pack}')
                            f1.write(f'execute store result storage {pack} args.i int 1 run scoreboard players get cache_idx {pack}')
                            f1.write(f'function {pack}:get_vc with storage {pack} args')
                            f1.write(f'function {pack}:mul')

                            f1.write(f'scoreboard players add cache_idx {pack} 1')

                            # Accumulate the weighted value into tx

                            f1.write(f'scoreboard players operation yb {pack} = {tx_ptr + i}b tx')
                            f1.write(f'scoreboard players operation ye {pack} = {tx_ptr + i}e tx')
                            f1.write(f'scoreboard players operation ys {pack} = {tx_ptr + i}s tx')
                            
                            f1.write(f'function {pack}:add')

                            f1.write(f'scoreboard players operation {tx_ptr + i}b tx = xb {pack}')
                            f1.write(f'scoreboard players operation {tx_ptr + i}e tx = xe {pack}')
                            f1.write(f'scoreboard players operation {tx_ptr + i}s tx = xs {pack}')

                        # t += 1, i += 1, loop back
                        f1.write(f'scoreboard players add t {pack} 1')
                        f1.write(f'scoreboard players add i {pack} 1')
                        f1.write(f'execute if score t {pack} <= pos {pack} run function {pack}:{val_funcname}')
                
                f.write(f'bossbar set progress value {l * 8 + 4}')
                f.split(1)

                # Final matrix multiplication to get the output of the attention
                f.write(f'data modify storage {pack} args.in set value "tx"')
                f.write(f'data modify storage {pack} args.out set value "tx2"')
                f.write(f'data modify storage {pack} args.m set value "wo_{l}"')
                f.write(f'execute store result score s1 {pack} run scoreboard players set s2 {pack} {dim}')
                f.write(f'function {pack}:matmul')

                f.write(f'bossbar set progress value {l * 8 + 5}')
                f.split(1)

                # Residual connection back into x
                for i in range(dim):
                    # x[i] += tx2[i]
                    f.write(f'scoreboard players operation xb {pack} = {i}b tx2')
                    f.write(f'scoreboard players operation xe {pack} = {i}e tx2')
                    f.write(f'scoreboard players operation xs {pack} = {i}s tx2')

                    f.write(f'scoreboard players operation yb {pack} = {i}b x')
                    f.write(f'scoreboard players operation ye {pack} = {i}e x')
                    f.write(f'scoreboard players operation ys {pack} = {i}s x')

                    f.write(f'function {pack}:add')
                    
                    f.write(f'scoreboard players operation {i}b x = xb {pack}')
                    f.write(f'scoreboard players operation {i}e x = xe {pack}')
                    f.write(f'scoreboard players operation {i}s x = xs {pack}')

                # FFN rmsnorm
                with self._rmsnorm_context(f):
                    for j in range(dim):
                        # tx[j] = rms_ffn_weight[l * dim + j] * ss * x[j]
                        idx = l * dim + j
                        f.write(f'execute store result score {j}b tx run data get storage {pack} params.ffn_w.{idx}b')
                        f.write(f'execute store result score {j}e tx run data get storage {pack} params.ffn_w.{idx}e')
                        f.write(f'execute store result score {j}s tx run data get storage {pack} params.ffn_w.{idx}s')

                # Calculate w1(x) and w3(x) for FFN
                f.write(f'data modify storage {pack} args.in set value "tx"')
                f.write(f'data modify storage {pack} args.out set value "th"')
                f.write(f'data modify storage {pack} args.m set value "w1_{l}"')
                f.write(f'scoreboard players set s1 {pack} {hidden_dim}')
                f.write(f'scoreboard players set s2 {pack} {dim}')
                f.write(f'function {pack}:matmul')
                
                f.write(f'bossbar set progress value {l * 8 + 6}')
                f.split(1)

                f.write(f'data modify storage {pack} args.in set value "tx"')
                f.write(f'data modify storage {pack} args.out set value "th2"')
                f.write(f'data modify storage {pack} args.m set value "w3_{l}"')
                f.write(f'scoreboard players set s1 {pack} {hidden_dim}')
                f.write(f'scoreboard players set s2 {pack} {dim}')
                f.write(f'function {pack}:matmul')
                
                f.write(f'bossbar set progress value {l * 8 + 7}')
                f.split(1)
                
                f.write(f'function {pack}:silu_activation')

                # Final matrix multiplication to get the output of the FFN
                f.write(f'data modify storage {pack} args.in set value "th"')
                f.write(f'data modify storage {pack} args.out set value "tx"')
                f.write(f'data modify storage {pack} args.m set value "w2_{l}"')
                f.write(f'scoreboard players set s1 {pack} {dim}')
                f.write(f'scoreboard players set s2 {pack} {hidden_dim}')
                f.write(f'function {pack}:matmul')

                f.write(f'bossbar set progress value {l * 8 + 8}')
                f.split(1)

                # Residual connection
                for i in range(dim):
                    # x[i] += tx[i]
                    f.write(f'scoreboard players operation xb {pack} = {i}b tx')
                    f.write(f'scoreboard players operation xe {pack} = {i}e tx')
                    f.write(f'scoreboard players operation xs {pack} = {i}s tx')

                    f.write(f'scoreboard players operation yb {pack} = {i}b x')
                    f.write(f'scoreboard players operation ye {pack} = {i}e x')
                    f.write(f'scoreboard players operation ys {pack} = {i}s x')

                    f.write(f'function {pack}:add')
                    
                    f.write(f'scoreboard players operation {i}b x = xb {pack}')
                    f.write(f'scoreboard players operation {i}e x = xe {pack}')
                    f.write(f'scoreboard players operation {i}s x = xs {pack}')

    def _generate_encode(self):
        pack = self.pack
        vocab = self.model.vocab

        with self.write_function('tokenize_prompt') as f:
            # Byte-pair encoding

            whitespace_tok = len(vocab) - vocab[::-1].index(" ") - 1
            # Set the first token to a whitespace
            f.write(f'data modify storage {pack} args.prompt_tokens set value [{whitespace_tok}]')

            # Enumerate every characters in the prompt string
            f.write(f'scoreboard players set i {pack} 0')
            f.write(f'scoreboard players set j {pack} 1')
            f.write(f'execute store result score len {pack} run data get storage {pack} args.prompt')
            f.write(f'function {pack}:tokenize_characters {{i:0,j:1}}')

            with self.write_function('tokenize_characters') as f1:
                # c = prompt[i]
                f1.write(f'$data modify storage {pack} args.c set string storage {pack} args.prompt $(i) $(j)')

                # tok = string_to_token(c)
                f1.write(f'function {pack}:string_to_token')
                f1.write(f'execute if score tok {pack} matches -1 run return run tellraw @a [{{"text":"Not a good prompt at position ","color":"red"}},{{"score":{{"name":"i","objective":"{pack}"}}}}]')

                # tokens[i] = tok
                f1.write(f'data modify storage {pack} args.prompt_tokens append value 0')
                f1.write(f'execute store result storage {pack} args.prompt_tokens[-1] int 1 run scoreboard players get tok {pack}')

                f1.write(f'execute store result storage {pack} args.i int 1 run scoreboard players add i {pack} 1')
                f1.write(f'execute store result storage {pack} args.j int 1 run scoreboard players add j {pack} 1')
                f1.write(f'execute if score i {pack} < len {pack} run function {pack}:tokenize_characters with storage {pack} args')
            
            # while True
            f.write(f'function {pack}:token_merge_loop')

            with self.write_function('token_merge_loop') as f1:
                # best_score = -inf
                f1.write(f'scoreboard players set bs {pack} -2147483648')
                # best_token = -1
                f1.write(f'scoreboard players set bt {pack} -1')
                # best_idx = -1
                f1.write(f'scoreboard players set bi {pack} -1')

                f1.write(f'scoreboard players set i {pack} 0')
                f1.write(f'scoreboard players set j {pack} 1')
                f1.write(f'execute store result score len {pack} run data get storage {pack} args.prompt_tokens')
                f1.write(f'data merge storage {pack} {{args:{{i:0,j:1}}}}')
                f1.write(f'function {pack}:fetch_token_pair with storage {pack} args')
                f1.write(f'function {pack}:try_merge_pair with storage {pack} args')

                with self.write_function('try_merge_pair') as f2:
                    # tok = string_to_token(tokens[i] + tokens[i + 1])
                    f2.write(f'$data modify storage {pack} args.c set value "$(tok0)$(tok1)"')
                    f2.write(f'function {pack}:string_to_token with storage {pack} args')

                    # if tok != -1: record_best_merge()
                    f2.write(f'execute unless score tok {pack} matches -1 run function {pack}:record_best_merge')

                    # loop back
                    f2.write(f'execute store result storage {pack} args.i int 1 run scoreboard players add i {pack} 1')
                    f2.write(f'execute store result storage {pack} args.j int 1 run scoreboard players add j {pack} 1')
                    f2.write(f'function {pack}:fetch_token_pair with storage {pack} args')
                    f2.write(f'execute if score i {pack} < len {pack} run function {pack}:try_merge_pair with storage {pack} args')

                with self.write_function('fetch_token_pair') as f2:
                    f2.write(f'$execute store result score tok {pack} run data get storage {pack} args.prompt_tokens[$(i)]')
                    f2.write(f'function {pack}:token_to_string')
                    f2.write(f'data modify storage {pack} args.tok0 set from storage {pack} args.tok')
                    f2.write(f'$execute store result score tok {pack} run data get storage {pack} args.prompt_tokens[$(j)]')
                    f2.write(f'function {pack}:token_to_string')
                    f2.write(f'data modify storage {pack} args.tok1 set from storage {pack} args.tok')

                with self.write_function('record_best_merge') as f2:
                    # score = vocab_scores[tok]
                    f2.write(f'function {pack}:lookup_token_score')
                    # if best_score > score: return
                    f2.write(f'execute if score bs {pack} >= score {pack} run return 0')

                    f2.write(f'scoreboard players operation bs {pack} = score {pack}')
                    f2.write(f'scoreboard players operation bt {pack} = tok {pack}')
                    f2.write(f'scoreboard players operation bi {pack} = i {pack}')
                
                # We couldn't find any more pairs to merge, so we're done
                f1.write(f'execute if score bi {pack} matches -1 run return 0')

                # tokens[best_idx] = best_tok; del tokens[best_idx + 1]
                f1.write(f'execute store result storage {pack} args.i int 1 run scoreboard players get bi {pack}')
                f1.write(f'execute store result storage {pack} args.j int 1 run scoreboard players add bi {pack} 1')
                f1.write(f'scoreboard players remove bi {pack} 1')
                f1.write(f'function {pack}:apply_token_merge with storage {pack} args')

                with self.write_function('apply_token_merge') as f2:
                    f2.write(f'$execute store result storage {pack} args.prompt_tokens[$(i)] int 1 run scoreboard players get bt {pack}')
                    f2.write(f'$data remove storage {pack} args.prompt_tokens[$(j)]')
                
                # loop back
                f1.write(f'function {pack}:token_merge_loop')

        with self.write_function('lookup_token_score') as f:
            for i, score in enumerate(self.model.vocab_scores):
                f.write(f'execute if score tok {pack} matches {i} run return run scoreboard players set score {pack} {score}')

        with self.write_function('string_to_token') as f:
            # Find the last perfect match for string in vocab, return its index or -1 if not found
            for i, tok in enumerate(vocab[::-1]):
                rtok = repr(tok)
                if r"\x" in rtok or r"\r" in rtok or r"\u" in rtok:
                    continue
                
                f.write(f'execute if data storage {pack} args{{c:{tok!r}}} run return run scoreboard players set tok {pack} {len(vocab) - i - 1}')
            f.write(f'scoreboard players set tok {pack} -1')

    def _generate_argmax(self):
        pack = self.pack
        vocab_size = self.model.vocab_size

        with self.write_function('argmax') as f:
            # y = -inf
            f.write(f'scoreboard players set yb {pack} 99999999')
            f.write(f'scoreboard players set ye {pack} 99999999')
            f.write(f'scoreboard players set ys {pack} -1')

            # update_best_token = () => { y = x; tok = i; }
            with self.write_function('update_best_token') as f1:
                f1.write(f'scoreboard players operation yb {pack} = xb {pack}')
                f1.write(f'scoreboard players operation ye {pack} = xe {pack}')
                f1.write(f'scoreboard players operation ys {pack} = xs {pack}')
                f1.write(f'scoreboard players operation tok {pack} = i {pack}')

            # Setup progress bar
            f.write(f'bossbar set progress max {vocab_size // 100 + 1}')
            f.write(f'bossbar set progress value 0')
            f.write(f'bossbar set progress name [{{"text":"Sampling","color":"gray","bold":true}}]')
            f.write(f'bossbar set progress players @a')

            f.write(f'scoreboard players set i {pack} 0')
            f.write(f'function {pack}:argmax_scan {{i:0}}')
            
            with self.write_function('argmax_scan') as f1:
                # x = logits[i]
                f1.write(f'$scoreboard players operation xb {pack} = $(i)b logits')
                f1.write(f'$scoreboard players operation xe {pack} = $(i)e logits')
                f1.write(f'$scoreboard players operation xs {pack} = $(i)s logits')
                
                # if x > y: update_best_token()
                f1.write(f'function {pack}:cmp')
                f1.write(f'execute if score r {pack} matches 1 run function {pack}:update_best_token')

                # i += 1; Update progress bar; loop back
                f1.write(f'execute store result score t0 {pack} store result score t1 {pack} store result storage {pack} args.i int 1 run scoreboard players add i {pack} 1')
                f1.write(f'scoreboard players operation t0 {pack} %= 100 {pack}')
                f1.write(f'execute if score t0 {pack} matches 99 store result bossbar progress value run scoreboard players operation t1 {pack} /= 100 {pack}')

                f1.write(f'execute if score i {pack} matches ..{vocab_size - 1} run function {pack}:argmax_scan with storage {pack} args')

    def _generate_temperature_sampling(self):
        pack = self.pack
        vocab = self.model.vocab
        vocab_size = self.model.vocab_size

        with self.write_function('temperature_sample') as f:
            # y = t (temperature)
            f.write(f'scoreboard players operation yb {pack} = tb {pack}')
            f.write(f'scoreboard players operation ye {pack} = te {pack}')
            f.write(f'scoreboard players operation ys {pack} = ts {pack}')
            
            # Setup progress bar
            f.write(f'bossbar set progress max {5 * vocab_size // 100 + 1}')
            f.write(f'bossbar set progress value 0')
            f.write(f'bossbar set progress name [{{"text":"Sampling","color":"gray","bold":true}}]')
            f.write(f'bossbar set progress players @a')

            # for i in range(vocab_size):
            f.write(f'scoreboard players set i {pack} 0')
            f.write(f'function {pack}:apply_temperature_scale {{i:0}}')

            with self.write_function('apply_temperature_scale') as f1:
                # logits[i] /= y
                f1.write(f'$scoreboard players operation xb {pack} = $(i)b logits')
                f1.write(f'$scoreboard players operation xe {pack} = $(i)e logits')
                f1.write(f'$scoreboard players operation xs {pack} = $(i)s logits')
                f1.write(f'function {pack}:div')
                f1.write(f'$scoreboard players operation $(i)b logits = xb {pack}')
                f1.write(f'$scoreboard players operation $(i)e logits = xe {pack}')
                f1.write(f'$scoreboard players operation $(i)s logits = xs {pack}')

                # i += 1; Update progress bar; loop back
                f1.write(f'execute store result score t0 {pack} store result score t1 {pack} store result storage {pack} args.i int 1 run scoreboard players add i {pack} 1')
                f1.write(f'scoreboard players operation t0 {pack} %= 100 {pack}')
                f1.write(f'execute if score t0 {pack} matches 99 store result bossbar progress value run scoreboard players operation t1 {pack} /= 100 {pack}')
                f1.write(f'execute if score i {pack} matches ..{vocab_size - 1} run function {pack}:apply_temperature_scale with storage {pack} args')

            # Apply softmax to the logits

            # y = -inf
            f.write(f'scoreboard players set yb {pack} 99999999')
            f.write(f'scoreboard players set ye {pack} 99999999')
            f.write(f'scoreboard players set ys {pack} -1')

            # setmax = () => { y = x; tok = i; }
            with self.write_function('update_max_logit') as f1:
                f1.write(f'scoreboard players operation yb {pack} = xb {pack}')
                f1.write(f'scoreboard players operation ye {pack} = xe {pack}')
                f1.write(f'scoreboard players operation ys {pack} = xs {pack}')

            # for i in range(vocab_size):
            f.write(f'scoreboard players set i {pack} 0')
            f.write(f'function {pack}:find_max_logit {{i:0}}')

            with self.write_function('find_max_logit') as f1:
                # x = logits[i]
                f1.write(f'$scoreboard players operation xb {pack} = $(i)b logits')
                f1.write(f'$scoreboard players operation xe {pack} = $(i)e logits')
                f1.write(f'$scoreboard players operation xs {pack} = $(i)s logits')
                
                # if x > y: setmax()
                f1.write(f'function {pack}:cmp')
                f1.write(f'execute if score r {pack} matches 1 run function {pack}:update_max_logit')
                
                # i += 1; Update progress bar; loop back
                f1.write(f'execute store result score t0 {pack} store result score t1 {pack} store result storage {pack} args.i int 1 run scoreboard players add i {pack} 1')
                f1.write(f'scoreboard players operation t0 {pack} %= 100 {pack}')
                f1.write(f'scoreboard players add t1 {pack} {vocab_size}')
                f1.write(f'execute if score t0 {pack} matches 99 store result bossbar progress value run scoreboard players operation t1 {pack} /= 100 {pack}')
                f1.write(f'execute if score i {pack} matches ..{vocab_size - 1} run function {pack}:find_max_logit with storage {pack} args')
            
            # expsum = 0.0
            f.write(f'scoreboard players set expsumb {pack} 0')
            f.write(f'scoreboard players set expsume {pack} 0')
            f.write(f'scoreboard players set expsums {pack} 0')

            # nmax = -y
            f.write(f'scoreboard players operation nmb {pack} = yb {pack}')
            f.write(f'scoreboard players operation nme {pack} = ye {pack}')
            f.write(f'scoreboard players operation nms {pack} = ys {pack}')
            f.write(f'scoreboard players operation nms {pack} *= -1 {pack}')

            # for i in range(vocab_size):
            f.write(f'scoreboard players set i {pack} 0')
            f.write(f'function {pack}:compute_exp_sum {{i:0}}')

            with self.write_function('compute_exp_sum') as f1:
                # logits[i] = exp(logits[i] + nmax)
                f1.write(f'$scoreboard players operation xb {pack} = $(i)b logits')
                f1.write(f'$scoreboard players operation xe {pack} = $(i)e logits')
                f1.write(f'$scoreboard players operation xs {pack} = $(i)s logits')

                f1.write(f'scoreboard players operation yb {pack} = nmb {pack}')
                f1.write(f'scoreboard players operation ye {pack} = nme {pack}')
                f1.write(f'scoreboard players operation ys {pack} = nms {pack}')

                f1.write(f'function {pack}:add')
                f1.write(f'function {pack}:exp')

                f1.write(f'$scoreboard players operation $(i)b logits = xb {pack}')
                f1.write(f'$scoreboard players operation $(i)e logits = xe {pack}')
                f1.write(f'$scoreboard players operation $(i)s logits = xs {pack}')

                # expsum += av[i]
                f1.write(f'scoreboard players operation yb {pack} = expsumb {pack}')
                f1.write(f'scoreboard players operation ye {pack} = expsume {pack}')
                f1.write(f'scoreboard players operation ys {pack} = expsums {pack}')

                f1.write(f'function {pack}:add')

                f1.write(f'scoreboard players operation expsumb {pack} = xb {pack}')
                f1.write(f'scoreboard players operation expsume {pack} = xe {pack}')
                f1.write(f'scoreboard players operation expsums {pack} = xs {pack}')

                # i += 1; Update progress bar; loop back
                f1.write(f'execute store result score t0 {pack} store result score t1 {pack} store result storage {pack} args.i int 1 run scoreboard players add i {pack} 1')
                f1.write(f'scoreboard players operation t0 {pack} %= 100 {pack}')
                f1.write(f'scoreboard players add t1 {pack} {2 * vocab_size}')
                f1.write(f'execute if score t0 {pack} matches 99 store result bossbar progress value run scoreboard players operation t1 {pack} /= 100 {pack}')
                f1.write(f'execute if score i {pack} matches ..{vocab_size - 1} run function {pack}:compute_exp_sum with storage {pack} args')

            # Normalize

            # for i in range(vocab_size):
            f.write(f'scoreboard players set i {pack} 0')
            f.write(f'function {pack}:normalize_probs {{i:0}}')

            with self.write_function('normalize_probs') as f1:
                # logits[i] /= expsum
                f1.write(f'$scoreboard players operation xb {pack} = $(i)b logits')
                f1.write(f'$scoreboard players operation xe {pack} = $(i)e logits')
                f1.write(f'$scoreboard players operation xs {pack} = $(i)s logits')

                f1.write(f'scoreboard players operation yb {pack} = expsumb {pack}')
                f1.write(f'scoreboard players operation ye {pack} = expsume {pack}')
                f1.write(f'scoreboard players operation ys {pack} = expsums {pack}')

                f1.write(f'function {pack}:div')

                f1.write(f'$scoreboard players operation $(i)b logits = xb {pack}')
                f1.write(f'$scoreboard players operation $(i)e logits = xe {pack}')
                f1.write(f'$scoreboard players operation $(i)s logits = xs {pack}')

                # i += 1; Update progress bar; loop back
                f1.write(f'execute store result score t0 {pack} store result score t1 {pack} store result storage {pack} args.i int 1 run scoreboard players add i {pack} 1')
                f1.write(f'scoreboard players operation t0 {pack} %= 100 {pack}')
                f1.write(f'scoreboard players add t1 {pack} {3 * vocab_size}')
                f1.write(f'execute if score t0 {pack} matches 99 store result bossbar progress value run scoreboard players operation t1 {pack} /= 100 {pack}')
                f1.write(f'execute if score i {pack} matches ..{vocab_size - 1} run function {pack}:normalize_probs with storage {pack} args')

            # Sample from this distribution to get the next token

            # r = random(0, 1)
            f.write(f'execute store result score xb {pack} run random value 0..99999999')
            f.write(f'scoreboard players set xe {pack} -1')
            f.write(f'scoreboard players set xs {pack} 1')
            f.write(f'function {pack}:float_5')
            f.write(f'execute if score xb {pack} matches 100000000.. run function {pack}:float_15')
            f.write(f'scoreboard players operation rb {pack} = xb {pack}')
            f.write(f'scoreboard players operation re {pack} = xe {pack}')
            f.write(f'scoreboard players operation rs {pack} = xs {pack}')

            # cdf = 0.0
            f.write(f'scoreboard players set cdfb {pack} 0')
            f.write(f'scoreboard players set cdfe {pack} 0')
            f.write(f'scoreboard players set cdfs {pack} 0')

            # for i in range(vocab_size):
            f.write(f'scoreboard players set i {pack} 0')
            f.write(f'function {pack}:sample_from_cdf {{i:0}}')

            with self.write_function('sample_from_cdf') as f1:
                # cdf += logits[i]; if (cdf > r) return (tok = i)
                f1.write(f'$scoreboard players operation xb {pack} = $(i)b logits')
                f1.write(f'$scoreboard players operation xe {pack} = $(i)e logits')
                f1.write(f'$scoreboard players operation xs {pack} = $(i)s logits')

                f1.write(f'scoreboard players operation yb {pack} = cdfb {pack}')
                f1.write(f'scoreboard players operation ye {pack} = cdfe {pack}')
                f1.write(f'scoreboard players operation ys {pack} = cdfs {pack}')

                f1.write(f'function {pack}:add')

                f1.write(f'scoreboard players operation cdfb {pack} = xb {pack}')
                f1.write(f'scoreboard players operation cdfe {pack} = xe {pack}')
                f1.write(f'scoreboard players operation cdfs {pack} = xs {pack}')

                f1.write(f'scoreboard players operation yb {pack} = rb {pack}')
                f1.write(f'scoreboard players operation ye {pack} = re {pack}')
                f1.write(f'scoreboard players operation ys {pack} = rs {pack}')

                f1.write(f'function {pack}:cmp')
                f1.write(f'execute if score r {pack} matches 1 run return run scoreboard players operation tok {pack} = i {pack}')

                # i += 1; Update progress bar; loop back
                f1.write(f'execute store result score t0 {pack} store result score t1 {pack} store result storage {pack} args.i int 1 run scoreboard players add i {pack} 1')
                f1.write(f'scoreboard players operation t0 {pack} %= 100 {pack}')
                f1.write(f'scoreboard players add t1 {pack} {4 * vocab_size}')
                f1.write(f'execute if score t0 {pack} matches 99 store result bossbar progress value run scoreboard players operation t1 {pack} /= 100 {pack}')
                f1.write(f'execute if score i {pack} matches ..{vocab_size - 1} run function {pack}:sample_from_cdf with storage {pack} args')

            # In case of rounding errors
            f.write(f'execute if score i {pack} matches {vocab_size} run scoreboard players set tok {pack} {vocab_size - 1}')

        with self.write_function('token_to_string') as f:
            for i, tok in enumerate(vocab):
                rtok = repr(tok)
                if r"\x" in rtok or r"\r" in rtok or r"\u" in rtok:
                    continue

                f.write(f'execute if score tok {pack} matches {i} run return run data modify storage {pack} args.tok set value {tok!r}')

    # TODO: Implement top_k / top_p sampling

    def _generate_text_generation(self):
        pack = self.pack
        n_layers = self.model.n_layers
        vocab_size = self.model.vocab_size
        dim = self.model.dim

        with self.write_function('generate') as f:
            # Get command prompt arguments
            # /function {pack}:generate {s:<steps>,t:<temperature>,i:<prompt>}
            f.write(f'$scoreboard players set steps {pack} $(s)')
            f.write(f'$scoreboard players set temperature {pack} $(t)')

            # if steps < 0: error
            f.write(f'execute if score steps {pack} matches ..0 run return run tellraw @a {{"text":"Steps must be a positive integer","color":"red"}}')

            # if not 0 <= temperature <= 100: error
            f.write(f'execute if score temperature {pack} matches ..-1 run return run tellraw @a {{"text":"Temperature must be an integer between 0 and 100 inclusive","color":"red"}}')
            f.write(f'execute if score temperature {pack} matches 101.. run return run tellraw @a {{"text":"Temperature must be an integer between 0 and 100 inclusive","color":"red"}}')

            # Clear the token buffer
            f.write(f'data modify storage {pack} args.prompt_tokens set value []')

            # If there is a prompt, tokenize it
            f.write(f'$data modify storage {pack} args.prompt set value "$(i)"')
            f.write(f'execute unless data storage {pack} args{{prompt:""}} run function {pack}:tokenize_prompt')

            # Initiallize
            f.write(f'function {pack}:setup')
            f.write(f'scoreboard players set tok {pack} 1')  # BOS token
            f.write(f'scoreboard players set pos {pack} 0')

            # t = float(temperature / 50)
            f.write(f'scoreboard players operation xb {pack} = temperature {pack}')
            f.write(f'scoreboard players operation xb {pack} *= 2 {pack}')
            f.write(f'scoreboard players set xe {pack} 5')
            f.write(f'scoreboard players set xs {pack} 1')
            f.write(f'function {pack}:float_5')
            f.write(f'execute if score xb {pack} matches 100000000.. run function {pack}:float_15')
            f.write(f'scoreboard players operation tb {pack} = xb {pack}')
            f.write(f'scoreboard players operation te {pack} = xe {pack}')
            f.write(f'scoreboard players operation ts {pack} = xs {pack}')

            # Start the main loop
            f.write(f'function {pack}:autoregressive')

        with self.write_function('autoregressive') as f:
            # Forward the transformer to get logits for the next token
            f.write(f'function {pack}:run_forward_pass')
            # Make sure the next inference is executed after all the scheduled ticks in {pack}:forward
            f.write(f'execute if data storage {pack} args{{prompt_tokens:[]}} run schedule function {pack}:autoregressive_next {8 * n_layers + 4 + vocab_size // dim}t')
            f.write(f'execute unless data storage {pack} args{{prompt_tokens:[]}} run schedule function {pack}:autoregressive_next {8 * n_layers + 1}t')
            
        with self.write_function('run_forward_pass') as f:
            f.write(f'execute unless data storage {pack} args{{prompt_tokens:[]}} run return run function {pack}:run_forward_prompt')
            # Setup the progress bar
            # There are 8 updates in each layer, and `vocab_size // dim` updates in the final matrix multiplication
            f.write(f'bossbar set progress max {n_layers * 8 + vocab_size // dim}')
            f.write(f'bossbar set progress value 0')
            f.write(f'bossbar set progress name [{{"text":"Inferring, Step #","color":"green","bold":true}},{{"score":{{"name":"pos","objective":"{pack}"}}}}]')
            f.write(f'bossbar set progress players @a')
            f.write(f'function {pack}:forward')

        with self.write_function('run_forward_prompt') as f:
            # Setup the progress bar
            # There are 8 updates in each layer
            f.write(f'bossbar set progress max {n_layers * 8}')
            f.write(f'bossbar set progress value 0')
            f.write(f'bossbar set progress name [{{"text":"Processing prompt token #","color":"gold","bold":true}},{{"score":{{"name":"pos","objective":"{pack}"}}}}]')
            f.write(f'bossbar set progress players @a')
            f.write(f'function {pack}:forward_hidden')

        clear_screen = "\\n" * 100

        with self.write_function('autoregressive_next') as f:
            # If we are still processing the input prompt, force the next prompt token
            f.write(f'execute unless data storage {pack} args{{prompt_tokens:[]}} run return run function {pack}:next_prompt_token')
            
            next_forward_cmds = [
                f'bossbar set progress players',  # Clear progress bar
                f'function {pack}:token_to_string',  # Get token string, store into `storage {pack} args.tok`
                f'data modify storage {pack} args.output append from storage {pack} args.tok',  # Append to output buffer
                f'tellraw @a ["{clear_screen}",{{"storage":"{pack}","nbt":"args.output[]","separator":""}}]',  # Print the output
                f'scoreboard players add pos {pack} 1',  # Increment position
                f'execute if score pos {pack} = steps {pack} run return 1',
                f'function {pack}:autoregressive',
            ]

            with self.write_function('next_prompt_token') as f1:
                f1.write(f'execute store result score tok {pack} run data get storage {pack} args.prompt_tokens[0]')
                f1.write(f'data remove storage {pack} args.prompt_tokens[0]')

                for cmd in next_forward_cmds:
                    f1.write(cmd)

            # Otherwise, sample the next token
            f.write(f'execute if score temperature {pack} matches 0 run function {pack}:argmax')
            f.write(f'execute unless score temperature {pack} matches 0 run function {pack}:temperature_sample')

            for cmd in next_forward_cmds:
                f.write(cmd)

    def _generate_resume_stop(self):
        pack = self.pack
        with self.write_function('resume') as f:
            f.write(f'$scoreboard players set steps {pack} $(s)')
            f.write(f'execute if score steps {pack} matches ..0 run return run tellraw @a {{"text":"Steps must be a positive integer!","color":"red"}}')
            f.write(f'tellraw @a [{{"text":"Resuming from step #","color":"green"}},{{"score":{{"name":"pos","objective":"{pack}"}}}}]')
            f.write(f'scoreboard players operation steps {pack} += pos {pack}')
            f.write(f'function {pack}:autoregressive')

        # Terminate the generation loop
        with self.write_function('stop') as f:
            f.write(f'bossbar set progress players')
            f.write(f'schedule clear {pack}:forward_0')
            for i in range(0, 8 * self.model.n_layers):
                f.write(f'schedule clear {pack}:forward_hidden_{i}')
            f.write(f'schedule clear {pack}:chat_process_input_0')
            f.write(f'schedule clear {pack}:respond_0')
            f.write(f'schedule clear {pack}:autoregressive_next')
            f.write(f'schedule clear {pack}:matmul_logits_resume')
            f.write(f'tellraw @a {{"text":"Stopped inferring","color":"red"}}')

    def _generate_chat(self):
        # Chat function
        pack = self.pack
        dim = self.model.dim
        n_layers = self.model.n_layers
        vocab_size = self.model.vocab_size

        with self.write_function('chat') as f:
            # Get command prompt arguments
            # /function {pack}:chat {t:<temperature>,i:<message>}
            f.write(f'$scoreboard players set temperature {pack} $(t)')

            # if not 0 <= temperature <= 100: error
            temperature_errmsg = '{"text":"Temperature must be an integer between 0 and 100 inclusive","color":"red"}'
            f.write(f'execute if score temperature {pack} matches ..-1 run return run tellraw @a {temperature_errmsg}')
            f.write(f'execute if score temperature {pack} matches 101.. run return run tellraw @a {temperature_errmsg}')

            # Make sure that the message is not empty
            f.write(f'$data modify storage {pack} args.prompt set value "$(i)"')
            f.write(f'execute if data storage {pack} args{{prompt:""}} run return run tellraw @a {{"text":"Message must be non-empty","color":"red"}}')
            # Render user/system prompts into the Llama 2 Chat schema
            f.write(f'$data modify storage {pack} args.prompt set value "[INST] $(i) [/INST]"')

            # Clear the token buffer
            clear_screen = "\\n" * 100

            f.write(f'$data modify storage {pack} args.output append value "\\nUser: $(i)\\nAssistant:"')
            f.write(f'tellraw @a ["{clear_screen}",{{"storage":"{pack}","nbt":"args.output[]","separator":""}}]')
            f.write(f'data modify storage {pack} args.prompt_tokens set value []')

            # Tokenize the rendered message
            f.write(f'function {pack}:tokenize_prompt')
            # Add EOS(1) token at the beginning
            f.write(f'data modify storage {pack} args.prompt_tokens prepend value 1')
            # Pop the final token into final_tok
            f.write(f'execute store result score final_tok {pack} run data get storage {pack} args.prompt_tokens[-1]')
            f.write(f'data remove storage {pack} args.prompt_tokens[0]')

            # t = float(temperature / 50)
            f.write(f'scoreboard players operation xb {pack} = temperature {pack}')
            f.write(f'scoreboard players operation xb {pack} *= 2 {pack}')
            f.write(f'scoreboard players set xe {pack} 5')
            f.write(f'scoreboard players set xs {pack} 1')
            f.write(f'function {pack}:float_5')
            f.write(f'execute if score xb {pack} matches 100000000.. run function {pack}:float_15')
            f.write(f'scoreboard players operation tb {pack} = xb {pack}')
            f.write(f'scoreboard players operation te {pack} = xe {pack}')
            f.write(f'scoreboard players operation ts {pack} = xs {pack}')

            # Process the user input
            f.write(f'scoreboard players set pos {pack} 0')
            f.write(f'execute store result score len {pack} run data get storage {pack} args.prompt_tokens')
            f.write(f'function {pack}:chat_process_input')

        with self.write_function('chat_process_input') as f:
            f.write(f'execute store result score tok {pack} run data get storage {pack} args.prompt_tokens[0]')
            f.write(f'data remove storage {pack} args.prompt_tokens[0]')
            # Setup the progress bar
            # There are 8 updates in each layer
            f.write(f'bossbar set progress max {n_layers * 8}')
            f.write(f'bossbar set progress value 0')
            f.write(f'bossbar set progress name [{{"text":"Processing prompt token #","color":"gold","bold":true}},{{"score":{{"name":"pos","objective":"{pack}"}}}}]')
            f.write(f'bossbar set progress players @a')
            f.write(f'function {pack}:forward_hidden')
            f.split(8 * n_layers + 1)
            f.write(f'scoreboard players add pos {pack} 1')

            f.write(f'execute if score pos {pack} = len {pack} run scoreboard players operation tok {pack} = final_tok {pack}')  # Restore to final_tok
            f.write(f'execute if score pos {pack} = len {pack} run return run schedule function {pack}:respond 1t')

            f.write(f'function {pack}:chat_process_input')

        with self.write_function('respond') as f:
            # Setup the progress bar
            # There are 8 updates in each layer, and `vocab_size // dim` updates in the final matrix multiplication
            f.write(f'bossbar set progress max {n_layers * 8 + vocab_size // dim}')
            f.write(f'bossbar set progress value 0')
            f.write(f'bossbar set progress name [{{"text":"Inferring, Step #","color":"green","bold":true}},{{"score":{{"name":"pos","objective":"{pack}"}}}}]')
            f.write(f'bossbar set progress players @a')
            f.write(f'function {pack}:forward')
            
            f.split(8 * n_layers + 4 + vocab_size // dim)

            f.write(f'execute if score temperature {pack} matches 0 run function {pack}:argmax')
            f.write(f'execute unless score temperature {pack} matches 0 run function {pack}:temperature_sample')
            f.write(f'bossbar set progress players')  # Clear progress bar
            f.write(f'execute if score tok {pack} matches 2 run return 1')  # Break the loop if it generates EOS token
            
            f.write(f'function {pack}:token_to_string')
            f.write(f'data modify storage {pack} args.output append from storage {pack} args.tok')
            f.write(f'tellraw @a ["{clear_screen}",{{"storage":"{pack}","nbt":"args.output[]","separator":""}}]')
            f.write(f'scoreboard players add pos {pack} 1')
            f.write(f'function {pack}:respond')

    def _generate_setup(self):
        pack = self.pack
        # Initialization function
        with self.write_function('setup') as f:
            f.write(f'gamerule max_command_sequence_length {(1 << 31) - 1}')
            f.write(f'scoreboard objectives add {pack} dummy')  # Global objective for temp vars
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
            for n in sorted(self.consts):
                f.write(f'scoreboard players set {n} {pack} {n}')
                
            # Create progress bar
            f.write(f'bossbar add progress "Progress"')

            # Initialize parameters
            f.write(f'function {pack}:load_parameters')

            # Clear KV cache
            f.write(f'scoreboard players reset * kc')
            f.write(f'scoreboard players reset * vc')


class Llama2Model:
    def __init__(self, checkpoint: str, tokenizer: str):
        self.BOS = 1
        self.EOS = 2

        with open(checkpoint, 'rb') as f:
            # Read in the config header
            _config = f.read(struct.calcsize('7i'))
            # Unpacking the config data
            (
                self.dim, self.hidden_dim, self.n_layers, self.n_heads,
                self.n_kv_heads, self.vocab_size, self.seq_len,
            ) = struct.unpack('7i', _config)
            
            # negative vocab size is hacky way of signaling unshared weights
            shared_weights = self.vocab_size > 0
            self.vocab_size = abs(self.vocab_size)

            # Initialize the weights from checkpoint
            def read_floats(count) -> tuple[float]:
                return struct.unpack(str(count) + 'f', f.read(count * 4 if count > 0 else count))

            head_size = self.dim // self.n_heads

            self.token_embedding_table = read_floats(self.vocab_size * self.dim)
            self.rms_att_weight = read_floats(self.n_layers * self.dim)
            self.wq = read_floats(self.n_layers * self.dim * self.dim)
            self.wk = read_floats(self.n_layers * self.dim * self.n_kv_heads * head_size)
            self.wv = read_floats(self.n_layers * self.dim * self.n_kv_heads * head_size)
            self.wo = read_floats(self.n_layers * self.dim * self.dim)
            self.rms_ffn_weight = read_floats(self.n_layers * self.dim)
            self.w1 = read_floats(self.n_layers * self.dim * self.hidden_dim)
            self.w2 = read_floats(self.n_layers * self.hidden_dim * self.dim)
            self.w3 = read_floats(self.n_layers * self.dim * self.hidden_dim)
            self.rms_final_weight = read_floats(self.dim)
            read_floats(self.seq_len * head_size)
            self.wcls = self.token_embedding_table if shared_weights else read_floats(self.vocab_size * self.dim)
        
        # Initialize the tokenizer
        self.vocab: list[str] = []
        self.vocab_scores: list[int] = []

        with open(tokenizer, 'rb') as f:
            self.max_token_length = struct.unpack('i', f.read(4))[0]
            for _ in range(0, self.vocab_size):
                self.vocab_scores.append(int(struct.unpack('f', f.read(4))[0]))
                length = struct.unpack('i', f.read(4))[0]
                bstr = f.read(length)
                if type(bstr) is not str:
                    bstr = bstr.decode('utf-8')

                # Be careful with `<0xXX>` raw byte representations
                if bstr.startswith('<0x') and bstr.endswith('>') and len(bstr) == 6:
                    for c in bstr[3:4]:
                        if c not in '1234567890ABCDEF':
                            break
                    else:
                        bstr = chr(int(bstr[1:5], 16))

                self.vocab.append(bstr)

    def generate_datapack(self, pack: str):
        datapack_gen = Llama2DatapackGenerator(self, pack)
        datapack_gen.generate()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Deploy Llama 2 models into vanilla Minecraft')
    parser.add_argument('checkpoint', help='Path to model checkpoint (.bin)')
    parser.add_argument('-z', '--tokenizer', required=True, help='Path to tokenizer model (.bin)')
    parser.add_argument('-o', '--output', default='out.zip', help='Path to output datapack')
    args = parser.parse_args()

    model = Llama2Model(args.checkpoint, args.tokenizer)
    model.generate_datapack(args.output)
