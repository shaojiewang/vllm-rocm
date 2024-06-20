import os
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from hipbsolidxgemm import hipb_create_extension, hipb_mm
from rocsolidxgemm import rocb_create_extension, rocb_mm

from vllm import _custom_C


class TunedGemm:

    def __init__(self):
        #rocb_create_extension()
        #hipb_create_extension()
        self.extensions_created = False
        self.save_gemm = int(os.environ.get('VLLM_TUNE_GEMM', 0))
        self.untune_path = os.environ.get('VLLM_UNTUNE_FILE',
                                          "/tmp/vllm_untuned.csv")
        self.tune_path = os.environ.get('VLLM_TUNE_FILE', "tuned.csv")
        self.bestsols = {}
        self.load_best_sols()
        self.create_ds()
        self.cu_count = torch.cuda.get_device_properties(
            device='cuda').multi_processor_count

        if (self.save_gemm == 1):
            self.tuned_df = pd.DataFrame(columns=['M', 'N', 'K'])
        else:
            self.tuned_df = None

    def load_best_sols(self):
        if self.tune_path is not None and Path(self.tune_path).is_file():
            self.bestsols = pd.read_csv(self.tune_path)

    def create_ds(self):
        df: pd.DataFrame = self.bestsols
        solds = {}
        for i in range(len(df)):
            ds = df.iloc[i]
            key = (ds['M'], ds['N'], ds['K'])
            if ds['libtype'] == 'hipblaslt':
                soltype = 1
            elif ds['libtype'] == 'rocblas':
                soltype = 2
            solds[key] = (soltype, int(ds['solidx']))
        self.solids = solds
        #print('>>>',solds)
    def query_sol(self, m, n, k):
        return self.solids.get((m, n, k), (0, 0))

    def apply_skinny(self, m, n, k, inp_view, weights):
        if inp_view.dtype != torch.float16 or k % 8 != 0:
            return None
        if m > 8 and n <= 4:
            out = torch.empty(inp_view.shape[0],
                              weights.shape[0],
                              dtype=inp_view.dtype,
                              device='cuda')
            _custom_C.wvSpltK(weights, inp_view, out, n, self.cu_count)
            return out
        elif m % 4 == 0 and n == 1 and k <= 8192:
            out = torch.empty(inp_view.shape[0],
                              weights.shape[0],
                              dtype=inp_view.dtype,
                              device='cuda')
            _custom_C.LLMM1(weights, inp_view, out, 4)
            return out
        else:
            return None

    def mm(self, inp, weights, bias=None):
        # F.Linear can take a 3 dimensional input. vllm
        # uses this for linear units. However, sampler
        # will use torch.matmul with 2 dimensions only
        if inp.dim() == 3:
            inp_view = inp.view(-1, inp.size(-1))
            batched = True
        else:
            inp_view = inp
            batched = False
        if self.extensions_created is False:
            rocb_create_extension()
            hipb_create_extension()
            self.extensions_created = True
        m = weights.shape[0]
        n = inp_view.shape[0]
        k = inp_view.shape[1]
        soltype, solidx = self.query_sol(m=m, n=n, k=k)
        out = self.apply_skinny(m, n, k, inp_view, weights)
        if out is not None:
            pass
        elif soltype == 1:
            out = hipb_mm(inp_view, weights.t(), solidx)
        elif soltype == 2:
            out = rocb_mm(inp_view, weights.t(), solidx)
        else:
            if (self.save_gemm == 1):
                self.tuned_df = pd.concat([
                    self.tuned_df,
                    pd.DataFrame({
                        'M': [m],
                        'N': [n],
                        'K': [k]
                    })
                ]).drop_duplicates()
                self.tuned_df.to_csv(self.untune_path, index=False)
            return F.linear(inp, weights, bias)
        if batched:
            out = out.view(inp.shape[0], inp.shape[1], weights.shape[0])
        if bias is not None:
            return out + bias
        return out


tgemm = TunedGemm()
