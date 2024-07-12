from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.nn.attention import SDPBackend, sdpa_kernel
import torch
import numpy as np
import gc

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", token='')
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B", token='', device_map='cuda:0', attn_implementation='sdpa', torch_dtype=torch.bfloat16)

example = ' '.join(['sentence'] * 500000)

for max_length in range(8192, 53192, 2000):
    input_ids = tokenizer(example, truncation=True, max_length=max_length, return_tensors='pt').input_ids
    #input_data = input_ids.repeat(batch_size, 1).to('cuda:0')

    input_data = input_ids.to('cuda:0')

    print(input_data.shape)

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 50
    timings = np.zeros((repetitions, 1))

    model.eval()
    with torch.no_grad():
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            for rep in range(repetitions):
                starter.record()
                _ = model(input_data)
                del _
                gc.collect()
                torch.cuda.empty_cache()
                ender.record()
                
                # Synchronize GPU
                torch.cuda.synchronize()
                
                curr_time = starter.elapsed_time(ender)
                timings[rep] = curr_time

    mean_time = np.sum(timings) / repetitions
    std_time = np.std(timings)

    print(mean_time, std_time)
    print(f'{max_length=}, {mean_time / 1000}sec, {input_data.shape[0] * input_data.shape[1] * 1000 / mean_time} tok/s')
