export VLLM_USE_FLASH_ATTN_TRITON=0 
export PYTORCH_TUNABLEOP_ENABLED=1 
export PYTORCH_TUNABLEOP_FILENAME=benchmarks/tuned.csv 
export PYTORCH_TUNABLEOP_TUNING=0 
export HIP_VISIBLE_DEVICES=2,3 
python benchmarks/benchmark_throughput_cust.py \
	--model /dockerx/models/Llama-2-70b-hf \
	--num-prompts=1000 \
	--input-len 1000 \
	--output-len 500 \
	--tensor-parallel-size 2 \
	--trust-remote-code \
	--worker-use-ray

