export VLLM_USE_FLASH_ATTN_TRITON=0 
export PYTORCH_TUNABLEOP_ENABLED=1 
export PYTORCH_TUNABLEOP_FILENAME=tuned.csv 
export PYTORCH_TUNABLEOP_TUNING=0 
export HIP_VISIBLE_DEVICES=2,3
export TOKENIZERS_PARALLELISM=false
torchrun --standalone --nproc_per_node=2 \
benchmark_throughput_cust.py \
	--model ../../../models/Llama-2-70b-hf \
	--num-prompts=10 \
	--input-len 2048 \
	--output-len 20 \
	--tensor-parallel-size 2 \
	--trust-remote-code \
	--dtype bfloat16 


