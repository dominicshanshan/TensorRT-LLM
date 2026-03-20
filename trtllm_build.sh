#!/bin/bash
set -e 

git config --global --add safe.directory \"*\"
python3 ./scripts/build_wheel.py \
	-a "native" \
	-b "RelWithDebInfo" \
	-G "Ninja" \
	-c \
	--use_ccache \
	--benchmarks \
	--micro_benchmarks

pip install ./build/tensorrt_llm*.whl
