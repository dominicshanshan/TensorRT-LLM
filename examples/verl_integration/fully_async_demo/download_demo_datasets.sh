#!/bin/bash

echo "=========================================="
echo "VERL Demo Dataset Downloader"
echo "=========================================="

# Create data directory
DATA_DIR="${DATA_DIR:-./demo_data}"
mkdir -p ${DATA_DIR}

# Choose dataset to download
DATASET=${1:-"hh-rlhf"}

case $DATASET in
    "gsm8k")
        echo "Downloading GSM8K dataset (math problems)..."
        echo "This is a small dataset with ~7.5k training examples"
        
        # Create Python script to download and process GSM8K
        cat > ${DATA_DIR}/download_gsm8k.py << 'EOF'
import os
import re
import datasets
import pandas as pd
from datasets import load_dataset

def extract_solution(solution_str):
    """Extract numerical answer from GSM8K solution"""
    solution = re.search(r"#### (\-?[0-9\.\,]+)", solution_str)
    if solution:
        final_solution = solution.group(0)
        final_solution = final_solution.split("#### ")[1].replace(",", "")
        return final_solution
    return ""

# Download GSM8K dataset
print("Downloading GSM8K dataset from HuggingFace...")
dataset = load_dataset("openai/gsm8k", "main")

instruction = 'Let\'s think step by step and output the final answer after "####".'

def process_example(example, idx, split):
    """Process GSM8K example into VERL format"""
    question_raw = example["question"]
    question = question_raw + " " + instruction
    answer_raw = example["answer"]
    solution = extract_solution(answer_raw)
    
    return {
        "data_source": "openai/gsm8k",
        "prompt": [{
            "role": "user",
            "content": question,
        }],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": solution},
        "extra_info": {
            "split": split,
            "index": idx,
            "answer": answer_raw,
            "question": question_raw,
        }
    }

# Process train and test sets
print("Processing training data...")
train_data = [process_example(ex, idx, "train") 
              for idx, ex in enumerate(dataset["train"])]

print("Processing test data...")
test_data = [process_example(ex, idx, "test") 
             for idx, ex in enumerate(dataset["test"])]

# Save as parquet files
print(f"Saving to parquet format...")
pd.DataFrame(train_data).to_parquet("train.parquet", index=False)
pd.DataFrame(test_data).to_parquet("test.parquet", index=False)

# Create smaller subsets for quick testing
print("Creating small subsets for quick testing...")
pd.DataFrame(train_data[:1000]).to_parquet("train_small.parquet", index=False)
pd.DataFrame(test_data[:100]).to_parquet("test_small.parquet", index=False)

print(f"✓ Full dataset: {len(train_data)} train, {len(test_data)} test examples")
print(f"✓ Small subset: 1000 train, 100 test examples")
print("Files saved:")
print("  - train.parquet (full training set)")
print("  - test.parquet (full test set)")
print("  - train_small.parquet (1000 examples)")
print("  - test_small.parquet (100 examples)")
EOF
        
        cd ${DATA_DIR}
        python download_gsm8k.py
        rm download_gsm8k.py
        ;;
    
    "math")
        echo "Downloading MATH dataset (competition math problems)..."
        
        cat > ${DATA_DIR}/download_math.py << 'EOF'
import os
import pandas as pd
from huggingface_hub import snapshot_download

# Download preprocessed MATH dataset from VERL team
print("Downloading MATH dataset from verl-team...")
snapshot_download(
    repo_id="verl-team/lighteval-MATH-preprocessed",
    repo_type="dataset",
    local_dir="./math_dataset",
)

# Create smaller subsets
print("Creating small subsets...")
train_df = pd.read_parquet("./math_dataset/train.parquet")
test_df = pd.read_parquet("./math_dataset/test.parquet")

# Save full datasets
train_df.to_parquet("train.parquet", index=False)
test_df.to_parquet("test.parquet", index=False)

# Save small subsets
train_df.head(1000).to_parquet("train_small.parquet", index=False)
test_df.head(100).to_parquet("test_small.parquet", index=False)

print(f"✓ Full dataset: {len(train_df)} train, {len(test_df)} test examples")
print(f"✓ Small subset: 1000 train, 100 test examples")
print("Files saved:")
print("  - train.parquet (full training set)")
print("  - test.parquet (full test set)")
print("  - train_small.parquet (1000 examples)")
print("  - test_small.parquet (100 examples)")

# Clean up
import shutil
shutil.rmtree("./math_dataset")
EOF
        
        cd ${DATA_DIR}
        python download_math.py
        rm download_math.py
        ;;
    
    "hh-rlhf")
        echo "Downloading Anthropic HH-RLHF dataset (helpful & harmless)..."
        
        cat > ${DATA_DIR}/download_hhrlhf.py << 'EOF'
import pandas as pd
from datasets import load_dataset

print("Downloading HH-RLHF dataset...")
# Using VERL team's preprocessed version since Anthropic/hh-rlhf is archived
dataset = load_dataset("verl-team/Anthropic-hh-rlhf-rm-preprocessed")

def process_example(example, idx, split):
    """Process HH-RLHF example into VERL format"""
    # Extract human prompt from the chosen response
    chosen = example["chosen"]
    # The format is usually "Human: ... Assistant: ..."
    human_part = chosen.split("\n\nAssistant:")[0]
    human_prompt = human_part.replace("Human: ", "").strip()
    
    return {
        "data_source": "verl-team/Anthropic-hh-rlhf-rm-preprocessed",
        "prompt": [{
            "role": "user",
            "content": human_prompt,
        }],
        "ability": "helpful-harmless",
        "chosen": example["chosen"],
        "rejected": example["rejected"],
        "extra_info": {
            "split": split,
            "index": idx,
        }
    }

# Process train and test sets
print("Processing training data...")
train_data = [process_example(ex, idx, "train") 
              for idx, ex in enumerate(dataset["train"]) if idx < 10000]

print("Processing test data...")
test_data = [process_example(ex, idx, "test") 
             for idx, ex in enumerate(dataset["test"]) if idx < 1000]

# Save as parquet
pd.DataFrame(train_data).to_parquet("train.parquet", index=False)
pd.DataFrame(test_data).to_parquet("test.parquet", index=False)

# Create smaller subsets
pd.DataFrame(train_data[:1000]).to_parquet("train_small.parquet", index=False)
pd.DataFrame(test_data[:100]).to_parquet("test_small.parquet", index=False)

print(f"✓ Dataset: {len(train_data)} train, {len(test_data)} test examples")
print(f"✓ Small subset: 1000 train, 100 test examples")
EOF
        
        cd ${DATA_DIR}
        python download_hhrlhf.py
        rm download_hhrlhf.py
        ;;
    
    *)
        echo "Unknown dataset: $DATASET"
        echo "Available datasets:"
        echo "  gsm8k    - Grade school math problems (~7.5k examples)"
        echo "  math     - Competition math problems from VERL team"
        echo "  hh-rlhf  - Anthropic helpful & harmless dataset"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "Dataset downloaded successfully!"
echo "Location: ${DATA_DIR}/"
echo ""
echo ""
echo "To use in training:"
echo "  Full dataset:"
echo "    export TRAIN_FILE=\"${DATA_DIR}/train.parquet\""
echo "    export TEST_FILE=\"${DATA_DIR}/test.parquet\""
echo ""
echo "  Small subset (for quick testing):"
echo "    export TRAIN_FILE=\"${DATA_DIR}/train_small.parquet\""
echo "    export TEST_FILE=\"${DATA_DIR}/test_small.parquet\""
echo "=========================================="