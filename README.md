# Reproducibility and Extension Challenge: DistilBERT

This project reproduces and partially extends the results from DistilBERT, a distilled version of BERT 
Sanh et al., 2019 

https://arxiv.org/abs/1910.01108

Goal:
- reproduce the knowledge distillation setup
- compare DistilBERT and BERT-base on downstream tasks

---

# Structure

---


# Reproduction

In order to cashe the data (smaller data for debugging and bigger data for the actual training), we first run:
```bash
python download_data.py
```
```bash
python download_eval_data.py
```
For training we run the run_pipeline.py script like this:
```bash
python src/run_pipeline.py --epochs 3 --batch_size 8 --paper_subset 0 --paper_data_dir .
```
Parameters:

--epochs 3 - number of training epochs

--batch_size 8 - batch size (kept small due to hardware limits)

--paper_subset 0 - use the full available dataset (no paper subset reduction)

--paper_data_dir . - data directory

- DistilBERT (student) is initialized from selected BERT layers

- training uses a combination of MLM loss and distillation loss

# Evaluation

We evaluate the trained models on multiple downstream tasks. Due to limited compute resources, some evaluations
are performed on reduced subsets of the datasets.

## GLUE (dev)

- **Tasks:** CoLA, MNLI, MRPC, QNLI, QQP, RTE, SST-2, STS-B, WNLI  
- **Metrics:** Task-specific metrics (accuracy, Matthew’s correlation, Pearson) 
- **Setup:** Identical hyperparameters for all models, same random seed, each model is fine-tuned separately per GLUE task, max_samples = 1000

**Command:**
```bash
python src/eval/eval_glue.py --model ./checkpoints_paper_like/seed42 --cache_dir ./data/hf_cache --seeds 42 --epochs 2 --batch_size 8 --max_samples 1000
```

### Results (dev sets, parameters above)

| Model       | CoLA  | MNLI  | MRPC  | QNLI  | QQP   | RTE   | SST-2 | STS-B | WNLI  |
|------------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| BERT-base  | 0.4012 | 0.4460 | 0.7574 | 0.6450 | 0.6610 | 0.6101 | 0.8819 | 0.7839 | 0.2958 |
| DistilBERT | 0.3245 | 0.3480 | 0.6985 | 0.5380 | 0.6620 | 0.5271 | 0.6617 | 0.1045 | 0.5493 |

### Results (Full Dataset, 5 Seeds Median – DistilBERT)

| Model       | CoLA  | MNLI  | MRPC  | QNLI  | QQP   | RTE   | SST-2 | STS-B | WNLI  |
|------------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| DistilBERT | 0.4642 | 0.8145 | 0.8284 | 0.8816 | 0.9001 | 0.5776 | 0.9071 | 0.8550 | 0.5493 |


## IMDb

- **Task:** Binary sentiment classification of movie reviews  
- **Metric:** Accuracy (test set)  
- **Setup:** Identical hyperparameters for all models, same random seed

**Command:**
```bash
python src/eval/eval_imdb.py --model ./checkpoints_paper_like/seed42 --cache_dir ./data/hf_cache --seeds 42 --epochs 2 --batch_size 8
```

###  Results

| Model       | IMDb Accuracy |
|------------|---------------|
| BERT-base  | 0.9214        |
| DistilBERT | 0.9040        |

DistilBERT achieves performance close to BERT-base while using
  significantly fewer parameters.

## SQuAD 1.1 (dev)

- **Task:** extractive question answering (predict answer span from context)  
- **Metric:** exact match (EM), F1 score  
- **Setup:** identical hyperparameters for all models, same random seed, limited training subset (10k samples)  
- **Observation:** DistilBERT achieves performance close to BERT-base while using
  significantly fewer parameters.

**Command:**
```bash
python src\eval\eval_squad.py --model bert-base-uncased --seeds 42 --train_bs 2 --eval_bs 2 --epochs 2 --max_train_samples 10000 --max_eval_samples 10000 --cache_dir data\hf_cache
```
### Results

| Model      | SQuAD (EM / F1) |
|------------|----------------|
| BERT-base  | 72.72 / 81.68  |
| DistilBERT | 54.81 / 65.70  |

## Inference Speed & Model Size

- **Dataset:** STS-B (GLUE)  
- **Metric:** end-to-end inference time and parameter numbers for a full dataset pass  
- **Setup:** batch size = 1, 1000 samples, identical hardware and settings across models
- **Observation:** DistilBERT achieves performance close to BERT-base while using
  significantly fewer parameters.

**Command:**
```bash
python src/eval/benchmark_speed.py --model ./checkpoints_paper_like/seed42 --cache_dir ./data/hf_cache --warmup 10 --threads 1 --max_samples 1000
```

### Results

| Model       | Parameters | Inference Time |
|------------|------------|----------------|
| BERT-base  | 109M       | 71.24 s        |
| DistilBERT | 67M        | 34.23 s        |

# Extension 1: DistilBERT Edge Deployment Reality Check

- Does DistilBERT really run efficiently on edge devices?  
- If not, how much can we optimize it?  

Implement quantization, pruning, and combination(quantization + pruning)  
- Benchmarked size, speed, quality trade-offs  
- Verified edge feasibility through ONNX Runtime on Android

## Research Questions  
- Does the trained DistilBERT actually run efficiently on edge devices?  
- What are the main deployment bottlenecks (model size, latency, memory)?  
- How can we make it smaller and faster (quantization / pruning / combined)?  
- What is the performance–size trade-off?

## Optimization Methods Implemented
(1) Baseline (Trained Student Model)  
- DistilBERT student trained using triple loss:  
  - MLM loss
  - Distillation loss (KL divergence vs teacher)
  - Cosine hidden-state loss

(2) Quantization (INT8)  
- Applied post-training dynamic quantization (Linear layers → INT8)
- Reduce model size + improve inference speed for deployment

(3) Pruning  
- Applied global unstructured magnitude pruning
- Reduce redundant weights / improve runtime

(4) Combined (Pruning + Quantization)  
- Applied pruning first, then quantization  
- Maximize compression + speed-up

## Benchmark Setup
### CPU Benchmark (Python)  
- Model size (MB)
- Parameter count
- Cold start latency (ms)
- Inference latency (multiple input lengths)
- Throughput (samples/sec)
- Peak memory usage (MB)

### Quality metrics (triple-loss components):
- MLM Eval Loss
- Distill Eval Loss
- Cosine Eval Loss

### Android Edge Benchmark (ONNX Runtime)  
- ONNX Runtime Android
- Device info:
  - Android 14 (API 34, UpsideDownCake)
  - ABI: arm64-v8a
  - RAM: 4GB
  - Emulator: Google APIs
 
## Result Summary
### Key Comparison Table

| Model | Size (MB) | Params | Cold Start (ms) | Throughput (samples/s) | Peak Memory (MB) |
|------|----------:|-------:|----------------:|------------------------:|-----------------:|
| Baseline | 255.57 | 66,985,530 | 15.51 | 82.33 | 60.70 |
| Quantized (INT8) | 154.77 | 66,985,530 | 12.15 | 136.39 | 4.05 |
| Pruned | 255.57 | 66,985,530 | 12.65 | 82.82 | 0.41 |
| Combined (Prune+INT8) | 154.77 | 66,985,530 | **9.28** | 135.05 | **0.00** |


### Quality (Loss-based Evaluation) (Lower is better)

| Model | MLM Eval Loss | Distill Eval Loss | Cosine Eval Loss |
|------|-----------------:|---------------------:|-------------------:|
| Baseline | 6.6854 | 7.6096 | 0.2679 |
| Quantized | 6.8083 | 7.8827 | 0.2818 |
| Pruned | 6.7859 | 7.9324 | 0.2761 |
| Combined | 6.9024 | 8.1377 | 0.2897 |

- Quantization + pruning introduces a small loss degradation  
- Overall losses remain in a similar range -> Quality is preserved for deployment use-cases  

## Conclusion
### What worked best for edge deployment?
- INT8 quantization provides the biggest practical gain.  
- Model size reduced by 39.4%  
- Throughput increased from 82.33 -> 136.39 samples/sec  
- Memory footprint dropped significantly (60.70MB -> 4.05MB)

### Does pruning help?
- Pruning alone did not reduce file size 
- Throughput improvement was minimal  
- It can reduce runtime memory depending on implementation, but file size remains large  

### Is pruning + quantization “orthogonal” like the paper says?  
Yes 
The combined model achieves:  
- best cold start (9.28ms)  
- high throughput (135.05 samples/sec)  
- same compressed size as quantized (154.77MB)

## ONNX Edge Deployment Notes (Android)  
- PyTorch models are not ideal for direct Android deployment  
- ONNX Runtime provides efficient mobile inference backends  

***Implementation Detail***  
We export FP32 ONNX first, then apply ONNX Runtime quantization:  
- Quantized PyTorch models are not straightforward to export.

### Android Edge Benchmark (ONNX Runtime)
code: https://drive.google.com/drive/folders/1GNy9ZcRREk7myLXbQ97So9-iF3E4_Sjx?usp=sharing
| Model (ONNX) | Size (MB) | Avg Latency (ms) | p50 (ms) | p90 (ms) |
|---|---:|---:|---:|---:|
| distilbert_baseline.onnx | 345.06 | 164.35 | 163.64 | 166.01 |
| distilbert_baseline_int8.onnx | 86.71 | 51.12 | 51.05 | 51.24 |
| distilbert_pruned.onnx | 345.06 | 163.45 | 162.88 | 163.99 |
| distilbert_pruned_int8.onnx | 86.71 | 51.29 | 51.28 | 51.73 |

INT8 quantization is the most effective optimization for edge.  
- Model size drops.  
- Latency becomes much faster on Android.  
- Throughput increases a lot.  

Pruning alone is not enough.  
- It does not reduce model file size (still ~255MB).  
- It gives small or inconsistent speed gain.  

Quantization + pruning does not beat pure INT8.
- Combined is similar size to INT8.  
- Combined latency can even be worse (because unstructured pruning is not hardware-friendly).

Quality did not collapse after INT8.  
- MLM, distillation, cosine losses stay close to baseline.  
- So INT8 keeps behavior reasonably similar for the student model.  


Best Model: INT8 quantization     
- Best for speed + size + deployability.    

**DistilBERT is “edge-runnable”, but not “edge-efficient” until you apply quantization.**


## How to Run
Full pipeline
```
python run_complete_pipeline.py --output-dir results
```
Quick test (debug)
```
python run_complete_pipeline.py --quick-test --output-dir results_quick
```



