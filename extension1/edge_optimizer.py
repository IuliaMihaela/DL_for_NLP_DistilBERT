"""
edge_optimizer.py
"""
import re
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import time
import os, sys
import psutil
import numpy as np
from typing import Dict, List, Optional
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, SRC_DIR)

from src.model import DistilBertStudent  

import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM, 
    DataCollatorForLanguageModeling,
)


# EDGE BENCHMARK
# Real-world edge deployment benchmarking
class EdgeBenchmark:
    def __init__(self, model, tokenizer, device='cpu', teacher_model: Optional[nn.Module] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        # Deployable student model
        self.m = self._deployable_model()
        self.m.to(device)
        self.m.eval()

        # Teacher model for quality metrics (triple loss components)
        self.teacher = teacher_model
        if self.teacher is not None:
            self.teacher.to(device)
            self.teacher.eval()

        # Collator for MLM-style masking (shared across student/teacher)
        self.mlm_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=True,
            mlm_probability=0.15,
        )

    # KL(student || teacher) on masked positions only.
    def _masked_kl_div(self, student_logits, teacher_logits, labels, temperature: float = 2.0):
        mask = labels != -100  # [B, L]
        if mask.sum().item() == 0:
            return torch.tensor(0.0, device=student_logits.device)

        s = student_logits[mask]
        t = teacher_logits[mask]

        T = temperature
        s_log_probs = F.log_softmax(s / T, dim=-1)
        t_probs = F.softmax(t / T, dim=-1)

        loss = F.kl_div(s_log_probs, t_probs, reduction="batchmean") * (T * T)
        return loss

    # Hidden state directions using cosine loss.
    def _cosine_hidden_loss(self, student_hidden_states, teacher_hidden_states, attention_mask):   
        attn = attention_mask.float()  # [B, L]
        denom = attn.sum().clamp(min=1.0)

        total = 0.0
        n = 0

        max_j = min(len(student_hidden_states), (len(teacher_hidden_states) + 1) // 2)
        for j in range(max_j):
            tj = 2 * j
            if tj >= len(teacher_hidden_states):
                break

            s_h = student_hidden_states[j]   # [B, L, H]
            t_h = teacher_hidden_states[tj]  # [B, L, H]

            cos = F.cosine_similarity(s_h, t_h, dim=-1)  # [B, L]
            layer_loss = ((1.0 - cos) * attn).sum() / denom
            total = total + layer_loss
            n += 1

        if n == 0:
            return torch.tensor(0.0, device=attention_mask.device)
        return total / n

    # Build eval samples (Wikipedia text dump)
    @staticmethod
    def load_eval_sentences_from_file(
        path: str,
        max_samples: int = 10000,
        min_chars: int = 40,
        chunk_tokens: int = 128,
        overlap_tokens: int = 32,
        seed: int = 1234,
    ) -> List[str]:
        
        # Clean lines
        paras: List[str] = []
        buf: List[str] = []

        header_pat = re.compile(r"^\s*=+\s*[^=].*[^=]\s*=+\s*$")  # "== Title ==" style
        bullet_pat = re.compile(r"^\s*[\*\-]\s+")                # bullet lines

        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()

                # Paragraph break
                if line == "":
                    if buf:
                        para = " ".join(buf).strip()
                        if len(para) >= min_chars:
                            paras.append(para)
                        buf = []
                    continue

                # Skip headers / noisy markers
                if header_pat.match(line):
                    continue

                # Skip single "="
                if line.replace("=", "").strip() == "":
                    continue

                # Keep decent lines
                if len(line) < 5:
                    continue

                buf.append(line)

        if buf:
            para = " ".join(buf).strip()
            if len(para) >= min_chars:
                paras.append(para)

        if not paras:
            return []

        # Chunk paragraphs into token windows
        def whitespace_tokens(text: str) -> List[str]:
            return text.split()

        chunks: List[str] = []
        for para in paras:
            toks = whitespace_tokens(para)
            if len(toks) <= chunk_tokens:
                chunks.append(para)
            else:
                step = max(1, chunk_tokens - overlap_tokens)
                for s in range(0, len(toks), step):
                    window = toks[s:s + chunk_tokens]
                    if len(window) < 20:
                        break
                    chunks.append(" ".join(window))

        # Shuffle
        rng = np.random.RandomState(seed)
        rng.shuffle(chunks)

        return chunks[:max_samples]


    # Run all benchmarks
    def run_all_benchmarks(self, test_sentences: List[str] = None, eval_sentences: List[str] = None) -> Dict:
        if test_sentences is None:
            test_sentences = [
                "This is a short test.",
                "This is a medium length sentence for testing inference time.",
                "This is a much longer sentence that we use to test how the model performs with more tokens and see if there are any performance degradation issues."
            ]

        print("\n" + "="*60)
        print("EDGE DEPLOYMENT BENCHMARK")
        print("="*60)

        results = {}

        # Model size
        print("\n[1/7] Measuring model size...")
        results['model_size_mb'] = self.measure_model_size()
        print(f"  Model size: {results['model_size_mb']:.2f} MB")

        # Parameter count
        print("\n[2/7] Counting parameters...")
        results['total_params'] = self.count_parameters()
        print(f"  Total parameters: {results['total_params']:,}")

        # Cold start latency
        print("\n[3/7] Measuring cold start latency...")
        results['cold_start_ms'] = self.measure_cold_start(test_sentences[0])
        print(f"  Cold start: {results['cold_start_ms']:.2f} ms")

        # Inference latency (warm)
        print("\n[4/7] Measuring inference latency...")
        latency_results = self.measure_inference_latency(test_sentences)
        results.update(latency_results)
        for key, val in latency_results.items():
            print(f"  {key}: {val:.2f} ms")

        # Throughput
        print("\n[5/7] Measuring throughput...")
        results['throughput_samples_per_sec'] = self.measure_throughput()
        print(f"  Throughput: {results['throughput_samples_per_sec']:.2f} samples/sec")

        # Memory usage
        print("\n[6/7] Measuring memory footprint...")
        results['peak_memory_mb'] = self.measure_memory_usage()
        print(f"  Peak memory: {results['peak_memory_mb']:.2f} MB")

        # Quality metrics (triple loss components)
        print("\n[7/7] Measuring quality metrics (MLM + Distill + Cosine)...")
        quality = self.evaluate_quality_metrics(eval_sentences, max_length=128, batch_size=16)
        results.update(quality)

        def _fmt(val, fmt="{:.4f}"):
            return "N/A" if val is None else fmt.format(val)

        print(f"  mlm_eval_loss:      {_fmt(results.get('mlm_eval_loss'))}")
        print(f"  distill_eval_loss:  {_fmt(results.get('distill_eval_loss'))}")
        print(f"  cosine_eval_loss:   {_fmt(results.get('cosine_eval_loss'))}")
        return results
    
    # Return deployable model 
    def _deployable_model(self):
        return self.model.student if hasattr(self.model, "student") else self.model

    # Clean inputs
    def _sanitize_inputs(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        inputs = dict(inputs)
        inputs.pop("token_type_ids", None)
        return inputs

    # Measure Model Size
    def measure_model_size(self) -> float:
        temp_path = "temp_model_size.pt"
        m = self.m
        torch.save(m.state_dict(), temp_path)
        size_bytes = os.path.getsize(temp_path)
        os.remove(temp_path)
        return size_bytes / (1024 * 1024)

    # Count parameters
    def count_parameters(self) -> int:
        m = self.m
        return sum(p.numel() for p in m.parameters())

    # Measure Cold Start (Frist inference time - includes model initialization)
    def measure_cold_start(self, text: str) -> float:
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs = self._sanitize_inputs(inputs)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        m = self.m

        start = time.perf_counter()
        with torch.no_grad():
            _ = m(**inputs)
        end = time.perf_counter()

        return (end - start) * 1000

    # Mesure average inference time (per different input lengths)
    def measure_inference_latency(self, test_sentences: List[str], num_runs: int = 100) -> Dict:
        results = {}
        m = self.m

        for _, text in enumerate(test_sentences):
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            inputs = self._sanitize_inputs(inputs)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Warmup
            with torch.no_grad():
                for _ in range(10):
                    _ = m(**inputs)

            # Measure
            times = []
            with torch.no_grad():
                for _ in range(num_runs):
                    start = time.perf_counter()
                    _ = m(**inputs)
                    end = time.perf_counter()
                    times.append((end - start) * 1000)

            seq_len = inputs['input_ids'].shape[1]
            results[f'latency_len_{seq_len}_ms'] = float(np.mean(times))

        return results

    # Measure samples processed (per sec)
    def measure_throughput(self, duration_sec: int = 3) -> float:
        text = "This is a test sentence."
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs = self._sanitize_inputs(inputs)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        m = self.m

        count = 0
        start = time.perf_counter()
        with torch.no_grad():
            while time.perf_counter() - start < duration_sec:
                _ = m(**inputs)
                count += 1

        elapsed = time.perf_counter() - start
        return count / elapsed

    # Measure peak memory during inference
    def measure_memory_usage(self) -> float:
        import gc
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        m = self.m

        process = psutil.Process()
        mem_before = process.memory_info().rss
        peak = mem_before

        text = "This is a test sentence. " * 50
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs = self._sanitize_inputs(inputs)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            for _ in range(50):
                _ = m(**inputs)
                peak = max(peak, process.memory_info().rss)

        return (peak - mem_before) / (1024 * 1024)

    # Extract last hidden states
    def _get_last_hidden(self, outputs):
        if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None and len(outputs.hidden_states) > 0:
            return outputs.hidden_states[-1]
        if isinstance(outputs, (tuple, list)) and len(outputs) >= 2:
            hs = outputs[1]
            if isinstance(hs, (tuple, list)) and len(hs) > 0:
                return hs[-1]
        return None

    # Evaluate Triple loss - MLM Eval Loss, Distill Eval Loss, Cosine Eval Loss
    def evaluate_quality_metrics(self, texts: List[str], max_length: int = 128, batch_size: int = 8) -> Dict:
        if texts is None or len(texts) == 0:
            return {
                "mlm_eval_loss": None,
                "distill_eval_loss": None,
                "cosine_eval_loss": None,
            }

        m = self.m
        t = self.teacher

        # Accumulators
        mlm_loss_sum = 0.0
        mlm_batches = 0

        distill_sum = 0.0
        distill_batches = 0

        cosine_sum = 0.0
        cosine_batches = 0

        temperature = 2.0

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            enc = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            enc = self._sanitize_inputs(enc)

            features = []
            for b in range(enc["input_ids"].shape[0]):
                features.append({
                    "input_ids": enc["input_ids"][b],
                    "attention_mask": enc["attention_mask"][b],
                })

            # Apply dynamic MLM masking
            batch = self.mlm_collator(features)
            batch = self._sanitize_inputs(batch)

            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)  # -100 for non-masked positions

            inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
            masked_positions = (labels != -100)

            with torch.no_grad():
                # Student forward (HF model)
                student_out = m(**inputs, labels=labels, output_hidden_states=True, return_dict=True)
                student_logits = student_out.logits
                student_hid = student_out.hidden_states  # tuple

                # MLM loss computed
                flat_logits = student_logits.reshape(-1, student_logits.size(-1))
                flat_labels = labels.reshape(-1)
                mlm_loss = F.cross_entropy(flat_logits, flat_labels, ignore_index=-100, reduction="mean")
                mlm_loss_sum += float(mlm_loss.item())
                mlm_batches += 1

                # Teacher-based metrics
                if t is not None:
                    teacher_out = t(**inputs, output_hidden_states=True, return_dict=True)
                    teacher_logits = teacher_out.logits
                    teacher_hid = teacher_out.hidden_states

                    # Distill loss
                    d = self._masked_kl_div(student_logits, teacher_logits, labels, temperature)
                    num_masked = int(masked_positions.sum().item())
                    if num_masked > 0:
                        distill_sum += float(d.item()) * num_masked
                        distill_batches += num_masked

                    # Cosine loss
                    c = self._cosine_hidden_loss(student_hid, teacher_hid, attention_mask)
                    num_tokens = int(attention_mask.sum().item())
                    if num_tokens > 0:
                        cosine_sum += float(c.item()) * num_tokens
                        cosine_batches += num_tokens

        mlm_eval_loss = (mlm_loss_sum / mlm_batches) if mlm_batches > 0 else None
        distill_eval_loss = (distill_sum / distill_batches) if distill_batches > 0 else None
        cosine_eval_loss = (cosine_sum / cosine_batches) if cosine_batches > 0 else None

        return {
            "mlm_eval_loss": float(mlm_eval_loss) if mlm_eval_loss is not None else None,
            "distill_eval_loss": float(distill_eval_loss) if distill_eval_loss is not None else None,
            "cosine_eval_loss": float(cosine_eval_loss) if cosine_eval_loss is not None else None,
        }

# Measure full inferene time
def measure_full_pass_inference_time(
    model: nn.Module,
    tokenizer,
    texts: List[str],
    device: str = "cpu",
    max_length: int = 128,
) -> float:
    model.eval()
    model.to(device)

    def _sanitize(inputs):
        inputs = dict(inputs)
        inputs.pop("token_type_ids", None)
        return inputs

    start = time.perf_counter()
    with torch.no_grad():
        for t in texts:
            enc = tokenizer(
                t,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            enc = _sanitize(enc)
            enc = {k: v.to(device) for k, v in enc.items()}
            _ = model(**enc)
    end = time.perf_counter()
    return float(end - start)



# QUANTIZATION
# Apply quantization to reduce model size (INT8 quantization)
class QuantizationOptimizer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    # Dynamic INT8 quantization
    # Quick deployment, 4x size reduction
    def apply_dynamic_quantization(self) -> nn.Module:
        print("\n[Quantization] Applying dynamic INT8 quantization...")

        # Extract student model
        model_to_quantize = self.model.student if hasattr(self.model, 'student') else self.model

        import platform
        if platform.system() == 'Darwin':
            print("  [INFO] Running on macOS - using qnnpack backend")
            torch.backends.quantized.engine = 'qnnpack'

        try:
            quantized = torch.quantization.quantize_dynamic(
                model_to_quantize,
                {nn.Linear},  # Quantize all Linear layers
                dtype=torch.qint8
            )
            print("Dynamic quantization complete")
            return quantized
        except RuntimeError as e:
            print(f"[WARNING] Quantization not supported on this platform: {e}")
            print("[INFO] Returning original model (quantization will be skipped)")
            return model_to_quantize


# PRUNING
# Remove unnecessary weights to reduce model size and computation
class PruningOptimizer:
    def __init__(self, model):
        self.model = model

    def apply_magnitude_pruning(self, amount: float = 0.3) -> nn.Module:
        print(f"\n[Pruning] Applying magnitude pruning (amount={amount})...")

        # Extract student model
        model_to_prune = self.model.student if hasattr(self.model, 'student') else self.model

        # Collect all linear layers
        parameters_to_prune = []
        for _, module in model_to_prune.named_modules():
            if isinstance(module, nn.Linear):
                parameters_to_prune.append((module, 'weight'))

        # Apply global magnitude pruning
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=amount,
        )

        # Make pruning permanent (remove masks)
        for module, param_name in parameters_to_prune:
            prune.remove(module, param_name)

        sparsity = self.calculate_sparsity()
        print(f"Pruning complete - Sparsity: {sparsity:.1f}%")

        return model_to_prune

    # Calculate percentage of zero weights
    def calculate_sparsity(self) -> float:
        model_to_check = self.model.student if hasattr(self.model, 'student') else self.model

        total_params = 0
        zero_params = 0

        for param in model_to_check.parameters():
            total_params += param.numel()
            zero_params += torch.sum(param == 0).item()

        return (zero_params / total_params) * 100



# COMBINED OPTIMIZATION PIPELINE
# Train -> Optimize -> Benchmark -> Compare
class EdgeOptimizationPipeline:
    def __init__(self, trained_model, tokenizer, device='cpu', teacher_model_name: str = "bert-base-uncased"):
        self.original_model = trained_model
        self.tokenizer = tokenizer
        self.device = device
        self.results = {}

        # Load teacher for quality metrics
        self.teacher = AutoModelForMaskedLM.from_pretrained(teacher_model_name)

    def run_complete_optimization(self, calibration_texts: List[str] = None, eval_texts: List[str] = None):
        print("\n" + "="*70)
        print("DISTILBERT EDGE OPTIMIZATION PIPELINE")
        print("="*70)
    
        # MLM metrics + full-pass time
        if eval_texts is None:
            eval_texts = EdgeBenchmark.load_eval_sentences_from_file(
                path="../src/data/validation.txt",
                max_samples=2000,
                seed=1234
            )
            print(f"[INFO] Loaded eval_texts from validation: {len(eval_texts)} samples")
    
        # 1) Baseline
        print("\n" + "="*70)
        print("STEP 1: BASELINE BENCHMARK")
        print("="*70)
    
        baseline_bench = EdgeBenchmark(self.original_model, self.tokenizer, self.device, teacher_model=self.teacher)
        self.results["baseline"] = baseline_bench.run_all_benchmarks(eval_sentences=eval_texts)
        self.results["baseline"]["full_pass_time_s_cpu_bs1"] = measure_full_pass_inference_time(
            model=baseline_bench.m, tokenizer=self.tokenizer, texts=eval_texts, device="cpu"
        )
    
        # 2) Quantization
        print("\n" + "="*70)
        print("STEP 2: QUANTIZATION")
        print("="*70)
    
        quant_optimizer = QuantizationOptimizer(self.original_model, self.tokenizer)
        quantized_model = quant_optimizer.apply_dynamic_quantization()
    
        quant_bench = EdgeBenchmark(quantized_model, self.tokenizer, self.device, teacher_model=self.teacher)
        self.results["quantized"] = quant_bench.run_all_benchmarks(eval_sentences=eval_texts)
        self.results["quantized"]["full_pass_time_s_cpu_bs1"] = measure_full_pass_inference_time(
            model=quant_bench.m, tokenizer=self.tokenizer, texts=eval_texts, device="cpu"
        )
            
        # 3) Pruning
        print("\n" + "="*70)
        print("STEP 3: PRUNING")
        print("="*70)
    
        import copy
        model_for_pruning = copy.deepcopy(self.original_model)
        prune_optimizer = PruningOptimizer(model_for_pruning)
        pruned_model = prune_optimizer.apply_magnitude_pruning(amount=0.3)
    
        prune_bench = EdgeBenchmark(pruned_model, self.tokenizer, self.device, teacher_model=self.teacher)
        self.results["pruned"] = prune_bench.run_all_benchmarks(eval_sentences=eval_texts)
        self.results["pruned"]["full_pass_time_s_cpu_bs1"] = measure_full_pass_inference_time(
            model=prune_bench.m, tokenizer=self.tokenizer, texts=eval_texts, device="cpu"
        )
    
        # 4) Combined (Prune + Quant)
        print("\n" + "="*70)
        print("STEP 4: COMBINED (PRUNING + QUANTIZATION)")
        print("="*70)
    
        model_for_combined = copy.deepcopy(self.original_model)
    
        prune_opt2 = PruningOptimizer(model_for_combined)
        model_pruned = prune_opt2.apply_magnitude_pruning(amount=0.3)
    
        quant_opt2 = QuantizationOptimizer(model_pruned, self.tokenizer)
        combined_model = quant_opt2.apply_dynamic_quantization()
    
        combined_bench = EdgeBenchmark(combined_model, self.tokenizer, self.device, teacher_model=self.teacher)
        self.results["combined"] = combined_bench.run_all_benchmarks(eval_sentences=eval_texts)
        self.results["combined"]["full_pass_time_s_cpu_bs1"] = measure_full_pass_inference_time(
            model=combined_bench.m, tokenizer=self.tokenizer, texts=eval_texts, device="cpu"
        )
    
        self.print_comparison_table()
        self.save_optimized_models(quantized_model, pruned_model, combined_model)
    
        return self.results


    def print_comparison_table(self):
        """Print comprehensive comparison table"""
        print("\n" + "="*70)
        print("OPTIMIZATION RESULTS COMPARISON")
        print("="*70)

        configs = ['baseline', 'quantized', 'pruned', 'combined']
        metrics = [
            ('model_size_mb', 'Model Size (MB)', '{:.2f}'),
            ('total_params', 'Parameters', '{:,}'),
            ('cold_start_ms', 'Cold Start (ms)', '{:.2f}'),
            ('throughput_samples_per_sec', 'Throughput (samples/s)', '{:.2f}'),
            ('peak_memory_mb', 'Peak Memory (MB)', '{:.2f}'),
            ('mlm_eval_loss', 'MLM Eval Loss', '{:.4f}'),
            ('distill_eval_loss', 'Distill Eval Loss', '{:.4f}'),
            ('cosine_eval_loss', 'Cosine Eval Loss', '{:.4f}'),
            ('full_pass_time_s_cpu_bs1', 'Full-pass Time (s) CPU bs=1', '{:.2f}'),
        ]

        # Print header
        print(f"\n{'Metric':<25} {'Baseline':<15} {'Quantized':<15} {'Pruned':<15} {'Combined':<15}")
        print("-" * 85)

        # Print each metric with N/A handling
        for metric_key, metric_name, fmt in metrics:
            row = f"{metric_name:<25}"
            for config in configs:
                if config not in self.results or metric_key not in self.results[config]:
                    row += f"{'N/A':<15}"
                    continue

                val = self.results[config][metric_key]
                if val is None:
                    row += f"{'N/A':<15}"
                else:
                    row += f"{fmt.format(val):<15}"
            print(row)

    # Save Optimized models
    def save_optimized_models(self, quantized_model, pruned_model, combined_model):
        save_dir = "optimized_models"
        os.makedirs(save_dir, exist_ok=True)

        print(f"\n[Saving] Models to {save_dir}/...")

        torch.save(quantized_model.state_dict(), f"{save_dir}/distilbert_quantized.pt")
        torch.save(pruned_model.state_dict(), f"{save_dir}/distilbert_pruned.pt")
        torch.save(combined_model.state_dict(), f"{save_dir}/distilbert_combined.pt")

        print("All optimized models saved")


# MAIN
def optimize_trained_model(checkpoint_path: str = None):
    print("Loading trained model...")

    # Load model (either from checkpoint or fresh)
    if checkpoint_path and os.path.exists(checkpoint_path):
        model = torch.load(checkpoint_path, map_location="cpu")
        print(f"Loaded model from {checkpoint_path}")
    else:
        model = DistilBertStudent(teacher_model_name="bert-base-uncased")
        model.initialize_from_teacher()
        print("Created fresh model (not trained - for testing)")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)

    # Run optimization pipeline
    pipeline = EdgeOptimizationPipeline(model, tokenizer, device='cpu')
    results = pipeline.run_complete_optimization()

    return results


if __name__ == "__main__":
    # Run the complete optimization
    results = optimize_trained_model()

    print("\n" + "="*70)
    print("Edge Optimization Complete")
    print("="*70)
    print("\nCheck 'optimized_models/' directory for saved models")
    print("Use these optimized models for edge deployment")
