# run_complete_pipeline.py

import torch
import os, sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, SRC_DIR)

from src.train import train

from edge_optimizer import EdgeOptimizationPipeline, EdgeBenchmark
from transformers import AutoTokenizer
import json
import matplotlib.pyplot as plt
import numpy as np

# End-to-end pipeline: Training, Optimization, Analysis
class CompletePipeline:
    def __init__(self, output_dir="results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.all_results = {}

    # Train DistilBERT
    def train_model(self, quick_test=False):
        print("\n" + "="*70)
        print("1. Training DistilBERT (Triple Loss)")
        print("="*70)
        
        if quick_test:
            print("[Quick Test Mode]")
            trained_model = train(
                teacher_model_name="bert-base-uncased",
                batch_size=4,
                subset_size=1000,
                lr=5e-5,
                epochs=1,
                temperature=2.0,
                alpha_mlm=1.0,
                beta_distill=1.0,
                gamma_cos=1.0,
                max_steps=100
            )
        else:
            print("[Full Training]")
            trained_model = train(
                teacher_model_name="bert-base-uncased",
                batch_size=8,
                subset_size=10000,
                lr=5e-5,
                epochs=3,
                temperature=2.0,
                alpha_mlm=1.0,
                beta_distill=1.0,
                gamma_cos=1.0,
                max_steps=2000
            )
        
        # Save trained model
        model_path = os.path.join(self.output_dir, "trained_distilbert.pt")
        torch.save(trained_model.state_dict(), model_path)
        print(f"\nModel saved to {model_path}")
        
        return trained_model

        # Apply edge optimizations
    def optimize_for_edge(self, trained_model):
        print("\n" + "="*70)
        print("2. Edge Optimization")
        print("="*70)
        

        tokenizer = AutoTokenizer.from_pretrained(
            "bert-base-uncased",
            use_fast=True
        )

        # Run optimization pipeline
        pipeline = EdgeOptimizationPipeline(
            trained_model, 
            tokenizer, 
            device='cpu'
        )
        
        results = pipeline.run_complete_optimization()        
        self.all_results = results
        self.save_results(results)
        
        return results

    # Generate charts
    def generate_visualizations(self):
        print("\n" + "="*70)
        print("3. Generating Visualizations")
        print("="*70)
        
        # Chart 1: Model Size Comparison
        self.plot_model_sizes()
        
        # Chart 2: Inference Latency
        self.plot_latency_comparison()
        
        # Chart 3: Size-Performance Trade-off
        self.plot_tradeoff()
        
        print(f"\nAll visualizations saved to {self.output_dir}/")

    # Bar chart - Model Sizes
    def plot_model_sizes(self):
        configs = ['baseline', 'quantized', 'pruned', 'combined']
        sizes = [self.all_results[c]['model_size_mb'] for c in configs]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(configs, sizes, color=['#3498db', '#2ecc71', '#f39c12', '#e74c3c'])
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f} MB',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_ylabel('Model Size (MB)', fontsize=12, fontweight='bold')
        ax.set_title('DistilBERT: Model Size Comparison', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'model_sizes.png'), dpi=300)
        print("model_sizes.png")
        plt.close()

    # Bar chart - Inference Latency
    def plot_latency_comparison(self):
        configs = ['baseline', 'quantized', 'pruned', 'combined']
        
        latencies = []
        for c in configs:
            latency_keys = [k for k in self.all_results[c].keys() if 'latency' in k]
            latencies.append(self.all_results[c][latency_keys[0]])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(configs, latencies, color=['#3498db', '#2ecc71', '#f39c12', '#e74c3c'])
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f} ms',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_ylabel('Inference Time (ms)', fontsize=12, fontweight='bold')
        ax.set_title('DistilBERT: Inference Latency Comparison', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'latency_comparison.png'), dpi=300)
        print("  latency_comparison.png")
        plt.close()

    # Scatter plot - Size vs Performance Trade-Off
    def plot_tradeoff(self):
        configs = ['baseline', 'quantized', 'pruned', 'combined']
        sizes = [self.all_results[c]['model_size_mb'] for c in configs]
        
        throughputs = [self.all_results[c]['throughput_samples_per_sec'] for c in configs]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
        for i, (config, size, throughput) in enumerate(zip(configs, sizes, throughputs)):
            ax.scatter(size, throughput, s=200, c=colors[i], label=config, alpha=0.7, edgecolors='black')
            ax.annotate(config, (size, throughput), 
                       xytext=(10, 10), textcoords='offset points',
                       fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Model Size (MB)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Throughput (samples/sec)', fontsize=12, fontweight='bold')
        ax.set_title('Size-Performance Trade-off', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'tradeoff.png'), dpi=300)
        print("  tradeoff.png")
        plt.close()

    # Save results (JSON)
    def save_results(self, results):
        results_path = os.path.join(self.output_dir, "optimization_results.json")
    
        json_results = {}
        for config, metrics in results.items():
            json_results[config] = {k: float(v) if isinstance(v, (int, float, np.integer, np.floating)) else v 
                                   for k, v in metrics.items()}
        
        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"\nResults saved to {results_path}")


    # Run pipline
    def run_complete_pipeline(self, quick_test=False):
        print("\n" + "="*70)
        print("Complete DistilBERT Edge Deployment Pipeline")
        print("="*70)
        print(f"Output directory: {self.output_dir}")
        
        # Train
        trained_model = self.train_model(quick_test=quick_test)
        
        # Optimize
        results = self.optimize_for_edge(trained_model)
        
        # Visualize
        self.generate_visualizations()
        
        print("\n" + "="*70)
        print("Pipeline Complete.")
        print("="*70)
        print(f"\nAll results saved to: {self.output_dir}/")
        
        return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Complete DistilBERT pipeline')
    parser.add_argument('--quick-test', action='store_true', 
                       help='Run quick test with minimal data (for debugging)')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    pipeline = CompletePipeline(output_dir=args.output_dir)
    results = pipeline.run_complete_pipeline(quick_test=args.quick_test)
    
    print("\nComplete.")
