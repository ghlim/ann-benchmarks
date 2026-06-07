import h5py
import os
import json

def read_results():
    results = []
    results_dir = 'results/sift-128-euclidean/10/hnswlib'
    
    for filename in os.listdir(results_dir):
        if filename.endswith('.hdf5'):
            filepath = os.path.join(results_dir, filename)
            with h5py.File(filepath, 'r') as f:
                attrs = dict(f.attrs)
                
                # Extract parameters from filename
                parts = filename.replace('.hdf5', '').split('_')
                m_value = int(parts[2])
                ef_construction = int(parts[4])
                ef_query = int(parts[5])
                
                # Get metrics
                recall = float(attrs.get('mean_recall', 0))
                qps = float(attrs.get('mean_qps', 0))
                build_time = float(attrs.get('build_time', 0))
                index_size = float(attrs.get('index_size', 0))
                
                results.append({
                    'M': m_value,
                    'efConstruction': ef_construction,
                    'efQuery': ef_query,
                    'recall': recall,
                    'qps': qps,
                    'build_time': build_time,
                    'index_size_kb': index_size / 1024
                })
    
    return sorted(results, key=lambda x: (x['M'], x['efQuery']))

# Read and display results
results = read_results()

print("\n" + "="*100)
print("HNSWLIB BENCHMARK RESULTS - SIFT-128-Euclidean Dataset")
print("Dataset: 1M vectors, 128 dimensions, Euclidean distance")
print("="*100)

# Group by M value
m_values = sorted(set(r['M'] for r in results))

for m in m_values:
    m_results = [r for r in results if r['M'] == m]
    print(f"\n{'='*100}")
    print(f"M = {m} (efConstruction = 500)")
    print(f"Build time: {m_results[0]['build_time']:.2f} seconds")
    print(f"Index size: {m_results[0]['index_size_kb']:.2f} KB")
    print(f"{'='*100}")
    print(f"{'efQuery':<12} {'Recall':<12} {'QPS':<15} {'Query Time (ms)':<15}")
    print("-"*100)
    
    for r in m_results:
        query_time_ms = 1000.0 / r['qps'] if r['qps'] > 0 else 0
        print(f"{r['efQuery']:<12} {r['recall']:<12.4f} {r['qps']:<15,.2f} {query_time_ms:<15.4f}")

# Summary table - best configurations
print(f"\n\n{'='*100}")
print("BEST CONFIGURATIONS SUMMARY")
print(f"{'='*100}")
print(f"{'Target Recall':<15} {'Best M':<10} {'efQuery':<12} {'QPS':<15} {'Query Time (ms)':<15}")
print("-"*100)

recall_targets = [0.90, 0.95, 0.99, 0.999, 1.0]
for target in recall_targets:
    # Find configurations that meet this recall
    candidates = [r for r in results if r['recall'] >= target]
    if candidates:
        # Pick the one with highest QPS
        best = max(candidates, key=lambda x: x['qps'])
        query_time_ms = 1000.0 / best['qps'] if best['qps'] > 0 else 0
        print(f"{target:<15.3f} {best['M']:<10} {best['efQuery']:<12} {best['qps']:<15,.2f} {query_time_ms:<15.4f}")

print(f"\n{'='*100}")
print("SPEED VS ACCURACY TRADEOFFS")
print(f"{'='*100}")
print("\nFastest configuration (regardless of recall):")
fastest = max(results, key=lambda x: x['qps'])
print(f"  M={fastest['M']}, efQuery={fastest['efQuery']}: {fastest['qps']:,.2f} QPS, Recall={fastest['recall']:.4f}")

print("\nHighest recall configuration:")
highest_recall = max(results, key=lambda x: x['recall'])
print(f"  M={highest_recall['M']}, efQuery={highest_recall['efQuery']}: Recall={highest_recall['recall']:.4f}, {highest_recall['qps']:,.2f} QPS")

print("\nBest balanced configuration (recall >= 0.99, highest QPS):")
balanced = [r for r in results if r['recall'] >= 0.99]
if balanced:
    best_balanced = max(balanced, key=lambda x: x['qps'])
    print(f"  M={best_balanced['M']}, efQuery={best_balanced['efQuery']}: Recall={best_balanced['recall']:.4f}, {best_balanced['qps']:,.2f} QPS")

print(f"\n{'='*100}\n")
