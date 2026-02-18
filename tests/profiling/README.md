# ⚡ Profiling Scripts

This folder contains performance profiling and benchmarking scripts.

---

## 📁 Files (1 script)

### **`profile_tcn_detail.py`**
**Purpose:** Detailed performance profiling of TCN implementations

**Features:**
- Profiles TCN forward passes
- Measures memory usage
- Identifies performance bottlenecks
- Provides timing breakdowns

**Size:** 2.4K

---

## 🎯 Usage

Profiling scripts are used to:
1. Measure execution time of specific operations
2. Identify performance bottlenecks
3. Compare different implementations
4. Monitor memory consumption
5. Validate optimization improvements

To run profiling:
```bash
cd /workspaces/multi_modal_drl_continuous/adaptive_portfolio_rl
python tests/profiling/profile_tcn_detail.py
```

---

## 📊 Output

Profiling scripts typically provide:
- Execution time measurements
- Memory usage statistics
- Line-by-line performance data
- Comparison tables
- Bottleneck identification

---

## 🔧 Related Scripts

For comprehensive benchmarking, see the root directory:
- **`benchmark_tcn_performance.py`** - TCN performance benchmarks
- **`benchmark_stateful_vs_stateless.py`** - Stateful vs Stateless comparison

---

## 💡 When to Use

Run profiling scripts when:
- Optimizing code performance
- Comparing implementation alternatives
- Diagnosing slow operations
- Validating speedup claims
- Before/after optimization comparisons

---

**Last Updated:** October 19, 2025  
**Category:** Performance Analysis
