# Testing-Kernels
# Custom CUDA Softmax Kernel for PyTorch

This project implements a custom CUDA kernel for row-wise softmax computation and integrates it with PyTorch as a C++ extension.

## 🚀 Features

- **Custom CUDA Implementation**: Hand-optimized CUDA kernels for row-wise softmax
- **Numerical Stability**: Uses the standard max subtraction trick to prevent overflow
- **PyTorch Integration**: Seamless integration with PyTorch tensors and autograd
- **Multiple Kernel Variants**: Optimized versions for different matrix sizes
- **Comprehensive Testing**: Correctness tests and performance benchmarks
- **Google Colab Ready**: Works out-of-the-box in CUDA-enabled Colab environments

## 📁 Files Structure

```
├── softmax.cu              # CUDA kernel implementation
├── compile_and_test.py     # Main compilation and testing script  
├── benchmark.py            # Detailed performance benchmark
└── README.md              # This file
```

## 🔧 Prerequisites

- CUDA-capable GPU
- PyTorch with CUDA support
- NVIDIA CUDA Toolkit (usually comes with PyTorch)
- Python packages: `torch`, `numpy`, `matplotlib` (for benchmarks)

## 🎯 Quick Start

### Option 1: Google Colab (Recommended)

1. **Create a new Colab notebook** and ensure GPU runtime is selected:
   ```python
   # Check CUDA availability
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"Device: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")
   ```

2. **Upload the files** to your Colab session or create them in cells:
   ```python
   # Create softmax.cu file
   %%writefile softmax.cu
   [paste the CUDA code here]
   ```

3. **Run the main test script**:
   ```python
   !python compile_and_test.py
