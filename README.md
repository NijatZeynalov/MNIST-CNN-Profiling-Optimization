# MNIST CNN Profiling & Optimization

The goal is to identify bottlenecks in the training and inference pipeline and apply optimizations to improve performance while maintaining accuracy.

## Project Steps

1. **Baseline Model**  
   Trained a basic CNN on MNIST with default settings (no optimizations).

2. **Profiling**  
   Used PyTorch Profiler to analyze the baseline model and identify performance bottlenecks.

3. **Optimized Model**  
   Applied targeted optimizations based on profiling results, such as:  
   - Switching optimizer from Adadelta to Adam  
   - Using mixed precision training (automatic mixed precision)  
   - Improving data loading with `num_workers`, `pin_memory`, and `persistent_workers`  
   - Async data transfer and CUDA improvements

4. **Performance Comparison**  
   Compared training time, inference time, and accuracy between the baseline and optimized models.

## Requirements

- Python 3.7+
- PyTorch
- torchvision
- matplotlib (optional, for visualization)
