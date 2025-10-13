# HiDeNN - DeepXDE Project

A project for solving differential equations using Physics-Informed Neural Networks (PINNs) with DeepXDE.

## 🚀 Quick Start

1. **Run the main example:**
   ```bash
   python main.py
   ```

2. **Test your installation:**
   ```bash
   python test_pytorch.py
   ```

## 📦 What's Included

- **DeepXDE** - Physics-informed neural networks
- **PyTorch** - Machine learning backend
- **NumPy, SciPy, Matplotlib** - Scientific computing stack

## 🔧 Environment

- **Python**: 3.13
- **Backend**: PyTorch
- **Environment Variable**: `DDE_BACKEND=pytorch`

## 📚 Examples

The `main.py` file contains a simple PINN example that solves:
- **1D Poisson equation**: d²u/dx² = -1
- **Boundary conditions**: u(0) = u(1) = 0

## 🎯 Next Steps

1. **Explore DeepXDE examples**: https://github.com/lululxvi/deepxde/tree/master/examples
2. **Read documentation**: https://deepxde.readthedocs.io/
3. **Start coding your own PINNs!**

## 📁 Project Structure

```
HiDeNN/
├── main.py              # Main DeepXDE example
├── test_pytorch.py      # Installation test
├── requirements.txt     # Dependencies
└── README.md           # This file
```

## 🆘 Troubleshooting

If you encounter issues:
1. **Check backend**: Make sure `DDE_BACKEND=pytorch` is set
2. **Restart terminal**: Sometimes needed after installation
3. **Reinstall packages**: `python -m pip install --upgrade deepxde`

Happy coding! 🎉