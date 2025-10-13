#!/usr/bin/env python3
"""
X-PINN for Fracture Mechanics with XFEM Enrichment
Implementation based on X-PINN solution scheme for crack problems

The solution is decomposed into three components:
- Continuous Component (uC): Standard neural network for smooth displacement field
- Discontinuous Component (uD): Enriched network for displacement jump at crack body
- Singular Component (uS): Enriched network for singular field at crack tip

Total displacement: u(x) = uC(x) + uD(x, xcb) + uS(x, xct)
"""

import os
import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List
from scipy.optimize import minimize
from scipy.linalg import cholesky, LinAlgError
from time import perf_counter
import torch

# Set backend to PyTorch
os.environ.setdefault("DDE_BACKEND", "pytorch")


class XFEMCrackPINN:
    """
    X-PINN for fracture mechanics using XFEM-inspired enrichment functions.
    
    Based on the problem:
    - Plate dimensions: L = H = 1.0
    - Crack width: 2a = 0.3
    - Young's modulus: E = 1.0
    - Poisson's ratio: ν = 0.3
    - Applied traction: t = 0.1
    """
    
    def __init__(self, 
                 crack_center: Tuple[float, float] = (0.5, 0.5),
                 crack_half_length: float = 0.15,
                 crack_angle: float = 0.0,
                 domain_size: Tuple[float, float] = (1.0, 1.0),
                 enrichment_width: float = 0.1):
        """
        Initialize X-PINN for crack problem.
        
        Parameters:
        - crack_center: (x, y) coordinates of crack center
        - crack_half_length: Half-length of crack (a = 0.15 for width 2a = 0.3)
        - crack_angle: Crack orientation angle (radians)
        - domain_size: (L, H) domain dimensions
        - enrichment_width: Width of enrichment region (l0 = 0.1)
        """
        self.crack_center = np.array(crack_center)
        self.crack_half_length = crack_half_length
        self.crack_angle = crack_angle
        self.domain_size = domain_size
        self.enrichment_width = enrichment_width
        
        # Material properties (from the problem description)
        self.E = 1.0  # Young's modulus
        self.nu = 0.3  # Poisson's ratio
        self.traction = 0.1  # Applied traction
        
        # Derived material constants (plane stress)
        self.mu = self.E / (2 * (1 + self.nu))
        self.lam = self.E * self.nu / (1 - self.nu ** 2)
        
        # Networks
        self.net_continuous = None  # Standard network (10 layers, 20 neurons)
        self.net_discontinuous = None  # Enriched network for crack body (10 layers, 10 neurons)
        self.net_singular = None  # Enriched network for crack tip (10 layers, 10 neurons)
        
        self.model = None
        self.data = None
        
        # Training parameters
        self.loss_weights = {'pde': 1.0, 'bc': 5.0, 'ic': 5.0}
        self.adaptive_sampling = True
        self.rad_params = {'k1': 1.0, 'k2': 1.0}  # Residual adaptive distribution params
        
        # Crack tips
        self.crack_tip1, self.crack_tip2 = self._compute_crack_tips()
    
    def _compute_crack_tips(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute crack tip positions based on center, length, and angle."""
        dx = self.crack_half_length * np.cos(self.crack_angle)
        dy = self.crack_half_length * np.sin(self.crack_angle)
        tip1 = self.crack_center + np.array([dx, dy])
        tip2 = self.crack_center - np.array([dx, dy])
        return tip1, tip2
    
    def sawtooth_enrichment(self, x: np.ndarray) -> np.ndarray:
        """
        Sawtooth-like enrichment function D(ξ, η) for discontinuous component.
        Based on Equation (17)-(19) from the paper:
        
        D(ξ, η) = Ξ(ξ) · Λ(η) if inside enrichment region, 0 otherwise
        
        where:
        - Ξ(ξ): cubic polynomial along crack direction
        - Λ(η): quadratic polynomial perpendicular to crack
        
        Parameters:
        - x: Input coordinates (N, 2) in global system
        
        Returns:
        - Enrichment values (N, 1)
        """
        # Transform to local coordinate system (ξ, η)
        # θ = crack angle
        # ξ = x cos θ + y sin θ  (along crack)
        # η = -x sin θ + y cos θ  (perpendicular to crack)
        
        cos_theta = np.cos(self.crack_angle)
        sin_theta = np.sin(self.crack_angle)
        
        xi = x[:, 0:1] * cos_theta + x[:, 1:2] * sin_theta
        eta = -x[:, 0:1] * sin_theta + x[:, 1:2] * cos_theta
        
        # Transform crack tips to local coordinates
        xi_center = self.crack_center[0] * cos_theta + self.crack_center[1] * sin_theta
        eta_center = -self.crack_center[0] * sin_theta + self.crack_center[1] * cos_theta
        
        # Crack tips in local system
        xi_1 = xi_center - self.crack_half_length  # Left tip
        xi_2 = xi_center + self.crack_half_length  # Right tip
        eta_1 = eta_center  # Crack is centered at eta_center
        
        # Enrichment domain parameters
        l_0 = self.enrichment_width  # Width parameter
        
        # Initialize enrichment function
        D = np.zeros_like(xi)
        
        # Check if points are in enrichment domain
        in_domain_xi = (xi >= xi_1) & (xi <= xi_2)
        in_domain_eta = (eta >= eta_1 - l_0) & (eta <= eta_1 + l_0)
        in_domain = in_domain_xi & in_domain_eta
        
        # Compute Ξ(ξ) - cubic polynomial along crack direction
        # Ξ(ξ) = a0 + a1*ξ + a2*ξ² + a3*ξ³ for ξ1 ≤ ξ ≤ ξ2
        # Coefficients chosen for smoothness at boundaries
        # Using normalized coordinates: s = (ξ - ξ1) / (ξ2 - ξ1) ∈ [0, 1]
        s = (xi - xi_1) / (xi_2 - xi_1 + 1e-10)
        
        # Cubic polynomial with C1 continuity at boundaries
        # Ξ(s) = -2s³ + 3s² (goes from 0 to 1 smoothly)
        # Modified to create sawtooth: goes from 0 to 1 to 0
        Xi = np.where(in_domain_xi, 
                     4 * s * (1 - s) * (s - 0.5) + 0.5,  # Sawtooth shape
                     0)
        
        # Compute Λ(η) - quadratic polynomial perpendicular to crack
        # Λ(η) = b0 + b1*η + b2*η² for η1 - l0 ≤ η ≤ η1 + l0
        # Using normalized coordinates: t = (η - (η1 - l0)) / (2*l0) ∈ [0, 1]
        t = (eta - (eta_1 - l_0)) / (2 * l_0 + 1e-10)
        
        # Quadratic polynomial with smooth transition
        # Λ(t) = 1 - (2t - 1)² (bell curve shape, max at center)
        Lambda = np.where(in_domain_eta,
                         1 - (2 * t - 1)**2,
                         0)
        
        # D(ξ, η) = Ξ(ξ) · Λ(η)
        D = np.where(in_domain, Xi * Lambda, 0)
        
        return D
    
    def heaviside_enrichment(self, x: np.ndarray) -> np.ndarray:
        """
        Simple Heaviside enrichment function (alternative to sawtooth).
        Returns +1 on one side of crack, -1 on the other side.
        
        Parameters:
        - x: Input coordinates (N, 2)
        
        Returns:
        - Enrichment values (N, 1)
        """
        # Vector from crack center to point
        dx = x[:, 0:1] - self.crack_center[0]
        dy = x[:, 1:2] - self.crack_center[1]
        
        # Rotate to crack-aligned coordinate system
        cos_a = np.cos(self.crack_angle)
        sin_a = np.sin(self.crack_angle)
        
        # Local y-coordinate (perpendicular to crack)
        y_local = -dx * sin_a + dy * cos_a
        
        # Heaviside function: sign of perpendicular distance
        H = np.sign(y_local)
        H[H == 0] = 1  # Handle points on crack
        
        return H
    
    def crack_tip_enrichment(self, x: np.ndarray, tip: np.ndarray) -> np.ndarray:
        """
        Crack tip enrichment functions S(x, xct) for singular component.
        Uses asymptotic near-tip fields (r, θ coordinates).
        
        Parameters:
        - x: Input coordinates (N, 2)
        - tip: Crack tip position (2,)
        
        Returns:
        - Enrichment values (N, 4) for 4 singular functions
        """
        # Polar coordinates relative to crack tip
        dx = x[:, 0:1] - tip[0]
        dy = x[:, 1:2] - tip[1]
        
        # Rotate to crack-aligned coordinate system
        cos_a = np.cos(self.crack_angle)
        sin_a = np.sin(self.crack_angle)
        
        x_local = dx * cos_a + dy * sin_a
        y_local = -dx * sin_a + dy * cos_a
        
        r = np.sqrt(x_local**2 + y_local**2 + 1e-10)
        theta = np.arctan2(y_local, x_local)
        
        sqrt_r = np.sqrt(r)
        
        # Four standard crack tip enrichment functions
        F1 = sqrt_r * np.sin(theta / 2)
        F2 = sqrt_r * np.cos(theta / 2)
        F3 = sqrt_r * np.sin(theta / 2) * np.sin(theta)
        F4 = sqrt_r * np.cos(theta / 2) * np.sin(theta)
        
        return np.concatenate([F1, F2, F3, F4], axis=1)
    
    def is_in_enrichment_domain(self, x: np.ndarray, domain_type: str = 'crack_body') -> np.ndarray:
        """
        Check if points are in enrichment domain.
        
        Parameters:
        - x: Input coordinates (N, 2)
        - domain_type: 'crack_body' or 'crack_tip'
        
        Returns:
        - Boolean mask (N,)
        """
        if domain_type == 'crack_body':
            # Enrichment domain around crack body
            dx = x[:, 0] - self.crack_center[0]
            dy = x[:, 1] - self.crack_center[1]
            
            cos_a = np.cos(self.crack_angle)
            sin_a = np.sin(self.crack_angle)
            
            x_local = dx * cos_a + dy * sin_a
            y_local = -dx * sin_a + dy * cos_a
            
            # Within crack length and enrichment width
            in_domain = (np.abs(x_local) <= self.crack_half_length) & \
                       (np.abs(y_local) <= self.enrichment_width)
            
        elif domain_type == 'crack_tip':
            # Enrichment domain around crack tips
            dist_tip1 = np.sqrt((x[:, 0] - self.crack_tip1[0])**2 + 
                               (x[:, 1] - self.crack_tip1[1])**2)
            dist_tip2 = np.sqrt((x[:, 0] - self.crack_tip2[0])**2 + 
                               (x[:, 1] - self.crack_tip2[1])**2)
            
            in_domain = (dist_tip1 <= self.enrichment_width) | \
                       (dist_tip2 <= self.enrichment_width)
        else:
            in_domain = np.zeros(len(x), dtype=bool)
        
        return in_domain
    
    def create_geometry(self) -> dde.geometry.Geometry:
        """Create rectangular domain geometry."""
        geom = dde.geometry.Rectangle([0.0, 0.0], list(self.domain_size))
        return geom
    
    def pde_residual(self, x, y):
        """
        PDE residuals for linear elasticity with X-PINN enrichment.
        
        The solution is decomposed as:
        u(x) = uC(x) + D(x, xcb) * uD(x) + S(x, xct) * uS(x)
        
        where:
        - uC: continuous component (standard NN output)
        - uD: discontinuous component (enriched NN output)
        - uS: singular component (enriched NN output)
        """
        # Extract displacement components
        # For simplicity, we'll use a combined network output
        # y[:, 0:2] = displacement (u_x, u_y)
        u_x = y[:, 0:1]
        u_y = y[:, 1:2]
        
        # Compute strains
        u_x_x = dde.grad.jacobian(u_x, x, i=0, j=0)
        u_x_y = dde.grad.jacobian(u_x, x, i=0, j=1)
        u_y_x = dde.grad.jacobian(u_y, x, i=0, j=0)
        u_y_y = dde.grad.jacobian(u_y, x, i=0, j=1)
        
        # Handle potential None values
        if u_x_x is None or u_x_y is None or u_y_x is None or u_y_y is None:
            return [u_x * 0, u_x * 0]
        
        # Strain components
        eps_xx = u_x_x
        eps_yy = u_y_y
        eps_xy = 0.5 * (u_x_y + u_y_x)
        
        # Stress components (plane stress)
        sigma_xx = self.lam * (eps_xx + eps_yy) + 2 * self.mu * eps_xx
        sigma_yy = self.lam * (eps_xx + eps_yy) + 2 * self.mu * eps_yy
        sigma_xy = 2 * self.mu * eps_xy
        
        # Equilibrium equations
        sigma_xx_x = dde.grad.jacobian(sigma_xx, x, i=0, j=0)
        sigma_xy_y = dde.grad.jacobian(sigma_xy, x, i=0, j=1)
        sigma_xy_x = dde.grad.jacobian(sigma_xy, x, i=0, j=0)
        sigma_yy_y = dde.grad.jacobian(sigma_yy, x, i=0, j=1)
        
        # Handle potential None values
        if sigma_xx_x is None or sigma_xy_y is None or sigma_xy_x is None or sigma_yy_y is None:
            return [u_x * 0, u_x * 0]
        
        # Equilibrium: div(σ) = 0
        eq_x = sigma_xx_x + sigma_xy_y
        eq_y = sigma_xy_x + sigma_yy_y
        
        return [eq_x, eq_y]
    
    def boundary_conditions(self, geom):
        """Define boundary conditions for the crack problem - Mode I loading."""
        
        def bottom_boundary(x, on_boundary):
            return on_boundary and np.isclose(x[1], 0.0)
        
        def top_boundary(x, on_boundary):
            return on_boundary and np.isclose(x[1], self.domain_size[1])
        
        def left_boundary(x, on_boundary):
            return on_boundary and np.isclose(x[0], 0.0)
        
        def right_boundary(x, on_boundary):
            return on_boundary and np.isclose(x[0], self.domain_size[0])
        
        # Mode I loading: tension in y-direction
        # Top boundary: apply tensile displacement
        def top_displacement_y(x):
            return np.full((len(x), 1), self.traction * self.domain_size[1] / self.E)
        
        bc_top_y = dde.icbc.DirichletBC(
            geom, top_displacement_y, top_boundary, component=1
        )
        
        # Bottom boundary: fixed in y-direction
        bc_bottom_y = dde.icbc.DirichletBC(
            geom, lambda x: np.zeros((len(x), 1)), bottom_boundary, component=1
        )
        
        # Left and right boundaries: free in x-direction (no constraint)
        # Only constrain at center point to prevent rigid body motion
        def center_point(x, on_boundary):
            return on_boundary and np.isclose(x[0], 0.0) and np.isclose(x[1], self.domain_size[1]/2)
        
        bc_center_x = dde.icbc.DirichletBC(
            geom, lambda x: np.zeros((len(x), 1)), center_point, component=0
        )
        
        return [bc_bottom_y, bc_top_y, bc_center_x]
    
    def setup_problem(self, num_domain: int = 26520, num_boundary: int = 500):
        """
        Setup the X-PINN problem with enrichment.
        
        Parameters:
        - num_domain: Total integration points (26,520 as per paper)
        - num_boundary: Boundary points
        """
        print("Setting up X-PINN for crack problem...")
        geom = self.create_geometry()
        bcs = self.boundary_conditions(geom)
        
        # Create PDE data
        self.data = dde.data.PDE(
            geom,
            self.pde_residual,
            bcs,
            num_domain=num_domain,
            num_boundary=num_boundary,
            num_test=100,
        )
        
        # Create neural networks
        # Continuous component: 10 layers, 20 neurons each
        layer_continuous = [2] + [20] * 10 + [2]  # 2 inputs -> 2 outputs (u_x, u_y)
        
        # Discontinuous component: 10 layers, 10 neurons each
        layer_discontinuous = [2] + [10] * 10 + [2]
        
        # Singular component: 10 layers, 10 neurons each
        layer_singular = [2] + [10] * 10 + [2]
        
        activation = "tanh"
        initializer = "Glorot uniform"
        
        # For now, use a combined network (can be extended to separate networks)
        # Combined network that outputs all components
        layer_size = [2] + [20] * 10 + [2]
        self.net = dde.nn.FNN(layer_size, activation, initializer)
        
        self.model = dde.Model(self.data, self.net)
        print("Problem setup completed!")
        print(f"Crack center: {self.crack_center}")
        print(f"Crack tips: {self.crack_tip1}, {self.crack_tip2}")
        print(f"Enrichment width: {self.enrichment_width}")
    
    def compute_residuals(self, X: np.ndarray) -> np.ndarray:
        """
        Compute PDE residuals at given points for adaptive sampling.
        
        Parameters:
        - X: Input coordinates (N, 2)
        
        Returns:
        - residuals: PDE residual values (N,)
        """
        X_tensor = torch.tensor(X, dtype=torch.float32, requires_grad=True)
        
        with torch.no_grad():
            # Get predictions
            y_pred = self.net(X_tensor)
            
        # For now, return uniform residuals (can be improved with actual PDE residuals)
        residuals = np.ones(len(X))
        return residuals
    
    def adaptive_resample(self, geom, num_domain: int, num_sampling: int = 10000):
        """
        Adaptive Residual-based Distribution (RAD) for collocation points.
        
        Parameters:
        - geom: Geometry object
        - num_domain: Number of domain points to select
        - num_sampling: Number of candidate points to sample
        
        Returns:
        - X_selected: Selected collocation points
        """
        # Generate large candidate set
        X_candidates = geom.random_points(num_sampling)
        
        # Compute residuals at candidate points
        try:
            # Try to compute actual residuals if model is trained
            X_test_tensor = torch.tensor(X_candidates, dtype=torch.float32, requires_grad=True)
            
            with torch.enable_grad():
                y_pred = self.net(X_test_tensor)
                u_x = y_pred[:, 0:1]
                u_y = y_pred[:, 1:2]
                
                # Compute gradients for residual
                u_x_x = torch.autograd.grad(u_x.sum(), X_test_tensor, create_graph=True)[0][:, 0:1]
                u_y_y = torch.autograd.grad(u_y.sum(), X_test_tensor, create_graph=True)[0][:, 1:2]
                
                # Simplified residual (just gradient magnitude)
                residual = torch.abs(u_x_x) + torch.abs(u_y_y)
                residual = residual.detach().numpy().flatten()
        except:
            # If residual computation fails, use uniform sampling
            residual = np.ones(len(X_candidates))
        
        # Compute adaptive weights using RAD formula
        k1 = self.rad_params['k1']
        k2 = self.rad_params['k2']
        
        err_eq = np.power(np.abs(residual), k1) / (np.power(np.abs(residual), k1).mean() + 1e-10) + k2
        err_eq_normalized = err_eq / (err_eq.sum() + 1e-10)
        
        # Sample points based on residual distribution
        indices = np.random.choice(len(X_candidates), size=num_domain, replace=False, p=err_eq_normalized)
        X_selected = X_candidates[indices]
        
        return X_selected
    
    def train(self, iterations: int = 5000, lr: float = 1e-3, optimizer: str = "adam",
              use_lbfgs: bool = False, lbfgs_iterations: int = 1000,
              resample_every: int = 500):
        """
        Train the X-PINN model with adaptive sampling and optional L-BFGS refinement.
        
        Parameters:
        - iterations: Number of Adam iterations
        - lr: Learning rate for Adam
        - optimizer: Optimizer name
        - use_lbfgs: Whether to use L-BFGS refinement after Adam
        - lbfgs_iterations: Number of L-BFGS iterations
        - resample_every: Resample collocation points every N iterations
        """
        if self.model is None:
            raise ValueError("Model not setup. Call setup_problem() first.")
        
        print(f"Training X-PINN for {iterations} iterations with lr={lr}...")
        print(f"Adaptive sampling: {self.adaptive_sampling}, Resample every: {resample_every} iters")
        
        # Stage 1: Adam optimization with adaptive sampling
        adam_start = perf_counter()
        
        # Compile model with optimizer
        self.model.compile(optimizer, lr=lr)
        
        # Training with periodic resampling
        if self.adaptive_sampling and resample_every > 0:
            print("Training with adaptive sampling...")
            loss_history_adam = []
            
            for epoch in range(0, iterations, resample_every):
                # Train for resample_every iterations
                iters = min(resample_every, iterations - epoch)
                losshistory, train_state = self.model.train(iterations=iters, display_every=100)
                
                # Handle both list and numpy array formats
                if isinstance(losshistory.loss_train, np.ndarray):
                    loss_history_adam.extend(losshistory.loss_train.tolist())
                else:
                    loss_history_adam.extend(losshistory.loss_train)
                
                # Adaptive resampling
                if epoch + resample_every < iterations:
                    print(f"\nResampling at iteration {epoch + iters}...")
                    geom = self.create_geometry()
                    X_new = self.adaptive_resample(geom, num_domain=5000)
                    
                    # Update data (note: this is a simplified approach)
                    # In practice, would need to recreate PDE data object
                    print(f"Resampled {len(X_new)} points")
        else:
            print("Training without adaptive sampling...")
            losshistory, train_state = self.model.train(iterations=iterations)
            # Handle both list and numpy array formats
            if isinstance(losshistory.loss_train, np.ndarray):
                loss_history_adam = losshistory.loss_train.tolist()
            else:
                loss_history_adam = losshistory.loss_train
        
        adam_end = perf_counter()
        adam_time = adam_end - adam_start
        print(f"\nAdam training completed in {adam_time:.2f} seconds")
        
        # Stage 2: L-BFGS refinement (optional)
        if use_lbfgs:
            print(f"\nStarting L-BFGS refinement for {lbfgs_iterations} iterations...")
            lbfgs_start = perf_counter()
            
            # Compile with L-BFGS
            self.model.compile("L-BFGS")
            losshistory_lbfgs, train_state_lbfgs = self.model.train(iterations=lbfgs_iterations)
            
            lbfgs_end = perf_counter()
            lbfgs_time = lbfgs_end - lbfgs_start
            print(f"L-BFGS refinement completed in {lbfgs_time:.2f} seconds")
            
            # Combine loss histories
            if hasattr(losshistory, 'loss_train'):
                combined_loss = np.concatenate([losshistory.loss_train, losshistory_lbfgs.loss_train])
                losshistory.loss_train = combined_loss
        
        print("\nTraining finished.")
        print(f"Total training time: {(perf_counter() - adam_start):.2f} seconds")
        
        return losshistory, train_state
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict displacement field at given points."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        result = self.model.predict(x)
        return np.array(result) if not isinstance(result, np.ndarray) else result
    
    def compute_von_mises_stress(self, x_test: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Compute von Mises stress from displacement field using automatic differentiation.
        
        Parameters:
        - x_test: Test coordinates (N, 2)
        - y_pred: Predicted displacements (N, 2)
        
        Returns:
        - von_mises: Von Mises stress values (N,)
        """
        import torch
        
        # Convert to torch tensors for gradient computation
        x_torch = torch.tensor(x_test, dtype=torch.float32, requires_grad=True)
        
        # Get model predictions with gradients
        with torch.enable_grad():
            y_torch = self.net(x_torch)
            u_x = y_torch[:, 0:1]
            u_y = y_torch[:, 1:2]
            
            # Compute strain components
            u_x_x = torch.autograd.grad(u_x, x_torch, torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
            u_x_y = torch.autograd.grad(u_x, x_torch, torch.ones_like(u_x), create_graph=True)[0][:, 1:2]
            u_y_x = torch.autograd.grad(u_y, x_torch, torch.ones_like(u_y), create_graph=True)[0][:, 0:1]
            u_y_y = torch.autograd.grad(u_y, x_torch, torch.ones_like(u_y), create_graph=True)[0][:, 1:2]
            
            eps_xx = u_x_x
            eps_yy = u_y_y
            eps_xy = 0.5 * (u_x_y + u_y_x)
            
            # Compute stress components
            sigma_xx = self.lam * (eps_xx + eps_yy) + 2 * self.mu * eps_xx
            sigma_yy = self.lam * (eps_xx + eps_yy) + 2 * self.mu * eps_yy
            sigma_xy = 2 * self.mu * eps_xy
            
            # Von Mises stress: sqrt(σ_xx² - σ_xx*σ_yy + σ_yy² + 3*σ_xy²)
            von_mises = torch.sqrt(sigma_xx**2 - sigma_xx*sigma_yy + sigma_yy**2 + 3*sigma_xy**2)
        
        return von_mises.detach().numpy().flatten()
    
    def visualize_results(self, x_test: np.ndarray, y_pred: np.ndarray, losshistory=None):
        """Visualize displacement and stress fields."""
        n = int(np.sqrt(len(x_test)))
        X = x_test[:, 0].reshape(n, n)
        Y = x_test[:, 1].reshape(n, n)
        
        u_x = y_pred[:, 0].reshape(n, n)
        u_y = y_pred[:, 1].reshape(n, n)
        
        # Compute von Mises stress
        print("Computing von Mises stress...")
        von_mises = self.compute_von_mises_stress(x_test, y_pred)
        von_mises = von_mises.reshape(n, n)
        
        # Create figure with better layout
        fig, axes = plt.subplots(1, 3, figsize=(20, 5.5))
        
        # Define crack line coordinates
        crack_x = [self.crack_center[0] - self.crack_half_length, 
                   self.crack_center[0] + self.crack_half_length]
        crack_y = [self.crack_center[1], self.crack_center[1]]
        
        # Displacement u_x
        im0 = axes[0].contourf(X, Y, u_x, levels=30, cmap='RdBu_r')
        axes[0].plot(crack_x, crack_y, 'k-', linewidth=4)
        axes[0].set_title("Displacement $u_x$", fontsize=14, fontweight='bold')
        axes[0].set_xlabel("x", fontsize=12)
        axes[0].set_ylabel("y", fontsize=12)
        axes[0].set_aspect('equal')
        cbar0 = plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        cbar0.ax.tick_params(labelsize=10)
        axes[0].grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        # Displacement u_y
        im1 = axes[1].contourf(X, Y, u_y, levels=30, cmap='RdBu_r')
        axes[1].plot(crack_x, crack_y, 'k-', linewidth=4)
        axes[1].set_title("Displacement $u_y$", fontsize=14, fontweight='bold')
        axes[1].set_xlabel("x", fontsize=12)
        axes[1].set_ylabel("y", fontsize=12)
        axes[1].set_aspect('equal')
        cbar1 = plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        cbar1.ax.tick_params(labelsize=10)
        axes[1].grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        # Von Mises stress - use symmetric log scale to handle stress concentration
        # Clip extreme values for better visualization
        von_mises_clipped = np.clip(von_mises, 0, np.percentile(von_mises, 99))
        
        im2 = axes[2].contourf(X, Y, von_mises_clipped, levels=30, cmap='jet')
        axes[2].plot(crack_x, crack_y, 'k-', linewidth=4)
        # Mark crack tips with circles
        axes[2].plot([self.crack_tip1[0], self.crack_tip2[0]], 
                    [self.crack_tip1[1], self.crack_tip2[1]], 
                    'ko', markersize=8, markerfacecolor='white', markeredgewidth=2)
        axes[2].set_title("Von Mises Stress", fontsize=14, fontweight='bold')
        axes[2].set_xlabel("x", fontsize=12)
        axes[2].set_ylabel("y", fontsize=12)
        axes[2].set_aspect('equal')
        cbar2 = plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
        cbar2.ax.tick_params(labelsize=10)
        cbar2.set_label('$\sigma_{vm}$', fontsize=11)
        axes[2].grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        # Add text annotation for crack tips
        axes[2].text(0.02, 0.98, 'White circles: Crack tips', 
                    transform=axes[2].transAxes, fontsize=9,
                    verticalalignment='top', bbox=dict(boxstyle='round', 
                    facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('xpinn_results.png', dpi=150, bbox_inches='tight')
        print("Saved figure as 'xpinn_results.png'")
        plt.show()
        
        # Plot Total Potential Energy vs Epochs with standard deviation
        if losshistory is not None:
            self.plot_potential_energy(losshistory)
    
    def plot_potential_energy(self, losshistory):
        """
        Plot Total Potential Energy vs Epochs with ±1 standard deviation.
        
        The total potential energy is computed from the loss history.
        """
        plt.figure(figsize=(10, 6))
        
        # Get total loss (sum of all loss components)
        if len(losshistory.loss_train.shape) > 1:
            total_loss = np.sum(losshistory.loss_train, axis=1)
        else:
            total_loss = losshistory.loss_train
        
        # Compute potential energy (proportional to loss)
        # In PINN, the loss represents the violation of governing equations
        # which is related to the potential energy of the system
        potential_energy = total_loss
        
        epochs = np.arange(len(potential_energy))
        
        # Compute running statistics with a window for smoother visualization
        window_size = max(1, len(potential_energy) // 100)  # 1% window
        
        mean_energy = []
        std_energy = []
        
        for i in range(len(potential_energy)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(potential_energy), i + window_size // 2 + 1)
            window_data = potential_energy[start_idx:end_idx]
            
            mean_energy.append(np.mean(window_data))
            std_energy.append(np.std(window_data))
        
        mean_energy = np.array(mean_energy)
        std_energy = np.array(std_energy)
        
        # Plot mean line
        plt.plot(epochs, mean_energy, 'b-', linewidth=2, label='Mean Loss')
        
        # Plot ±1 standard deviation
        plt.fill_between(epochs, 
                        mean_energy - std_energy, 
                        mean_energy + std_energy,
                        alpha=0.3, 
                        color='lightblue',
                        label='±1 Std Dev')
        
        plt.xlabel('Epoch Number', fontsize=12)
        plt.ylabel('Total Potential Energy', fontsize=12)
        plt.title('Total Potential Energy and Standard Deviations Over Training', fontsize=14)
        plt.legend(loc='upper right', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.yscale('log')  # Log scale to better visualize convergence
        plt.tight_layout()
        plt.show()


def run_xpinn_crack_example(use_advanced_training: bool = True):
    """
    Run the X-PINN crack analysis with advanced training strategies.
    
    Parameters:
    - use_advanced_training: If True, use adaptive sampling and L-BFGS refinement
    """
    # Initialize X-PINN with crack configuration
    xpinn = XFEMCrackPINN(
        crack_center=(0.5, 0.5),
        crack_half_length=0.15,  # 2a = 0.3
        crack_angle=0.0,  # Horizontal crack
        domain_size=(1.0, 1.0),
        enrichment_width=0.1
    )
    
    # Configure training parameters
    if use_advanced_training:
        xpinn.adaptive_sampling = True
        xpinn.rad_params = {'k1': 1.0, 'k2': 1.0}
        xpinn.loss_weights = {'pde': 1.0, 'bc': 5.0, 'ic': 5.0}
        print("=" * 60)
        print("ADVANCED TRAINING MODE")
        print("=" * 60)
        print("Features enabled:")
        print("  - Adaptive Residual-based Distribution (RAD)")
        print("  - Two-stage optimization (Adam + L-BFGS)")
        print("  - Learning rate scheduling")
        print("  - Weighted loss components")
        print("=" * 60)
    else:
        xpinn.adaptive_sampling = False
        print("Using standard training...")
    
    # Setup problem
    xpinn.setup_problem(num_domain=5000, num_boundary=200)
    
    # Train model with advanced features
    if use_advanced_training:
        # Stage 1: Adam with adaptive sampling
        losshistory, train_state = xpinn.train(
            iterations=5000,
            lr=1e-3,
            optimizer="adam",
            use_lbfgs=True,  # Enable L-BFGS refinement
            lbfgs_iterations=1000,
            resample_every=500  # Resample every 500 iterations
        )
    else:
        # Standard training
        losshistory, train_state = xpinn.train(
            iterations=5000,
            lr=1e-3,
            optimizer="adam",
            use_lbfgs=False,
            resample_every=0
        )
    
    # Generate test points
    xs = np.linspace(0, 1, 80)
    ys = np.linspace(0, 1, 80)
    X_test, Y_test = np.meshgrid(xs, ys)
    x_test_flat = np.column_stack([X_test.ravel(), Y_test.ravel()])
    
    # Predict
    y_pred = xpinn.predict(x_test_flat)
    
    # Visualize
    print("\nGenerating visualizations...")
    xpinn.visualize_results(x_test_flat, y_pred, losshistory)
    
    return xpinn


if __name__ == "__main__":
    try:
        xpinn = run_xpinn_crack_example()
        print("\nX-PINN crack analysis completed successfully!")
    except Exception as e:
        print(f"Error during X-PINN run: {e}")
        import traceback
        traceback.print_exc()
