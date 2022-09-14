---
title: 💀 Домашка
nav_order: 3
---

## Matrix calculus

1. Find the gradient $$\nabla f(x)$$ and hessian $$f''(x)$$, if $$f(x) = \dfrac{1}{2} \|Ax - b\|^2_2$$.
1. Find gradient and hessian of $$f : \mathbb{R}^n \to \mathbb{R}$$, if:

    $$
    f(x) = \log \sum\limits_{i=1}^m \exp (a_i^\top x + b_i), \;\;\;\; a_1, \ldots, a_m \in \mathbb{R}^n; \;\;\;  b_1, \ldots, b_m  \in \mathbb{R}
    $$
1. Calculate the derivatives of the loss function with respect to parameters $$\frac{\partial L}{\partial W}, \frac{\partial L}{\partial b}$$ for the single object $$x_i$$ (or, $$n = 1$$)
![](../images/simple_learning.svg)
1. Calculate: $$\dfrac{\partial }{\partial X} \sum \text{eig}(X), \;\;\dfrac{\partial }{\partial X} \prod \text{eig}(X), \;\;\dfrac{\partial }{\partial X}\text{tr}(X), \;\; \dfrac{\partial }{\partial X} \text{det}(X)$$
1. Calculate the first and the second derivative of the following function $$f : S \to \mathbb{R}$$
	$$
	f(t) = \text{det}(A − tI_n),
	$$
	where $$A \in \mathbb{R}^{n \times n}, S := \{t \in \mathbb{R} : \text{det}(A − tI_n) \neq 0\}	$$.
1. Find the gradient $$\nabla f(x)$$, if $$f(x) = \text{tr}\left( AX^2BX^{-\top} \right)$$.

## Automatic differentiation
1. Implement analytical expression of the gradient and hessian of the following functions:

	a. $$f(x) = \dfrac{1}{2}x^TAx + b^Tx + c$$
	b. $$f(x) = \ln \left( 1 + \exp\langle a,x\rangle\right)$$
	c. $$f(x) = \dfrac{1}{2} \|Ax - b\|^2_2$$

	and compare the analytical answers with those, which obtained with any automatic differentiation framework (autograd\jax\pytorch\tensorflow). Manuals: [Jax autograd manual](https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html), [general manual](https://colab.research.google.com/github/MerkulovDaniil/optim/blob/master/assets/Notebooks/Autograd.ipynb).

```python
import numpy as np

n = 10
A = np.random.rand((n,n))
b = np.random.rand(n)
c = np.random.rand(n)

def f(x):
    # Your code here
    return 0

def analytical_df(x):
    # Your code here
    return np.zeros(n)

def analytical_ddf(x):
    # Your code here
    return np.zeros((n,n))

def autograd_df(x):
    # Your code here
    return np.zeros(n)

def autograd_ddf(x):
    # Your code here
    return np.zeros((n,n))

x_test = np.random.rand(n)

print(f'Analytical and autograd implementations of the gradients are close: {np.allclose(analytical_df(x_test), autograd_df(x_test))}')
print(f'Analytical and autograd implementations of the hessians are close: {np.allclose(analytical_ddf(x_test), autograd_ddf(x_test))}')

```

## Convex sets

1. Prove that the set of square symmetric positive definite matrices is convex.
1. Show, that $$ \mathbf{conv}\{xx^\top: x \in \mathbb{R}^n, \|x\| = 1\} = \{A \in \mathbb{S}^n_+: \text{tr}(A) = 1\}$$.
1. Show that the hyperbolic set of $$ \{x \in \mathbb{R}^n_+ | \prod\limits_{i=1}^n x_i \geq 1 \} $$ is convex. 
Hint: For $$0 \leq \theta \leq 1$$ it is valid, that $$a^\theta b^{1 - \theta} \leq \theta a + (1-\theta)b$$ with non-negative $$a,b$$.
1. Prove, that the set $S \subseteq \mathbb{R}^n$ is convex if and only if $(\alpha + \beta)S = \alpha S + \beta S$ for all non-negative $\alpha$ and $\beta$.
1. Let $$x \in \mathbb{R}$$ is a random variable with a given probability distribution of $$\mathbb{P}(x = a_i) = p_i$$, where $$i = 1, \ldots, n$$, and $$a_1 < \ldots < a_n$$. It is said that the probability vector of outcomes of $$p \in \mathbb{R}^n$$ belongs to the probabilistic simplex, i.e. $$P = \{ p \mid \mathbf{1}^Tp = 1, p \succeq 0 \} = \{ p \mid p_1 + \ldots + p_n = 1, p_i \ge 0 \}$$. 
    Determine if the following sets of $$p$$ are convex:
    
	1. \$$\mathbb{P}(x > \alpha) \le \beta$$
	1. \$$\mathbb{E} \vert x^{201}\vert \le \alpha \mathbb{E}\vert x \vert$$
	1. \$$\mathbb{E} \vert x^{2}\vert \ge \alpha $$
	1. \$$\mathbb{V}x \ge \alpha$$

## Convex functions

1. Prove, that function $$f(X) = \mathbf{tr}(X^{-1}), X \in S^n_{++}$$ is convex, while $$g(X) = (\det X)^{1/n}, X \in S^n_{++}$$ is concave.
1. Kullback–Leibler divergence between $$p,q \in \mathbb{R}^n_{++}$$ is:
	
	$$
	D(p,q) = \sum\limits_{i=1}^n (p_i \log(p_i/q_i) - p_i + q_i)
	$$
	
	Prove, that $$D(p,q) \geq 0 \; \forall p,q \in \mathbb{R}^n_{++}$$ и $$D(p,q) = 0 \leftrightarrow p = q$$  
	
	Hint: 
	$$
	D(p,q) = f(p) - f(q) - \nabla f(q)^T(p-q), \;\;\;\; f(p) = \sum\limits_{i=1}^n p_i \log p_i
	$$
1. Let $$x$$ be a real variable with the values $$a_1 < a_2 < \ldots < a_n$$ with probabilities $$\mathbb{P}(x = a_i) = p_i$$. Derive the convexity or concavity of the following functions from $$p$$ on the set of $$\left\{p \mid \sum\limits_{i=1}^n p_i = 1, p_i \ge 0 \right\}$$  
	* \$$\mathbb{E}x$$
	* \$$\mathbb{P}\{x \ge \alpha\}$$
	* \$$\mathbb{P}\{\alpha \le x \le \beta\}$$
	* \$$\sum\limits_{i=1}^n p_i \log p_i$$
	* \$$\mathbb{V}x = \mathbb{E}(x - \mathbb{E}x)^2$$
	* \$$\mathbf{quartile}(x) = {\operatorname{inf}}\left\{ \beta \mid \mathbb{P}\{x \le \beta\} \ge 0.25 \right\}$$ 
1.  Is the function returning the arithmetic mean of vector coordinates is a convex one: $$a(x) = \frac{1}{n}\sum\limits_{i=1}^n x_i$$, what about geometric mean: $$g(x) = \prod\limits_{i=1}^n \left(x_i \right)^{1/n}$$?
1. Is $$f(x) = -x \ln x - (1-x) \ln (1-x)$$ convex?
1. Let $$f: \mathbb{R}^n \to \mathbb{R}$$ be the following function:
    $$
    f(x) = \sum\limits_{i=1}^k x_{\lfloor i \rfloor},
    $$
    where $$1 \leq k \leq n$$, while the symbol $$x_{\lfloor i \rfloor}$$ stands for the $$i$$-th component of sorted ($$x_{\lfloor 1 \rfloor}$$ - maximum component of $$x$$ and $$x_{\lfloor n \rfloor}$$ - minimum component of $$x$$) vector of $$x$$. Show, that $$f$$ is a convex function.

## General optimization problems

1. Give an explicit solution of the following LP.
	
	$$
	\begin{split}
	& c^\top x \to \min\limits_{x \in \mathbb{R}^n }\\
	\text{s.t. } & Ax = b
	\end{split}
	$$

1. Give an explicit solution of the following LP.
	
	$$
	\begin{split}
	& c^\top x \to \min\limits_{x \in \mathbb{R}^n }\\
	\text{s.t. } & 1^\top x = 1, \\
	& x \succeq 0 
	\end{split}
	$$

	This problem can be considered as a simplest portfolio optimization problem.

1. Give an explicit solution of the following LP.
	
	$$
	\begin{split}
	& c^\top x \to \min\limits_{x \in \mathbb{R}^n }\\
	\text{s.t. } & 1^\top x = \alpha, \\
	& 0 \preceq x \preceq 1,
	\end{split}
	$$

	where $$\alpha$$ is an integer between $$0$$ and $$n$$. What happens if $$\alpha$$ is not an integer (but satisfies $$0 \leq \alpha \leq n$$)? What if we change the equality to an inequality $$1^\top x \leq \alpha$$?

1. Give an explicit solution of the following QP.
	
	$$
	\begin{split}
	& c^\top x \to \min\limits_{x \in \mathbb{R}^n }\\
	\text{s.t. } & x^\top A x \leq 1,
	\end{split}
	$$

	where $$A \in \mathbb{S}^n_{++}, c \neq 0$$. What is the solution if the problem is not convex $$(A \notin \mathbb{S}^n_{++})$$ (Hint: consider eigendecomposition of the matrix: $$A = Q \mathbf{diag}(\lambda)Q^\top = \sum\limits_{i=1}^n \lambda_i q_i q_i^\top$$ and different cases of $$\lambda >0, \lambda=0, \lambda<0$$)?

1. Give an explicit solution of the following QP.
	
	$$
	\begin{split}
	& c^\top x \to \min\limits_{x \in \mathbb{R}^n }\\
	\text{s.t. } & (x - x_c)^\top A (x - x_c) \leq 1,
	\end{split}
	$$

	where $$A \in \mathbb{S}^n_{++}, c \neq 0, x_c \in \mathbb{R}^n$$.

1. Give an explicit solution of the following QP.
	
	$$
	\begin{split}
	& x^\top Bx \to \min\limits_{x \in \mathbb{R}^n }\\
	\text{s.t. } & x^\top A x \leq 1,
	\end{split}
	$$

	where $$A \in \mathbb{S}^n_{++}, B \in \mathbb{S}^n_{+}$$.

1.  Consider the equality constrained least-squares problem
	
	$$
	\begin{split}
	& \|Ax - b\|_2^2 \to \min\limits_{x \in \mathbb{R}^n }\\
	\text{s.t. } & Cx = d,
	\end{split}
	$$

	where $$A \in \mathbb{R}^{m \times n}$$ with $$\mathbf{rank }A = n$$, and $$C \in \mathbb{R}^{k \times n}$$ with $$\mathbf{rank }C = k$$. Give the KKT conditions, and derive expressions for the primal solution $$x^*$$ and the dual solution $$\lambda^*$$.

1. Derive the KKT conditions for the problem
	
	$$
	\begin{split}
	& \mathbf{tr \;}X - \log\text{det }X \to \min\limits_{X \in \mathbb{S}^n_{++} }\\
	\text{s.t. } & Xs = y,
	\end{split}
	$$

	where $$y \in \mathbb{R}^n$$ and $$s \in \mathbb{R}^n$$ are given with $$y^\top s = 1$$. Verify that the optimal solution is given by

	$$
	X^* = I + yy^\top - \dfrac{1}{s^\top s}ss^\top
	$$
