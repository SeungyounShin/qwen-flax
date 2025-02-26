import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

# Define function f(x) = sin(x) * x
def f(x):
    return jnp.sin(x)

# Calculate the derivative (gradient) of f(x) using automatic differentiation
grad_f = jax.grad(f)

# JIT 컴파일로 함수와 미분 함수를 최적화
f_jit = jax.jit(f)
grad_f_jit = jax.jit(grad_f)

# 여러 입력값에 대해 벡터화(vmap) 적용
f_vectorized = jax.vmap(f_jit)
grad_vectorized = jax.vmap(grad_f_jit)

# 0부터 2π까지 10개의 구간으로 나눈 입력값 생성
x = jnp.linspace(0, 2 * jnp.pi, 100)

# 함수값과 기울기 계산
fx = f_vectorized(x)
grad_fx = grad_vectorized(x)

print("x values:", x)
print("f(x):", fx)
print("f'(x):", grad_fx)

# 결과 시각화
plt.figure(figsize=(10, 6))
plt.plot(x, fx, 'b-o', label='f(x) = sin(x) * x')
plt.plot(x, grad_fx, 'r-o', label="f'(x)")
plt.grid(True)
plt.xlabel('x')
plt.ylabel('Value')
plt.title('Function f(x) = sin(x) * x and its derivative')
plt.legend()
plt.savefig('function_and_derivative.png')
plt.show()
