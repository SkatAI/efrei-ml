import matplotlib.pyplot as plt
import numpy as np

def gradient(current_x, alpha, ):
    iters       = 0       #iteration counter
    max_iters   = 50    # maximum number of iterations

    xa = []
    while (iters < max_iters):
        xa.append(current_x)
        prev_x = current_x
        # gradient update
        current_x -= alpha * fct_prime(prev_x)
        # calulate error
        delta = abs(current_x - prev_x)
        print(current_x, delta)
        iters+=1

    print("The minimum is {:.4f}", current_x)
    print(f"In x = {current_x:.4f}, the fonction min is {fct(current_x):.4f}  ")
    return xa




# function
fct = lambda x: x**4 - 3 * x**3
# its derivative
# 4x^3 - 9 x^2 = x^2 (4x -9) => minima in x = 9/4 = 2.25
fct_prime = lambda x: 4 * x**3 - 9 * x**2

xx = np.arange(0, 3.6, 0.1)
y = [fct(x) for x in xx]



start_x   = 4      # The algorithm starts at x=6
alpha       = 0.01    # step size multiplier


xa = gradient(start_x, alpha)
ya = [fct(x) for x in xa]

fig, ax = plt.subplots(figsize=(10, 6))
plt.grid()
plt.plot(xx, y)
plt.scatter(xa,ya, color = 'red')
plt.title("function x^4 - 3 x^3 in [0, 3.5]")
plt.show()



