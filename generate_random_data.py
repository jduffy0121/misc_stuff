import numpy as np


def generate_noisy_data(func_type, n_points=250, noise_level=0.25, **kwargs):    
    x = np.linspace(0, 10, n_points)
    y = np.zeros_like(x)
    if func_type == 'linear':
        a = kwargs.get('a', 1)
        b = kwargs.get('b', 1)
        y = a * x + b
    elif func_type == 'quadratic':
        a = kwargs.get('a', 1)
        b = kwargs.get('b', 1)
        c = kwargs.get('c', 1)
        y = a * x**2 + b * x + c
    elif func_type == 'exp':
        a = kwargs.get('a', 0.5)
        b = kwargs.get('b', 0.5)
        y = a * np.exp(b * x)
    elif func_type == 'log':
        a = kwargs.get('a', 1)
        b = kwargs.get('b', 0)
        y = a * np.log(x + 1e-5) + b
    elif func_type == 'gaussian':
        means = kwargs.get('means', [2, 5, 8])
        std_devs = kwargs.get('std_devs', [0.5, 0.5, 0.5])
        amplitudes = kwargs.get('amplitudes', [1, 0.8, 0.6])
        y = np.zeros_like(x)
        for mean, std, amp in zip(means, std_devs, amplitudes):
            y += amp * np.exp(-(x - mean) ** 2 / (2 * std ** 2))
    elif func_type == 'trig':
        n = kwargs.get('n', 1)
        a = kwargs.get('a', 1)
        b = kwargs.get('b', 1)
        y = a*np.sin(x)**n + b*np.cos(x)**n
    noise = noise_level * np.random.randn(n_points)
    y_noisy = y + noise
    return x, y_noisy
