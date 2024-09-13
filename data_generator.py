import argparse

import pandas as pd

from generate_random_data import generate_noisy_data

parser = argparse.ArgumentParser(description='Process some data.')
parser.add_argument('func_type', type=str, help='Type of data to generate')
parser.add_argument('--n_points', type=int, default=100, help='Number of data points')
parser.add_argument('--noise_level', type=float, default=0.1, help='Noise level')
parser.add_argument('--a', type=float, default=1, help='Parameter a for functions')
parser.add_argument('--b', type=float, default=0, help='Parameter b for functions')
parser.add_argument('--c', type=float, default=0, help='Parameter c for functions')
parser.add_argument('--means', type=float, nargs='+', default=[2, 5, 8], help='Means for Gaussian')    
parser.add_argument('--std_devs', type=float, nargs='+', default=[0.5, 0.5, 0.5], help='Standard deviations for Gaussian')
parser.add_argument('--amplitudes', type=float, nargs='+', default=[1, 0.8, 0.6], help='Amplitudes for Gaussian')
parser.add_argument('--n', type=int, default=2, help='Power for trigonometric functions')

args = parser.parse_args()
func = args.func_type
    
x_data, y_data = generate_noisy_data(
    func,
    n_points=args.n_points,
    noise_level=args.noise_level,
    a=args.a,
    b=args.b,
    c=args.c,
    means=args.means,
    std_devs=args.std_devs,
    amplitudes=args.amplitudes,
    n=args.n
    )

data = pd.DataFrame({'x': x_data, 'y': y_data})
csv_file_path = 'generated_data.csv'
data.to_csv(csv_file_path, index=False, header=False)

print(f'Data saved to {csv_file_path}')
