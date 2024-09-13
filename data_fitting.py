import argparse
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich.console import Console
from rich.rule import Rule
from rich.table import Table
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit
from scipy.special import wofz

from generate_random_data import generate_noisy_data

warnings.filterwarnings("ignore", category=RuntimeWarning)

def gaussian(x, a, b, c):
    y = a*np.exp(-1*(x-b)**2/(2*c**2))
    return y

def double_gaussian(x, a, b, c, d, e, f):
    return (gaussian(x, a, b, c) + gaussian(x, d, e, f))

def trip_gaussian(x, a, b, c, d, e, f, g, h, i):
    return (gaussian(x, a, b, c) + gaussian(x, d, e, f) + 
            gaussian(x, g, h, i))

def quad_gaussian(x, a, b, c, d, e, f, g, h, i, j, k, l):
    return (gaussian(x, a, b, c) + gaussian(x, d, e, f) + 
            gaussian(x, g, h, i) + gaussian(x, j, k, l))

def quin_gaussian(x, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o):
    return (gaussian(x, a, b, c) + gaussian(x, d, e, f) + 
            gaussian(x, g, h, i) + gaussian(x, j, k, l) + 
            gaussian(x, m, n, o))

def lorentzian(x, a, b, c):
    return a / (1 + ((x - b) / c)**2)

def voigt(x, A, x0, sigma, gamma):
    z = (x - x0 + 1j * gamma) / (sigma * np.sqrt(2))
    return A * np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi))

def pseudo_voigt(x, A, x0, sigma, gamma, eta):
    return eta * gaussian(x, A, x0, sigma) + (1 - eta) * lorentzian(x, A, x0, gamma)

def exponential(x, a, b):
    y = a*np.exp(b*x)
    return y

def linear(x, a, b):
    y = a*x + b
    return y

def quadratic(x, a, b, c):
    y = a*x**2 + b*x + c
    return y

def logarithmic(x, a, b, c, epsilon=1e-10):
    y = a*np.log(b*x + epsilon) + c        
    return y

def trig(x, a, b, c, d, e, n):
    y = a*np.sin(x+b)**n+c*np.cos(x+d)**n+e
    return y

def create_table_selection():
    console.print()

    table = Table(border_style=None, show_header=True, show_edge=False, show_lines=False)
    table.add_column('[bold][green][underline]Data Fitting', justify='left')
    table.add_column('[bold][yellow][underline]Other', justify='left')
    table.add_row('[cyan]0[/] linear', '[red]p[/] plot data')
    table.add_row('[cyan]1[/] quadratic', '[red]c[/] clear console')
    table.add_row('[cyan]2[/] exponential', '[red]g[/] generate random data')
    table.add_row('[cyan]3[/] natural log', '[red]r[/] reload data')
    table.add_row('[cyan]4[/] gaussian', '[red]h[/] help')
    table.add_row('[cyan]5[/] lorentzian', '[red]q[/] quit')
    table.add_row('[cyan]6[/] voigt', '')
    table.add_row('[cyan]7[/] pseudo voigt', '')
    table.add_row('[cyan]8[/] sinⁿ(x) + cosⁿ(x)', '')
    table.add_row('[cyan]9[/] cubic spline', '')
        
    console.print(table)
    console.print()

def r_squared_val(fit_func, y_data):
    residuals = y_data - fit_func 
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_data - np.mean(y_data))**2)
    r_squared = 1 - (ss_res / ss_tot)
    return r_squared

def plot_data (fit_func, x_data, y_data, polar):
    plt.figure()
    if polar:
        ax = plt.subplot(111, projection='polar')
    else: 
        ax = plt.subplot(111)   
    ax.plot(x_data, y_data, 'o', label='data')
    ax.plot(x_data, fit_func, '-', label='fit')
    if polar:
        plt.legend(loc=(1,1))
    else:
        plt.legend()
    plt.show()

parser = argparse.ArgumentParser(description='Process some data.')
parser.add_argument('data_path', type=str, help='Path to the data file')
parser.add_argument('--header', action='store_true', help='Data file contains file headers')
parser.add_argument('--polar', action='store_true', help='Plot all data in polar coordinates')
args = parser.parse_args()
file_path = args.data_path
headers = args.header
polar = args.polar
console = Console()
try:
    if headers:
        df = pd.read_csv(file_path)
    else:
        df = pd.read_csv(file_path, header=None)
except Exception:
    console.print("Unable to read file")
    exit()
console.clear()
data_list = list(zip(df.iloc[:, 0], df.iloc[:, 1]))
x_data = np.array([x for (x,y) in data_list])
y_data = np.array([y for (x,y) in data_list])
while True:
    create_table_selection()
    selection = input('Selection(s): ')
    results = selection.split(",")
    rule = Rule(style="white")
    console.print(rule)
    if selection == '0':
        initial_guess = [0,0]
        parameters, covariance = curve_fit(linear, x_data, y_data, p0=initial_guess, maxfev=5000000)
        fit_a, fit_b = parameters
        fit_func = linear(x_data, fit_a, fit_b)
        r_squared = r_squared_val(fit_func, y_data)
        console.print('Data fitted to: f(x) = [blue]a[/]x+[blue]b[/]')
        console.print(f'\t[blue]a[/] = [green]{fit_a}[/]\n\t[blue]b[/] = [green]{fit_b}[/]')
        console.print(f'\n\t[yellow]r²[/] = [magenta]{r_squared}[/]')
        console.print(rule)
        plot_data(fit_func,x_data,y_data,polar)

    elif selection == '1':
        initial_guess = [0,0,0]
        parameters, covariance = curve_fit(quadratic, x_data, y_data, p0=initial_guess, maxfev=5000000)
        fit_a, fit_b, fit_c = parameters
        fit_func = quadratic(x_data, fit_a, fit_b, fit_c)
        r_squared = r_squared_val(fit_func, y_data)
        console.print('Data fitted to: f(x) = [blue]a[/]x²+[blue]b[/]x+[blue]c[/]')
        console.print(f'\t[blue]a[/] = [green]{fit_a}[/]\n\t[blue]b[/] = [green]{fit_b}[/]\n\t[blue]c[/] = [green]{fit_c}[/]')
        console.print(f'\n\t[yellow]r²[/] = [magenta]{r_squared}[/]')
        console.print(rule)
        plot_data(fit_func,x_data,y_data,polar)

    elif selection == '2':
        initial_guess = [0,0]
        parameters, covariance = curve_fit(exponential, x_data, y_data, p0=initial_guess, maxfev=5000000)
        fit_a, fit_b = parameters
        fit_func = exponential(x_data, fit_a, fit_b)
        r_squared = r_squared_val(fit_func, y_data)
        console.print('Data fitted to: f(x) = [blue]a[/]e^([blue]b[/]x)')
        console.print(f'\n\t[blue]a[/] = [green]{fit_a}[/]\n\t[blue]b[/] = [green]{fit_b}[/]')
        console.print(f'\n\t[yellow]r²[/] = [magenta]{r_squared}[/]')
        console.print(rule)
        plot_data(fit_func,x_data,y_data,polar)

    elif selection == '3':
        initial_guess = [1,1,1]
        parameters, covariance = curve_fit(logarithmic, x_data, y_data, p0=initial_guess, maxfev=5000000)
        fit_a, fit_b, fit_c = parameters
        fit_func = logarithmic(x_data, fit_a, fit_b, fit_c)
        r_squared = r_squared_val(fit_func, y_data)
        console.print('Data fitted to: f(x) = [blue]a[/]ln([blue]b[/]x)+[blue]c[/]')
        console.print(f'\t[blue]a[/] = [green]{fit_a}[/]\n\t[blue]b[/] = [green]{fit_b}[/]\n\t[blue]c[/] = [green]{fit_c}[/]')
        console.print(f'\n\t[yellow]r²[/] = [magenta]{r_squared}[/]')
        console.print(rule)
        plot_data(fit_func,x_data,y_data,polar)
 
    elif selection == '4':
        num_of_peaks = input('Number of Gaussians (1-5): ')
        try:
            num_of_peaks = int(num_of_peaks)
            if num_of_peaks not in [1, 2, 3, 4, 5]:
                print('Please enter a valid int')
                console.print(rule)
                continue
        except ValueError:
            print('Please enter a valid int')
            console.print(rule)
            continue

        if num_of_peaks == 1:
            initial_guess = [max(y_data), np.mean(x_data), np.std(x_data)]
            parameters, covariance = curve_fit(gaussian, x_data, y_data, p0=initial_guess, maxfev=5000000)
            fit_a, fit_b, fit_c = parameters
            fit_func = gaussian(x_data, fit_a, fit_b, fit_c)
            r_squared = r_squared_val(fit_func, y_data)
            console.print('Data fitted to: f(x) = [blue]a[/]e^[-(x-[blue]μ[/])²/(2[blue]σ[/]²)]')
            console.print(f'\t[blue]a[/] = [green]{fit_a}[/]\n\t[blue]μ[/] = [green]{fit_b}[/]\n\t[blue]σ[/] = [green]{fit_c}[/]')
            console.print(f'\n\t[yellow]r²[/] = [magenta]{r_squared}[/]')

        elif num_of_peaks == 2:
            initial_guess = [max(y_data), np.mean(x_data), np.std(x_data), max(y_data), np.mean(x_data), np.std(x_data)]
            parameters, covariance = curve_fit(double_gaussian, x_data, y_data, p0=initial_guess, maxfev=5000000)
            fit_a, fit_b, fit_c, fit_d, fit_e, fit_f = parameters
            fit_func = double_gaussian(x_data, fit_a, fit_b, fit_c, fit_d, fit_e, fit_f)
            r_squared = r_squared_val(fit_func, y_data)
            console.print('Data fitted to: f(x) = [blue]a1[/]e^[-(x-[blue]μ1[/])²/(2[blue]σ1[/]²)]+[blue]a2[/]e^[-(x-[blue]μ2[/])²/(2[blue]σ2[/]²)]')
            console.print(f'\t[blue]a1[/] = [green]{fit_a}[/]\n\t[blue]μ1[/] = [green]{fit_b}[/]\n\t[blue]σ1[/] = [green]{fit_c}[/]')
            console.print(f'\n\t[blue]a2[/] = [green]{fit_d}[/]\n\t[blue]μ2[/] = [green]{fit_e}[/]\n\t[blue]σ2[/] = [green]{fit_f}[/]')
            console.print(f'\n\t[yellow]r²[/] = [magenta]{r_squared}[/]')
        
        elif num_of_peaks == 3:
            initial_guess = [max(y_data), np.mean(x_data), np.std(x_data), max(y_data), np.mean(x_data), np.std(x_data), max(y_data), np.mean(x_data), np.std(x_data)]
            parameters, covariance = curve_fit(trip_gaussian, x_data, y_data, p0=initial_guess, maxfev=5000000)
            fit_a, fit_b, fit_c, fit_d, fit_e, fit_f, fit_g, fit_h, fit_i = parameters
            fit_func = trip_gaussian(x_data, fit_a, fit_b, fit_c, fit_d, fit_e, fit_f, fit_g, fit_h, fit_i)
            r_squared = r_squared_val(fit_func, y_data)
            console.print('''Data fitted to:\nf(x) = [blue]a1[/]e^[-(x-[blue]μ1[/])²/(2[blue]σ1[/]²)]+[blue]a2[/]e^[-(x-[blue]μ2[/])²/(2[blue]σ2[/]²)]+[blue]a3[/]e^[-(x-[blue]μ3[/])²/(2[blue]σ3[/]²)]''')
            console.print(f'\t[blue]a1[/] = [green]{fit_a}[/]\n\t[blue]μ1[/] = [green]{fit_b}[/]\n\t[blue]σ1[/] = [green]{fit_c}[/]')
            console.print(f'\n\t[blue]a2[/] = [green]{fit_d}[/]\n\t[blue]μ2[/] = [green]{fit_e}[/]\n\t[blue]σ2[/] = [green]{fit_f}[/]')
            console.print(f'\n\t[blue]a3[/] = [green]{fit_g}[/]\n\t[blue]μ3[/] = [green]{fit_h}[/]\n\t[blue]σ3[/] = [green]{fit_i}[/]')
            console.print(f'\n\t[yellow]r²[/] = [magenta]{r_squared}[/]')

        elif num_of_peaks == 4:
            initial_guess = [max(y_data), np.mean(x_data), np.std(x_data), max(y_data), np.mean(x_data), np.std(x_data), max(y_data), np.mean(x_data), np.std(x_data),max(y_data), np.mean(x_data), np.std(x_data)]
            parameters, covariance = curve_fit(quad_gaussian, x_data, y_data, p0=initial_guess, maxfev=5000000)
            fit_a, fit_b, fit_c, fit_d, fit_e, fit_f, fit_g, fit_h, fit_i, fit_j, fit_k, fit_l = parameters
            fit_func = quad_gaussian(x_data, fit_a, fit_b, fit_c, fit_d, fit_e, fit_f, fit_g, fit_h, fit_i, fit_j, fit_k, fit_l)
            r_squared = r_squared_val(fit_func, y_data)
            console.print('Data fitted to:\nf(x) = [blue]a1[/]e^[-(x-[blue]μ1[/])²/(2[blue]σ1[/]²)]+[blue]a2[/]e^[-(x-[blue]μ2[/])²/(2[blue]σ2[/]²)]+[blue]a3[/]e^[-(x-[blue]μ3[/])²/(2[blue]σ3[/]²)]+[blue]a4[/]e^[-(x-[blue]μ4[/])²/(2[blue]σ4[/]²)]')
            console.print(f'\t[blue]a1[/] = [green]{fit_a}[/]\n\t[blue]μ1[/] = [green]{fit_b}[/]\n\t[blue]σ1[/] = [green]{fit_c}[/]')
            console.print(f'\n\t[blue]a2[/] = [green]{fit_d}[/]\n\t[blue]μ2[/] = [green]{fit_e}[/]\n\t[blue]σ2[/] = [green]{fit_f}[/]')
            console.print(f'\n\t[blue]a3[/] = [green]{fit_g}[/]\n\t[blue]μ3[/] = [green]{fit_h}[/]\n\t[blue]σ3[/] = [green]{fit_i}[/]')
            console.print(f'\n\t[blue]a4[/] = [green]{fit_j}[/]\n\t[blue]μ4[/] = [green]{fit_k}[/]\n\t[blue]σ4[/] = [green]{fit_l}[/]')
            console.print(f'\n\t[yellow]r²[/] = [magenta]{r_squared}[/]')

        elif num_of_peaks == 5:
            initial_guess = [max(y_data), np.mean(x_data), np.std(x_data), max(y_data), np.mean(x_data), np.std(x_data), max(y_data), np.mean(x_data), np.std(x_data), max(y_data), np.mean(x_data), np.std(x_data), max(y_data), np.mean(x_data), np.std(x_data)]
            parameters, covariance = curve_fit(quin_gaussian, x_data, y_data, p0=initial_guess, maxfev=5000000)
            fit_a, fit_b, fit_c, fit_d, fit_e, fit_f, fit_g, fit_h, fit_i, fit_j, fit_k, fit_l, fit_m, fit_n, fit_o = parameters
            fit_func = quin_gaussian(x_data, fit_a, fit_b, fit_c, fit_d, fit_e, fit_f, fit_g, fit_h, fit_i, fit_j, fit_k, fit_l, fit_m, fit_n, fit_o)
            r_squared = r_squared_val(fit_func, y_data)
            console.print('Data fitted to:\n f(x) = [blue]a1[/]e^[-(x-[blue]μ1[/])²/(2[blue]σ1[/]²)]+[blue]a2[/]e^[-(x-[blue]μ2[/])²/(2[blue]σ2[/]²)]+[blue]a3[/]e^[-(x-[blue]μ3[/])²/(2[blue]σ3[/]²)]+[blue]a4[/]e^[-(x-[blue]μ4[/])²/(2[blue]σ4[/]²)]+[blue]a5[/]e^[-(x-[blue]μ5[/])²/(2[blue]σ5[/]²)]')
            console.print(f'\t[blue]a1[/] = [green]{fit_a}[/]\n\t[blue]μ1[/] = [green]{fit_b}[/]\n\t[blue]σ1[/] = [green]{fit_c}[/]')
            console.print(f'\n\t[blue]a2[/] = [green]{fit_d}[/]\n\t[blue]μ2[/] = [green]{fit_e}[/]\n\t[blue]σ2[/] = [green]{fit_f}[/]')
            console.print(f'\n\t[blue]a3[/] = [green]{fit_g}[/]\n\t[blue]μ3[/] = [green]{fit_h}[/]\n\t[blue]σ3[/] = [green]{fit_i}[/]')
            console.print(f'\n\t[blue]a4[/] = [green]{fit_j}[/]\n\t[blue]μ4[/] = [green]{fit_k}[/]\n\t[blue]σ4[/] = [green]{fit_l}[/]')
            console.print(f'\n\t[blue]a5[/] = [green]{fit_m}[/]\n\t[blue]μ5[/] = [green]{fit_n}[/]\n\t[blue]σ5[/] = [green]{fit_o}[/]')
            console.print(f'\n\t[yellow]r²[/] = [magenta]{r_squared}[/]')
        
        console.print(rule)
        plot_data(fit_func,x_data,y_data,polar) 

    elif selection == '5':
        initial_guess = [max(y_data), np.mean(x_data), np.std(x_data)]
        parameters, covariance = curve_fit(lorentzian, x_data, y_data, p0=initial_guess, maxfev=5000000)
        fit_a, fit_b, fit_c = parameters
        fit_func = lorentzian(x_data, fit_a, fit_b, fit_c)
        r_squared = r_squared_val(fit_func, y_data)
        console.print('Data fitted to: f(x) = [blue]a[/]/(1+((x-[blue]x0[/])/([blue]γ[/]))²)')
        console.print(f'\t[blue]a[/] = [green]{fit_a}[/]\n\t[blue]x0[/] = [green]{fit_b}[/]\n\t[blue]γ[/] = [green]{fit_c}[/]')
        console.print(f'\n\t[yellow]r²[/] = [magenta]{r_squared}[/]')
        console.print(rule)
        plot_data(fit_func,x_data,y_data,polar)

    elif selection == '6':
        initial_guess = [max(y_data), 1, np.mean(x_data), np.std(x_data)]
        parameters, covariance = curve_fit(voigt, x_data, y_data, p0=initial_guess, maxfev=500000000)
        fit_a, fit_b, fit_c, fit_d = parameters
        fit_func = voigt(x_data, fit_a, fit_b, fit_c, fit_d)
        r_squared = r_squared_val(fit_func, y_data)
        console.print('Data fitted to: f(x) = ∫[blue]a[/]/π+([blue]γ[/]e^(-(x-[blue]x0[/]-t)²)/2[blue]σ[/]²)/(t²+[blue]γ[/]²)dt')
        console.print(f'\t[blue]a[/] = [green]{fit_a}[/]\n\t[blue]x0[/] = [green]{fit_b}[/]\n\t[blue]σ[/] = [green]{fit_c}[/]\n\t[blue]γ[/] = [green]{fit_d}[/]')
        console.print(f'\n\t[yellow]r²[/] = [magenta]{r_squared}[/]')
        console.print(rule)
        plot_data(fit_func,x_data,y_data,polar)

    elif selection == '7':
        initial_guess = [max(y_data), 1, np.mean(x_data), np.std(x_data),0.5]
        parameters, covariance = curve_fit(pseudo_voigt, x_data, y_data, p0=initial_guess, maxfev=50000000)
        fit_a, fit_b, fit_c, fit_d, fit_e = parameters
        fit_func = pseudo_voigt(x_data, fit_a, fit_b, fit_c, fit_d, fit_e)
        r_squared = r_squared_val(fit_func, y_data)
        console.print('Data fitted to: f(x) = [blue]e[/]*[blue]a[/]e^[-(x-[blue]μ[/])²/(2[blue]σ[/]²)]+(1-[blue]e[/])[blue]a[/]/(1+((x-[blue]x0[/])/([blue]γ[/]))²)')
        console.print(f'\t[blue]a1[/] = [green]{fit_a}[/]\n\t[blue]μ[/] = [green]{fit_b}[/]\n\t[blue]σ[/] = [green]{fit_c}[/]\n\n\t[blue]x0[/] = [green]{fit_b}[/]\n\t[blue]γ[/] = [green]{fit_d}[/]\n\n\t[blue]e[/] = [green]{fit_e}[/]')
        console.print(f'\n\t[yellow]r²[/] = [magenta]{r_squared}[/]')
        console.print(rule)
        plot_data(fit_func,x_data,y_data,polar)

    elif selection == '8':
        initial_guess = [1,1,1,1,1]
        n = input('Choose n value: ')
        try:
            n = float(n)
        except ValueError:
            print('Please enter a float for n')
            console.print(rule)
            continue 
        parameters, covariance = curve_fit(lambda x, a, b, c, d, e: trig(x, a, b, c, d, e, n), x_data, y_data, p0=initial_guess,maxfev=5000000)
        fit_a, fit_b, fit_c, fit_d, fit_e = parameters
        fit_func = trig(x_data, fit_a, fit_b, fit_c, fit_d, fit_e, n)
        r_squared = r_squared_val(fit_func, y_data)
        console.print('Data fitted to: f(x) = [blue]a[/]sinⁿ([blue]b[/]x)+[blue]c[/]cosⁿ([blue]d[/]x)+e')
        console.print(f'\t[blue]a[/] = [green]{fit_a}[/]\n\t[blue]b[/] = [green]{fit_b}[/]\n\t[blue]c[/] = [green]{fit_c}[/]\n\t[blue]d[/] = [green]{fit_d}[/]\n\t[blue]e[/] = [green]{fit_e}[/]')
        console.print(f'\n\t[yellow]r²[/] = [magenta]{r_squared}[/]')
        console.print(rule)
        plot_data(fit_func,x_data,y_data,polar)

    elif selection == '9':
        spline=CubicSpline(x_data, y_data)
        fit_func=spline(x_data)
        console.print("Data fitted to a cubic spline piecewise polynomial.")
        while True:
            download_select = input('Download spline data to cwd? (y/n): ')
            if download_select in ['Y', 'y']:
                print("Downloaded")
                break
            elif download_select in ['N', 'n']:
                print("NO")
                break
        console.print(rule)
        plot_data(fit_func,x_data,y_data,polar)

    elif selection == 'p':
        plt.figure()
        if polar:
            ax = plt.subplot(111, projection='polar')
            plt.plot(x_data, y_data, 'o', label='data')
            plt.legend(loc=(1,1))
        else:
            ax = plt.subplot(111)
            plt.plot(x_data, y_data, 'o', label='data')
            plt.legend()
        plt.show()
    
    elif selection == 'g':
        console.print('Random data types:\n[yellow]linear\tquadratic\texp\tlog\tgaussian\ttrig[/]\n')
        func = input('Type of data to generate: ')
        if func not in ['linear', 'quadratic', 'exp', 'log', 'gaussian', 'trig']:
            print('Unsupported function type')
            console.print(rule)
            continue
        x_rand_data, y_rand_data = generate_noisy_data(func)
        data = pd.DataFrame({'x': x_rand_data, 'y': y_rand_data})

        csv_file_path = 'generated_data.csv'
        data.to_csv(csv_file_path, index=False, header=False)

        print(f'Data saved to {csv_file_path}')

        console.print(rule)

    elif selection == 'r':
        if headers:
            df = pd.read_csv(file_path)
        else:
            df = pd.read_csv(file_path, header=None)
        data_list = list(zip(df.iloc[:, 0], df.iloc[:, 1]))
        x_data = np.array([x for (x,y) in data_list])
        y_data = np.array([y for (x,y) in data_list])
        console.print('Data reloaded')
        console.print(rule)

    elif selection == 'c':
        console.clear()

    elif selection == 'q':
        exit()
