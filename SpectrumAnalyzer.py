import tkinter as tk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

def PrintSpectrum(spectrum, freq):
    with open("Spectrum.txt", "w") as sp:
        for i in range(0, len(spectrum)):
            sp.write("%s -- %s\n" % (str(freq[i]), str(spectrum[i])))

# SMA Calculator for Floating-Point Data
def CalculateSma(data, sma_period):
    sma = np.empty_like(data)
    for i in range(len(data)):
        if i < sma_period:
            sma[i] = np.mean(data[:i+1])
        else:
            sma[i] = np.mean(data[i-sma_period+1:i+1])
    return sma

def generate_plot():
    ticker = ticker_entry.get()
    start_date = start_entry.get()
    end_date = end_entry.get()
    sma_period = int(sma_entry.get())

    # Fetch the stock data from Yahoo Finance
    data = yf.download(ticker, start=start_date, end=end_date)
    prices = data['Close'].values

    # Subtract SMA from prices to level DC offset
    sma = CalculateSma(prices, sma_period)
    sma_adjusted_prices = prices - sma

    # Perform Fourier transform
    fft = np.fft.fft(sma_adjusted_prices) / (len(sma_adjusted_prices) + 1)
    fft_freq = np.fft.fftfreq(len(prices))

    # Calculate the amplitude spectrum
    amplitude_spectrum = np.abs(fft)

    # Define the frequency range for display
    frequency_min = -1
    frequency_max = 1
    assert len(amplitude_spectrum) == len(fft_freq), "Length mismatch between x, y plot variables"

    # Find peaks using a sliding window approach
    window_width = int(len(amplitude_spectrum) / 5)
    maxima_indices = []
    previous_max_index = -window_width
    for i in range(len(amplitude_spectrum) - window_width + 1):
        window = amplitude_spectrum[i:i + window_width]
        max_index = i + np.argmax(window)
        if max_index != previous_max_index:
            maxima_indices.append(max_index)
            previous_max_index = max_index

    # Remove duplicates if there are multiple maxima with the same x value
    maxima_indices = list(set(maxima_indices))

    # Sort the maxima indices based on their x values
    maxima_indices.sort(key=lambda index: fft_freq[index])

    # Plot the spectral distribution
    plt.subplot(2, 1, 1)
    plt.plot(fft_freq[(fft_freq > frequency_min) & (fft_freq < frequency_max)],
             amplitude_spectrum[(fft_freq > frequency_min) & (fft_freq < frequency_max)])
    plt.plot(fft_freq[maxima_indices][(fft_freq[maxima_indices] > frequency_min) & (fft_freq[maxima_indices] < frequency_max)],
             amplitude_spectrum[maxima_indices][(fft_freq[maxima_indices] > frequency_min) & (fft_freq[maxima_indices] < frequency_max)],
             'ro', label='Maxima')
    plt.xlabel('Frequency')
    plt.ylabel('Amplitude Spectrum')
    plt.title(f'Spectral Distribution of {ticker}')
    plt.grid(True)
    plt.legend()

    # Label the first maxima with their corresponding x-values on the spectral distribution plot
    for index in maxima_indices:
        plt.annotate(f'{fft_freq[index]:.4f}', xy=(fft_freq[index], amplitude_spectrum[index]),
                     xytext=(0, -10), textcoords='offset points', ha='center', va='top', fontsize=8, color='blue')

    # Plot the stock market data and its SMA
    plt.subplot(2, 1, 2)
    plt.plot(prices, label='Data')
    plt.plot(sma, label='SMA')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.title(f'Stock Market Data and SMA of {ticker} - Green Line Shows Maximum Fall')
    plt.grid(True)
    plt.legend()

    # Find the points where the SMA and data cross
    crosses = np.where(np.diff(np.sign(prices - sma)) != 0)[0]

    # Find the adjacent crossing points with the maximum difference in y-values
    max_difference = 0
    max_difference_index = -1
    for i in range(len(crosses) - 1):
        current_cross = crosses[i]
        next_cross = crosses[i + 1]
        if prices[current_cross] > prices[next_cross]:
            difference = prices[current_cross] - prices[next_cross]
            if difference > max_difference:
                max_difference = difference
                max_difference_index = i

    # Plot the line between the adjacent crossing points with maximum difference in y-values
    if max_difference_index >= 0:
        current_cross = crosses[max_difference_index]
        next_cross = crosses[max_difference_index + 1]
        plt.plot([current_cross, next_cross], [prices[current_cross], prices[next_cross]], 'g--')
        plt.annotate(f'{prices[current_cross]:.2f}', xy=(current_cross, prices[current_cross]),
                     xytext=(0, -10), textcoords='offset points', ha='center', va='top', fontsize=8, color='black')
        plt.annotate(f'{prices[next_cross]:.2f}', xy=(next_cross, prices[next_cross]),
                     xytext=(0, -10), textcoords='offset points', ha='center', va='top', fontsize=8, color='black')

    # Calculate the average difference in values at regions of adjacent crossings where the y value of the first crossing is greater than the y value of the second crossing
    average_diff = np.mean([prices[crosses[i]] - prices[crosses[i+1]] for i in range(len(crosses) - 1) if prices[crosses[i]] > prices[crosses[i+1]]])

    # Display the average difference above the plot
    plt.text(0.5, 1.1, f'Average Difference: {average_diff:.2f}', transform=plt.gca().transAxes, ha='center')

    plt.tight_layout()
    plt.show()

# Create the main window
window = tk.Tk()
window.title("Spectral Distribution Plot")

# Create labels and text fields for user input
tk.Label(window, text="Stock Ticker:").grid(row=0, column=0, sticky="E")
ticker_entry = tk.Entry(window)
ticker_entry.grid(row=0, column=1)

tk.Label(window, text="Start Date (YYYY-MM-DD):").grid(row=1, column=0, sticky="E")
start_entry = tk.Entry(window)
start_entry.grid(row=1, column=1)

tk.Label(window, text="End Date (YYYY-MM-DD):").grid(row=2, column=0, sticky="E")
end_entry = tk.Entry(window)
end_entry.grid(row=2, column=1)

tk.Label(window, text="SMA Period (in days):").grid(row=3, column=0, sticky="E")
sma_entry = tk.Entry(window)
sma_entry.grid(row=3, column=1)

# Create the generate button
generate_button = tk.Button(window, text="Generate", command=generate_plot)
generate_button.grid(row=4, column=0, columnspan=2, pady=10)

# Start the GUI main loop
window.mainloop()

