import tkinter as tk
from PIL import ImageTk, Image
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

# Find the most-probable long and short non-zero market cycle frequency
def ShortLongCycleFrequency(maxima_indices, fft_freq, amplitude_spectrum):
	# Filter maxima_indices based on fft_freq not equal to zero
	filtered_indices_long = [index for index in maxima_indices if fft_freq[index] != 0]
	# Find the index of the maximum value in amplitude_spectrum among the filtered indices
	max_index_long = np.argmax(amplitude_spectrum[filtered_indices_long])
	# Return the corresponding fft_freq value
	max_freq_long = fft_freq[filtered_indices_long[max_index_long]]
	# Find maximas any where between 2 (f = 0.5) to 14 (f = 0.07) days
	filtered_indices_short = [index for index in maxima_indices if 0.07 <= fft_freq[index] <= 0.5]
	max_index_short = np.argmax(amplitude_spectrum[filtered_indices_short])
	max_freq_short = fft_freq[filtered_indices_short[max_index_short]]
	return abs(max_freq_long), abs(max_freq_short)

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

	# Find the Most probable short and long market cycle period and the corresponding SMAs around the periods
	LongFreq, ShortFreq = ShortLongCycleFrequency(maxima_indices, fft_freq, amplitude_spectrum)
	LongPeriod = int(1/LongFreq)
	ShortPeriod = int(1/ShortFreq)
	SmaLong = CalculateSma(prices, LongPeriod)
	SmaShort = CalculateSma(prices, ShortPeriod)

	# Plot the spectral distribution
	plt.subplot(2, 1, 1)
	plt.plot(fft_freq[(fft_freq > frequency_min) & (fft_freq < frequency_max)],
	         amplitude_spectrum[(fft_freq > frequency_min) & (fft_freq < frequency_max)])
	plt.plot(fft_freq[maxima_indices][(fft_freq[maxima_indices] > frequency_min) & (fft_freq[maxima_indices] < frequency_max)],
	         amplitude_spectrum[maxima_indices][(fft_freq[maxima_indices] > frequency_min) & (fft_freq[maxima_indices] < frequency_max)],
	         'ro', label='Maxima')
	plt.xlabel('Frequency')
	plt.ylabel('Amplitude Spectrum')
	plt.title(f'Market Cycle Distribution of {ticker}')
	plt.grid(True)
	plt.legend()

	# Label the first maxima with their corresponding x-values on the spectral distribution plot
	for index in maxima_indices:
	    plt.annotate(f'{fft_freq[index]:.4f}', xy=(fft_freq[index], amplitude_spectrum[index]),
	                 xytext=(0, -10), textcoords='offset points', ha='center', va='top', fontsize=8, color='blue')

	# Plot the stock market data and its SMA
	plt.subplot(2, 1, 2)
	plt.plot(prices, label='Data')
	plt.plot(SmaLong, label='Long-term SMA')
	plt.plot(SmaShort, label='Short-term SMA')
	plt.xlabel('Time')
	plt.ylabel('Price')
	plt.title(f'Stock Market Data and Long/Short SMA of {ticker} ')
	plt.grid(True)
	plt.legend()


	# Find crossings for SmaLong
	crosses_long = np.where(np.diff(np.sign(prices - SmaLong)) != 0)[0]

	# Find crossings for SmaShort
	crosses_short = np.where(np.diff(np.sign(prices - SmaShort)) != 0)[0]

	# Initialize variables for max difference and indices for SmaLong
	max_difference_long = 0
	max_difference_index_long = -1

	# Find the adjacent crossing points with the maximum difference in y-values for SmaLong
	for i in range(len(crosses_long) - 1):
	    current_cross_long = crosses_long[i]
	    next_cross_long = crosses_long[i + 1]
	    if prices[current_cross_long] > prices[next_cross_long]:
	        difference_long = prices[current_cross_long] - prices[next_cross_long]
	        if difference_long > max_difference_long:
	            max_difference_long = difference_long
	            max_difference_index_long = i


	# Find the adjacent crossing points with the maximum difference in y-values for SmaShort
	max_differences_short = []  # List to store the maximum differences
	max_differences_indexes_short = []  # List to store the indexes of maximum differences
	min_price_indexes_short = [] # list to store indexes of minimum price during fall

	for i in range(len(crosses_short) - 1):
	    current_cross_short = crosses_short[i]
	    next_cross_short = crosses_short[i + 1]
	    if prices[current_cross_short] > prices[next_cross_short] and (next_cross_short - current_cross_short) <= 7:
	        difference_short = prices[current_cross_short] - prices[next_cross_short]
	        if difference_short > 0:
	            max_differences_short.append(difference_short)
	            max_differences_indexes_short.append(i)

	# Sort the maximum differences in descending order. Zip takes in multiple iterables
	sorted_differences_short = sorted(zip(max_differences_short, max_differences_indexes_short), reverse=True)
	# Retrieve up to 5 maximum point pairs
	top_5_point_pairs_short = sorted_differences_short[:5]
	# Extract the top 5 maximum differences
	top_5_diffrences_short = [fall for fall, _ in top_5_point_pairs_short]
    # Index for the crossing point start of maximum fall
	max_difference_index_short = top_5_point_pairs_short[0][1]
	# Find average difference
	average_fall_short = sum(top_5_diffrences_short) / len(top_5_diffrences_short)

	# Plot the line between the adjacent crossing points with maximum difference in y-values for SmaLong
	if max_difference_index_long >= 0:
	    current_cross_long = crosses_long[max_difference_index_long]
	    next_cross_long = crosses_long[max_difference_index_long + 1]
	    min_price_index_long = np.argmin(prices[current_cross_long: next_cross_long + 1]) + current_cross_long
	    plt.plot([current_cross_long, next_cross_long], [prices[current_cross_long], prices[next_cross_long]], 'r--')
	    plt.plot([current_cross_long, min_price_index_long], [prices[current_cross_long], prices[min_price_index_long]], 'r--')
	    plt.plot([min_price_index_long, next_cross_long], [prices[min_price_index_long], prices[next_cross_long]], 'r--')
	    plt.annotate(f'{prices[current_cross_long]:.2f}', xy=(current_cross_long, prices[current_cross_long]),
	                 xytext=(0, -10), textcoords='offset points', ha='center', va='top', fontsize=6, color='red')
	    plt.annotate(f'{prices[next_cross_long]:.2f}', xy=(next_cross_long, prices[next_cross_long]),
	                 xytext=(0, -10), textcoords='offset points', ha='center', va='top', fontsize=6, color='red')
	    plt.annotate(f'{prices[min_price_index_long]:.2f}', xy=(min_price_index_long, prices[min_price_index_long]),
	                 xytext=(0, -10), textcoords='offset points', ha='center', va='top', fontsize=6, color='red')


	# Plot the line between the adjacent crossing points with maximum difference in y-values for SmaShort
	if len(top_5_diffrences_short) > 0:
	    current_cross_short = crosses_short[max_difference_index_short]
	    next_cross_short = crosses_short[max_difference_index_short + 1]
	    min_price_short_index = np.argmin(prices[current_cross_short: next_cross_short + 1]) + current_cross_short
	    plt.plot([current_cross_short, min_price_short_index], [prices[current_cross_short], prices[min_price_short_index]], 'g--')
	    plt.plot([min_price_short_index, next_cross_short], [prices[min_price_short_index], prices[next_cross_short]], 'g--')
	    plt.plot([current_cross_short, next_cross_short], [prices[current_cross_short], prices[next_cross_short]], 'g--')
	    plt.annotate(f'{prices[current_cross_short]:.2f}', xy=(current_cross_short, prices[current_cross_short]),
	                 xytext=(0, -10), textcoords='offset points', ha='center', va='top', fontsize=6, color='green')
	    plt.annotate(f'{prices[next_cross_short]:.2f}', xy=(next_cross_short, prices[next_cross_short]),
	                 xytext=(0, -10), textcoords='offset points', ha='center', va='top', fontsize=6, color='green')
	    plt.annotate(f'{prices[min_price_short_index]:.2f}', xy=(min_price_short_index, prices[min_price_short_index]),
	                 xytext=(0, -10), textcoords='offset points', ha='center', va='top', fontsize=6, color='green')


	# Market Model paramater calculations - Adaptive Model: 1_(x --> 2x --> 4x) --> 2_(16x --> 64x) 
	# Short-term factor calculation
	safety_factor = 5
	#x_1 = (((average_fall_short) * 100) / (4 * prices[current_cross_short])) 
	x_1 = (((prices[current_cross_short] - prices[min_price_short_index]) * 100) / (4 * prices[current_cross_short])) * 0.75
	fall_difference = (prices[current_cross_long] - prices[min_price_index_long]) - (prices[current_cross_short] - prices[min_price_short_index])
	fall_difference = (prices[current_cross_long] - prices[min_price_index_long]) - (average_fall_short)
	x_2 = (((fall_difference * 100) /prices[current_cross_long]) + safety_factor)/64
	s_1 = x_1
	s_2 = 2 * x_1
	s_3 = 4 * x_1
	s_4 = (4 * (x_1 / 0.75)) + (16 * x_2)
	s_5 = (4 * (x_1 / 0.75)) + (64 * x_2)

	trade_sequence_label.configure(text=f"Trade Sequence - {s_1:.2f} --> {s_2:.2f} --> {s_3:.2f} --> {s_4:.2f} --> {s_5:.2f}")
	plt.tight_layout()
	plt.show()


# Create the main window
window = tk.Tk()
window.title("Adaptive Investment - 1.0")

# Set the background image
bg_image = Image.open("/Users/abdulkhadertm/Documents/DayTradeModel/SpectrumGenerator/background.jpg")  # Replace "background_image.jpg" with your image file
bg_image = bg_image.resize((800, 600))  # Resize the image to fit the window size
background_image = ImageTk.PhotoImage(bg_image)
background_label = tk.Label(window, image=background_image)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

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

# Create the trade sequence label
trade_sequence_label = tk.Label(window, text="Trade Sequence - ")
trade_sequence_label.grid(row=5, column=0, columnspan=2, pady=5)

# Configure padding and stretching for all rows and columns
for i in range(6):
	window.grid_rowconfigure(i, pad=10)
	window.grid_columnconfigure(0, weight=1)
	window.grid_columnconfigure(1, weight=1)

# Start the GUI main loop
window.mainloop()





