import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

filename = 'logs_save/8797/s2ef_predictions.npz'

data = np.load(filename)
energies = data['energy']

# Histogram: Shows the distribution as a histogram
plt.figure(figsize=(8, 6))
plt.hist(energies, bins=30, density=True, alpha=0.7)  # 'density=True' normalizes the histogram
plt.title('Distribution of Predicted Energies for O Absorabtes')
plt.xlabel('Predicted Energy')
plt.ylabel('Density')
plt.grid(True)
plt.savefig('stats/histogram.png')

# KDE Plot: Uses a kernel density estimate to smooth the distribution
density = gaussian_kde(energies)
xs = np.linspace(np.min(energies), np.max(energies), 200)

plt.figure(figsize=(8, 6))
plt.plot(xs, density(xs), linewidth=2)
plt.fill_between(xs, density(xs), alpha=0.3)
plt.title('Kernel Density Estimate of Predicted Energies')
plt.xlabel('Predicted Energy')
plt.ylabel('Density')
plt.grid(True)
plt.savefig('stats/kde.png')
