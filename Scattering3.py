"""
This script simulates the evolution of an intially Gaussian wave packet 
which hits a barrier. The barrier has a "smoothly rectangular" shape.
In addition to simulating the evolution of the wave packet, the script
estimates the transmission and reflection probabilities after the 
collision.

 Numerical inputs:
   L       - The extension of the spatial grid 
   N       - The number of grid points
   Tfinal  - The duration of the simulation
   dt      - The step size in time

 Inputs for the initial Gaussian:
   x0      - The mean position of the initial wave packet
   p0      - The mean momentum of the initial wave packet
   sigmaP  - The momentum width of the initial, Gaussian wave packet
   tau     - The time at which the Gaussian is narrowest (spatially)
 
 Input for the barrier:
   V0      - The height of the barrier (can be negative)
   w       - The width of the barrier
   s       - Smoothness parameter

 All inputs are hard coded initially.
"""

# Import libraries  
import numpy as np
from matplotlib import pyplot as plt
from scipy import linalg
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button, Slider
import os

# def setStringFile(Lnumber, figNum, iterator=""):
#     return './' + 'L' + str(Lnumber) + '/' + str(imageIterator) + '/fig' + str(figNum) + '/' + 'fig'+ iterator +'.png'
def setStringFile(imageNum, directory):
    return './png/' + str(directory) + '/' + str("{:03d}".format(imageNum)) +'.png'

imageNum = 1
filmIterator = 1

# Numerical grid parameters
L = 500.0
N = 2048       # Should be 2**k, with k being an integer
# N = 1024       # Should be 2**k, with k being an integer

# Simulation on/off boolean
running = True

# Numerical time parameters
Tfinal = 150
dt = 0.4

# Inputs for the smoothly rectangular potential
V0Array = [-1,0,1,2,3]
V0Iterator = 0
V0 = V0Array[0]          # Height
s = 30.0             # Smoothness
width = 3.0         # Width

# Inputs for the absorbing potential
Gamma0 = 1e-4
Onset = 90

# Inputs for the Gaussian
p0Array = [.5,1,2]
p0Iterator = 0
p0 = p0Array[0]
sigmaX = 2
sigmaP = 1/(2*sigmaX)
tau = 0.0

# Initial mean position
x0 = -12*sigmaX - width/2

# Set up grid
x = np.linspace(-L/2, L/2, N)
h = L/(N-1)

# Window fixation values
xMin = -100
xMax = 100
yMin = -0.100
yMax = 0.350

# Determine double derivative by means of the fast Fourier transform.
# Set up vector of k-values
k_max = np.pi/h
dk = 2*k_max/N
k = np.append(np.linspace(0, k_max-dk, int(N/2)), 
              np.linspace(-k_max, -dk, int(N/2)))
# Allocate and declare
Tmat_FFT = np.zeros((N, N), dtype=complex)
# Transform identity matrix
Tmat_FFT = np.fft.fft(np.identity(N, dtype=complex))
# Multiply by (ik)^2
Tmat_FFT = np.matmul(np.diag(-k**2), Tmat_FFT)
# Transform back to x-representation. 
# Transpose necessary as we want to transform columnwise
Tmat_FFT = np.fft.ifft(np.transpose(Tmat_FFT))
# Correct pre-factor
Tmat_FFT = -1/2*Tmat_FFT    

# Add potential
Vpot = V0/(np.exp(s*(np.abs(x)-width/2))+1.0)
# Absorber
Gamma = (np.abs(x) > Onset)*Gamma0*(np.abs(x) - Onset)**2

# Full Hamiltonian
Ham = Tmat_FFT + np.diag(Vpot -1j*Gamma)
# Propagator
U = linalg.expm(-1j*Ham*dt)

# Set up Gaussian - analytically
InitialNorm = np.power(2/np.pi, 1/4) * np.sqrt(sigmaP/(1-2j*sigmaP**2*tau))
# Initial Gaussian
Psi0 = InitialNorm*np.exp(-sigmaP**2*(x-x0)**2/(1-2j*sigmaP**2*tau)+1j*p0*x)

# Initiate plots
plt.ion()
fig = plt.figure(1, figsize=(15,8))
plt.clf()

# Fix window
plt.xlim(xMin, xMax)
plt.ylim(yMin, yMax)


# Scaling and plotting the potential
Psi0Max = np.max(np.abs(Psi0)**2)
line2, = plt.plot(x, V0 * Psi0Max/(np.exp(s*(np.abs(x)-width/2))+2.5), '-', 
                 color='#333', label = 'FD3', linewidth=2)

# Plot line
line1, = plt.plot(x, np.abs(Psi0)**2, '-', color='#2073E8', 
                 label = 'Analytical', linewidth=4)
# Initiate wave functons and time
# Turn Psi0 into N \times 1 matrix (column vector)
Psi0 = np.matrix(Psi0)                 
Psi0 = Psi0.reshape(N,1)
Psi = Psi0
t = 0

plt.axis('off')
plt.xticks([])  
plt.yticks([])


while(V0Iterator < len(V0Array)):

  while(p0Iterator < len(p0Array)):
    imageNum = 0
    p0 = p0Array[p0Iterator]
    
    # Initial Gaussian
    Psi0 = InitialNorm*np.exp(-sigmaP**2*(x-x0)**2/(1-2j*sigmaP**2*tau)+1j*p0*x)
    # Plot line

    # Scaling and plotting the potential
    Psi0Max = np.max(np.abs(Psi0)**2)

    # Initiate wave functons and time
    # Turn Psi0 into N \times 1 matrix (column vector)
    Psi0 = np.matrix(Psi0)                 
    Psi0 = Psi0.reshape(N,1)
    Psi = Psi0

    # line1.set_ydata(np.power(np.abs(Psi), 2))
    imageNum = 0
    t = 0

    for filename in os.listdir('./png/' + str(filmIterator)):
      os.remove('./png/' + str(filmIterator) + '/' + filename)
    
    # while (t < (np.abs(x0) / p0) or Integral > 1e-4) and running:
    # while (t < (np.abs(x0) / p0)) and running:
    while (t < 200) and running:
      # Update time
      t = t+dt              
      Psi = np.matmul(U, Psi)
      # Update data for plots
      line1.set_ydata(np.power(np.abs(Psi), 2))

      Integral = np.trapz(np.power(np.abs(np.asarray(Psi.T)), 2)*(np.abs(x) < 2 * width), dx = h)
      strFile = setStringFile(imageNum, filmIterator)
      if os.path.isfile(strFile):
        os.remove(strFile)
      plt.margins(0)
      plt.subplots_adjust(left=0, right=1, top=1, bottom=0) 
      plt.savefig(strFile, facecolor="#F1F1F1", edgecolor='none')
      # plt.clf()
      # plt.close(fig)
      # fig.canvas.draw()
      # fig.canvas.flush_events()
      # plt.plot
      imageNum += 1

    p0Iterator += 1
    filmIterator += 1
    print(p0Iterator)

  p0Iterator = 0
  V0Iterator += 1

  # Add potential
  Vpot = V0Array[V0Iterator]/(np.exp(s*(np.abs(x)-width/2))+1.0)
  # Absorber
  Gamma = (np.abs(x) > Onset)*Gamma0*(np.abs(x) - Onset)**2

  # Full Hamiltonian
  Ham = Tmat_FFT + np.diag(Vpot -1j*Gamma)
  # Propagator
  U = linalg.expm(-1j*Ham*dt)
  line2.set_ydata(V0Array[V0Iterator] * Psi0Max/(np.exp(s*(np.abs(x)-width/2))+2.5))
# Update plots
# 

