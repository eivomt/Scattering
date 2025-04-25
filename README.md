# Scattering

# To view; 
open index.html in browser

# To change variables;
Scattering3.py

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

to change V0 or p0 change the corresponding arrays
V0Array and/or p0Array

run the python script,
python Scattering3.py

directories in ./png/1..15 should fill with your new images.

now run the shellscript to convert images to .mp4 files
make_video.sh

change innerHTML of buttons in index.html as needed.