# Scattering

## To View
Open `index.html` in your browser.

---

## To Change Variables

Edit `Scattering3.py`.

### Numerical Inputs:
- `L` – Extension of the spatial grid  
- `N` – Number of grid points  
- `Tfinal` – Duration of the simulation  
- `dt` – Time step size  

### Initial Gaussian Wave Packet:
- `x0` – Mean position of the initial wave packet  
- `p0` – Mean momentum of the initial wave packet  
- `sigmaP` – Momentum width of the Gaussian wave packet  
- `tau` – Time at which the Gaussian is narrowest (spatially)  

### Barrier Parameters:
- `V0` – Height of the barrier (can be negative)  
- `w` – Width of the barrier  
- `s` – Smoothness parameter  

---

To change `V0` or `p0`, modify the arrays:
- `V0Array`
- `p0Array`

---

## How to Run

Run Python and shell script:

```bash
python Scattering3.py
./make_video.sh