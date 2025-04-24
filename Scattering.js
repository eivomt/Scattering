function linspace(startValue, stopValue, cardinality) {
    var arr = [];
    var step = (stopValue - startValue) / (cardinality - 1);
    for (var i = 0; i < cardinality; i++) {
      arr.push(startValue + (step * i));
    }
    return arr;
}

let line
let log = true

function setup() {

    createCanvas(1024,1024)
    background(150)

    L = 300
    N = 128
    

    Tfinal = 150
    t = 0
    dt =.5

    V0 = 2
    s = 15
    width = 3

    Gamma0 = 1e-4
    Onset = 90

    p0 = 1
    sigmaX = 2
    sigmaP = 1/(2*sigmaX)
    tau = 0

    x0 = -10*sigmaX-width/2

    x = linspace(-L/2, L/2, N)
    h = L/(N-1)

    // Set up vector of k-values
    k_max = Math.PI / h
    dk = 2*k_max/N

    kPositive = linspace(0,k_max,N/2)
    kNegative = linspace(-k_max,-dk, N/2)
    // k = Array.prototype.push.apply(kNegative,kPositive)
    k = [...kPositive, ...kNegative]

    // Tmat_FFT = createMatrix(N,N, math.complex(0,0)) // dtype = complex
    // er ikke dette unødvendig hvis neste steg er;

    Tmat_FFT = math.fft(math.identity(N))
    ksquared = k.map(val => val * -1)
    ksquared = math.map(k, math.square)
    ksquared = math.diag(ksquared)
    // ksquared = k.map(val => val * val)
    Tmat_FFT = math.multiply(ksquared, Tmat_FFT)
    Tmat_FFT = math.ifft(math.transpose(Tmat_FFT))
    Tmat_FFT = math.multiply(-1/2, Tmat_FFT)

    // Potential
    Vpot = x.map(val => V0/(math.exp(s*(Math.abs(val)-width/2)) + 1.0))

    //Absorber
    let Gamma = x.map(val => {
        let condition = Math.abs(val) > Onset ? 1 : 0;
        let shifted = Math.abs(val) - Onset;
        return condition * Gamma0 * shifted;
    })
    Gamma = math.map(Gamma, math.square)
    // Gamma = Gamma.map(val => math.complex(val))
    // Gamma = Gamma.map(val => val * 10000)

    // Full Hamiltonian
    // Ham = Gamma.map(val => math.multiply(math.complex(0,-1), val))
    // Ham = math.add(Vpot, Tmat_FFT)
    Ham = math.diag(Vpot)
    Ham = math.add(Tmat_FFT, Ham)

    matrixExponential = Ham.map(val => math.multiply(dt, (math.multiply(val, math.complex(0,-1)))))
    U = math.expm(matrixExponential)

    // InitialNorm = Math.pow(2/Math.PI, 1/4) * Math.sqrt(sigmaP/math.multiply(math.complex(1,-2),sigmaP)**(2*tau)) // tau = 0 
    InitialNorm = Math.pow(2/Math.PI, 1/4) * math.sqrt(sigmaP) // tau = 0
    console.log(InitialNorm)
    console.log(x)
    // Psi0 = x.map( val => InitialNorm * math.exp(minusSigmaP**2*(val-x0)**2 + math.complex(0,1)*p0*val))
    Psi0 = x.map( val => InitialNorm * math.exp(-1*sigmaP**2*(val-x0)**2))
    Psi0 = x.map((val, index) => math.multiply(Psi0[index], math.multiply(math.complex(0,1),p0,val)))
    // Psi0 = x.map(val => Psi0 * (val - x0)**2 * p0 * val)

    Psi = Psi0.map(val => val == Infinity ? 0 : val)


    console.log('Psi0;')
    console.log(Psi0)
    console.log('k;')
    console.log(k)
    console.log(typeof(k))
    console.log('Tmat_FFT;')
    console.log(Tmat_FFT)
    console.log(typeof(Tmat_FFT))
    console.log('Vpot;')
    console.log(Vpot)
    console.log(typeof(Vpot))
    console.log('Gamma;')
    console.log(Gamma)
    console.log(typeof(Gamma))
    console.log('Hamiltonian;')
    console.log(Ham)
    console.log(typeof(Ham))
    console.log('U;')
    console.log(U)
}



function draw() {
    translate(0, windowHeight/2)
    if (t<Tfinal) {
        if(log) {
            console.log('Psi before multiplication with U')
            console.log(Psi)
        }
        Psi = math.multiply(U, Psi)
        if(log) {
            console.log('Psi after multiplication with U')
            console.log(Psi)
        }
        line = Psi.map(val => val==Infinity ? 0 : math.multiply(math.abs(val), math.abs(val)))
    }
    t = t+dt

    stroke(0)
    strokeWeight(2)
    noFill()

    beginShape()
    for (let i = 0; i < line.length; i++) {
        vertex(i, line[i])
        if(log) {
            console.log(i)
            console.log(line[i])
        }
    }
    endShape()

    beginShape()
    for(let i=0; i< Psi0.length; i++) {
        vertex(math.abs(Psi0)**2*10, i*10)
    }
    endShape()

    beginShape()
    for (let i = 0; i < Vpot.length; i++) {
        vertex(i *10, -1*Vpot[i]*10)
    }
    endShape()

    if (log) {
        console.log('line')
        console.log(line)
    }
    log = false
}