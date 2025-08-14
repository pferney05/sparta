# Rerun of the Reference numerical validation dataset
import time
import numpy as np
import pandas as pds
import sparta as acl
import canvaslib2 as can

def create_signaloffset(offsetReactivity, reactivityStep,reactivitySlope,nCycle,cycleDuration,rampDuration,factorStep,factorSlope,PK,frequency, **kwargs):
    tPoint = [0.]
    rPoint = [offsetReactivity]
    fPoint = [1.]
    cPoints = [[0.],[0.]]
    for i in range(1,nCycle+1):
        tStepStart  = i * cycleDuration - rampDuration
        tStepEnd    = i * cycleDuration
        rStepStart  = reactivitySlope * tStepStart + (i-1) * reactivityStep + offsetReactivity
        rStepEnd    = reactivitySlope * tStepEnd   +  i    * reactivityStep + offsetReactivity
        fStepStart  = 1. + factorSlope * tStepStart + (i-1) * factorStep
        fStepEnd    = 1. + factorSlope * tStepEnd   +  i    * factorStep
        cStepStart  = (i-1) * abs(reactivityStep)
        cStepEnd    =  i    * abs(reactivityStep)
        cSlopeStart = abs(reactivitySlope) * tStepStart
        cSlopeEnd   = abs(reactivitySlope) * tStepEnd
        # Add every data in the point lists
        tPoint.append(tStepStart)
        tPoint.append(tStepEnd)
        rPoint.append(rStepStart)
        rPoint.append(rStepEnd)
        fPoint.append(fStepStart)
        fPoint.append(fStepEnd)
        cPoints[0].append(cStepStart)
        cPoints[0].append(cStepEnd)
        cPoints[1].append(cSlopeStart)
        cPoints[1].append(cSlopeEnd)
    t0 = tPoint[0]
    t1 = tPoint[1]
    r0 = rPoint[0]
    r1 = rPoint[1]
    f0 = fPoint[0]
    f1 = fPoint[1]
    c00 = cPoints[0][0]
    c01 = cPoints[0][1]
    c10 = cPoints[1][0]
    c11 = cPoints[1][1]
    tn = (r1*t0 - r0*t1)/(r1-r0)
    fn = tn*(f1-f0)/(t1-t0) + (t1*f0-t0*f1)/(t1-t0)
    tPoint[0] = tn
    rPoint[0] = 0.
    fPoint[0] = fn
    cPoints[0][0] = tn*(c01-c00)/(t1-t0) + (t1*c00-t0*c01)/(t1-t0)
    cPoints[1][0] = tn*(c11-c10)/(t1-t0) + (t1*c10-t0*c11)/(t1-t0)
    myNumeric = acl.Numerical([x-tn for x in tPoint], rPoint, cPoints, PK, ['step', 'slope'], frequency, **kwargs)
    myNumeric.factors([x/fn for x in fPoint])
    return myNumeric

def calculation_rfPoints(signal, solver, rSlope, fSlope):
    tPoint0 = np.interp(solver.configPoints[0], signal.cPoints[0], signal.tPoint)
    tPoint1 = np.interp(solver.configPoints[1], signal.cPoints[1], signal.tPoint)
    rPointSlope = rSlope * tPoint1
    fPointSlope = 1 + fSlope * tPoint1
    rPointStep  = np.interp(tPoint0, signal.tPoint, signal.rPoint) - (rSlope * tPoint0)
    fPointStep  = np.interp(tPoint0, signal.tPoint, signal.dataDict['fPoint']) - ((fSlope * tPoint0))
    return [rPointStep, rPointSlope], [fPointStep, fPointSlope]

# 2 in. of all CS at 25 in -> around 300 pcm
# Based on Ben Baker work you get around 2% flux shape deformation

# TREAT kinetic parameters - Reactor PhysicsMeasurements in TREAT, ANL-6174 ANL-6173
L = 9.e-4
B = 717.8e-5
ai = [0.03299957, 0.21900027, 0.19599953, 0.39500059, 0.11500059, 0.04199945]
li = [0.0124,0.0305,0.1110,0.3010,1.1300,3.0000]
PK = acl.PointKinetic(L, B, ai, li)

# ****************************** SIGNAL DEFINITION
offsetReactivity = +0.8950e-5  # reactivity
reactivityStep   = +2.0000e-5  # reactivity / step
reactivitySlope  = -0.0625e-5  # reactivity / s
nCycle           =  50         # Steps 
cycleDuration    =  32.        # Step duration (s)
rampDuration     =  1.5        # Ramp duration (s)
factorStep       = +0.0008     # factor / step
factorSlope      = -0.00001875 # factor / second
# ************************************************
signal = create_signaloffset(offsetReactivity,reactivityStep,reactivitySlope,nCycle,cycleDuration,rampDuration,factorStep,factorSlope,PK, 100.,useInvertedReactimeter = True)
stepArray = signal.df['step'].to_numpy()
slopeArray = signal.df['slope'].to_numpy()
rPoint = signal.rPoint
fPoint = signal.dataDict['fPoint']

# solver definition
solver = acl.SPARTA(signal.t, signal.n, [stepArray, slopeArray], ['step', 'slope'], [1, 1], PK )
rPoints, fPoints = calculation_rfPoints(signal, solver, reactivitySlope, factorSlope)

# Computation
t0 = time.time()
rCalcs, fCalcs = solver.main_loop(ftol = 5.e-16, xtol = 5.e-16, display=True, interactive=False)
t1 = time.time()

# Print Out
print('------------')
print('FINISHED JOB')
print('------------')
print(f'rCalc : {[rCalc.tolist() for rCalc in rCalcs]}')
print(f'fCalc : {[fCalc.tolist() for fCalc in fCalcs]}')
for i in range(0,len(rCalcs)):
    rCalc = rCalcs[i]
    fCalc = fCalcs[i]
    rPoint = rPoints[i]
    fPoint = fPoints[i]
    print(f'CVector : {solver.configNames[i]} - REACTIVITY DIFFERENCE [pcm]', 1.e5*(rCalc - rPoint))
    print(f'CVector : {solver.configNames[i]} - FACTOR DIFFERENCE', fCalc - fPoint)
print('TIME', t1 - t0, 'SEC')

# ------------------------------------------------ Graphics

address = "./"
name    = "example"
xParams = solver.get_xParams(rCalcs, fCalcs)
nf, r, f = solver.fitting_function(xParams)
rref = np.interp(solver.t, signal.tPoint, signal.rPoint)
fref = np.interp(solver.t, signal.tPoint, signal.dataDict["fPoint"])

# Graphique 1 - True Residual
print("STATUS:",solver.output.status)
residual = (nf - solver.n)/solver.n
xmin1, xmax1 = np.min(solver.t), np.max(solver.t)
yabs1 = 10**int(np.log10(np.max(np.abs(residual)))+17)
graphic99 = can.Graphic(xmin1, xmax1, -yabs1, +yabs1, "Time [s]", r"(Fit - Signal)/Signal [$\times 10^{-16}$]", dpi=300, _wparams=[0.18,0.12,0.78,0.84])
graphic99.line_plot(solver.t , 1e16*solver.output.fun, lw=0.75)
graphic99.save(f"{address}/{name}_true_residual.png")

# Graphique 1 - Residual
residual = (nf - solver.n)/solver.n
xmin1, xmax1 = np.min(solver.t), np.max(solver.t)
yabs1 = 10**int(np.log10(np.max(np.abs(residual)))+17)
graphic01 = can.Graphic(xmin1, xmax1, -yabs1, +yabs1, "Time [s]", r"(Fit - Signal)/Signal [$\times 10^{-16}$]", dpi=300, _wparams=[0.18,0.12,0.78,0.84])
graphic01.line_plot(solver.t , 1e16*residual, lw=0.75)
graphic01.save(f"{address}/{name}_residual.png")

# Graphique 2 - Signal
xmin2, xmax2 = np.min(solver.t), np.max(solver.t)
ymin2, ymax2 = 0.985, 1.015
graphic02 = can.Graphic(xmin2, xmax2, ymin2, ymax2, "Time [s]", "Signal", dpi=300)
graphic02.line_plot(solver.t , solver.n/f, lw=0.75, color=3, legend="Flux amplitude (Corrected signal)")
graphic02.line_plot(solver.t , solver.n, lw=0.75, legend="Signal")
graphic02.legend()
graphic02.save(f"{address}/{name}_signal.png")

# Graphique 3 - Monitored Quantities
xmin3, xmax3 = np.min(solver.t), np.max(solver.t)
ymin3, ymax3 = 0., 1.
qty0 = (solver.configList[0] - np.min(solver.configList[0]))/(np.max(solver.configList[0]) - np.min(solver.configList[0]))
qty1 = (solver.configList[1] - np.min(solver.configList[1]))/(np.max(solver.configList[1]) - np.min(solver.configList[1]))
graphic03 = can.Graphic(xmin3, xmax3, ymin3, ymax3, "Time [s]", "Value of the input quantity", dpi=300)
graphic03.line_plot(solver.t , qty1, lw=0.75, color="#FF0000", legend="Temperature")
graphic03.line_plot(solver.t , qty0, lw=0.75, color="#0000FF", legend="Rods position")
graphic03.legend()
graphic03.save(f"{address}/{name}_monitor.png")

# Graphique 4 - Reactivity
xmin4, xmax4 = np.min(solver.t), np.max(solver.t)
ymin4, ymax4 = -2., 2.
graphic04 = can.Graphic(xmin4, xmax4, ymin4, ymax4, "Time [s]", "Reactivity [pcm]", dpi=300)
graphic04.line_plot(solver.t, 1e5*rref, lw=0.75)
graphic04.save(f"{address}/{name}_reactivity.png")

# Graphique 5 - Factors
xmin5, xmax5 = np.min(solver.t), np.max(solver.t)
ymin5, ymax5 = 0.97, 1.03
graphic05 = can.Graphic(xmin5, xmax5, ymin5, ymax5, "Time [s]", "Correction factors", dpi=300)
graphic05.line_plot(solver.t, fref, lw=0.75)
graphic05.save(f"{address}/{name}_factors.png")

# Graphique 6 - Reactivity Characteristics
xmin6, xmax6 = 0., 1.
ymin6, ymax6 = -100., +100.
qty0min, qty0max = np.min(solver.configList[0]), np.max(solver.configList[0])
qty1min, qty1max = np.min(solver.configList[1]), np.max(solver.configList[1])
poly0 = solver.create_rPolynomial(0, xParams)
poly1 = solver.create_rPolynomial(1, xParams)
myRange = np.linspace(0., 1., 101, endpoint=True)
c0 = qty0min + myRange*(qty0max-qty0min)
c1 = qty1min + myRange*(qty1max-qty1min)
graphic06 = can.Graphic(xmin6, xmax6, ymin6, ymax6, "Quantity [A.U.]", "Reactivity [pcm]", dpi=300)
graphic06.line_plot(myRange, 1e5*poly0(c0), lw=0.75, color="#FF0000", legend="Temperature")
graphic06.line_plot(myRange, 1e5*poly1(c1), lw=0.75, color="#0000FF", legend="Rods position")
graphic06.legend()
graphic06.save(f"{address}/{name}_rChar.png")

# Graphique 7 - Reactivity Characteristics
xmin7, xmax7 = 0., 1.
ymin7, ymax7 = 0.95, 1.05
qty0min, qty0max = np.min(solver.configList[0]), np.max(solver.configList[0])
qty1min, qty1max = np.min(solver.configList[1]), np.max(solver.configList[1])
poly0 = solver.create_fPolynomial(0, xParams)
poly1 = solver.create_fPolynomial(1, xParams)
myRange = np.linspace(0., 1., 101, endpoint=True)
c0 = qty0min + myRange*(qty0max-qty0min)
c1 = qty1min + myRange*(qty1max-qty1min)
graphic07 = can.Graphic(xmin7, xmax7, ymin7, ymax7, "Quantity [A.U.]", "Correction factors", dpi=300)
graphic07.line_plot(myRange, poly0(c0), lw=0.75, color="#FF0000", legend="Temperature")
graphic07.line_plot(myRange, poly1(c1), lw=0.75, color="#0000FF", legend="Rods position")
graphic07.legend()
graphic07.save(f"{address}/{name}_fChar.png")

# Graphique 8 - Reactivity difference
difference8 = 1e5*(r - rref)
xmin8, xmax8 = np.min(solver.t), np.max(solver.t)
ymin8, ymax8 = np.min(difference8), np.max(difference8)
graphic08 = can.Graphic(xmin8, xmax8, 1e16*ymin8, 1e16*ymax8, "Time [s]", r"Reactivity difference [$\times 10^{-16}$ pcm]", dpi=300, _wparams=[0.18,0.12,0.78,0.84])
graphic08.line_plot(solver.t, 1e16*difference8, lw=0.75)
graphic08.save(f"{address}/{name}_rDiff.png")

# Graphique 9 - Factors difference
difference9 = (f - fref)
xmin9, xmax9 = np.min(solver.t), np.max(solver.t)
ymin9, ymax9 = np.min(difference9), np.max(difference9)
graphic09 = can.Graphic(xmin9, xmax9, 1e16*ymin9, 1e16*ymax9, "Time [s]", r"Correction factors difference [$\times 10^{-16}$]", dpi=300, _wparams=[0.18,0.12,0.78,0.84])
graphic09.line_plot(solver.t, 1e16*difference9, lw=0.75)
graphic09.save(f"{address}/{name}_fDiff.png")

# ------------------------------------------------ Tables
def create_table(data:dict, filename):
    newParams = {}
    for key, value in data.items():
        newParams[key] = [value]
    df = pds.DataFrame.from_dict(newParams, orient='columns')
    df.to_csv(filename, index=False)

# Table 1 - mean & std of r - rref and f - fref 
dataDict01 = {
    "rMean":1e5*np.mean(r - rref),
    "rStd":1e5*np.std(r - rref),
    "fMean":np.mean(f - fref),
    "fStd":np.std(f - fref),
}
create_table(dataDict01, f"{address}/{name}_table01.csv")
