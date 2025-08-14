import numpy as np
from acralib_src.detector import Detector
from acralib_src.pointkinetic import PointKinetic
from acralib_src.acralib_typing import *

class Numerical(Detector):
    """[This Class is still under development aand major changes can still occur]
    Specific class to generate signals for numerical validation of the ACRA_V2 workflow. It inherites from the `Signal` class.
    There are at least 3 columns: `time`, `signal` and all the control quantities that can be rod positions, fuel temperature...
    The default columns names for control quantities are `default_0`, `default_1` etc.

    Attributes
    ----------
    df: pandas.core.frame.DataFrame
        Data frame containing the data asscoiated to the signal.
    t: numpy.ndarray[float]
        Array containing the time values. Is recovered from `df`.
    n: numpy.ndarray[float]
        Array containing the signal values. Is recovered from `df`.
    r: numpy.ndarray[float]
        Array containing the reactivity values at each time step that built the signal. Is obtained with the `self.interp` function.
    tPoint: numpy.ndarray[float]
        Array containing the time data of points used to generate the signal.
    rPoint: numpy.ndarray[float]
        Array containing the reactivity data of points used to generate the signal.
    cPoints: numpy.ndarray[float]
        2D-array containing the values of each control quantity for points used to generate the signal.
    dataDict: dict
        Dictionnary containing the base data input of functions that transform the signal (e.g., noise, spatial effects).
    PK: PointKinetic
        Point kinetic object to be used in the generation of the signal from point kinetic equations.
    _controlWeights: numpy.ndarray[float]
        Weight associated to each control quantity at each time step.
    _interp: interpFunc
        Interpolation function to interpolate point quantities to array quantities with the same length as t.

    """
    _stdPK = PointKinetic(L=25e-6, B=760e-5)

    def __init__(self, tPoint:np.ndarray[float], rPoint:np.ndarray[float], cPoints:list[np.ndarray[float]], PK:PointKinetic = _stdPK, controlNames:Optional[list[str]] = None, frequency:float = 100., 
                 _wPoints:Optional[np.ndarray[float]] = None, _interp: interpFunc = np.interp, useInvertedReactimeter = False, **kwargs) -> None:
        """Initialisation function of the Numerical class
        
        Parameters
        ----------
        tPoint: np.ndarray[float]
            Array containing the time data of points used to generate the signal.   
        rPoint: np.ndarray[float]
            Array containing the reactivity data of points used to generate the signal.
        cPoints: list[np.ndarray[float]]
            2D-array containing the values of each control quantity for points used to generate the signal.
        PK: PointKinetic
            Point kinetic object to be used in the generation of the signal from point kinetic equations.
        controlNames: Optional[list[str]]
            Names associated to each control quantity. Must be the same lenght as `controlVectors`. If None, the quantity names will be `default_0`, `default_1` etc.
        frequency: float
            The frequency of the signal to be generated.
        _wPoints: Optional[np.ndarray[float]]
            Array containing the weight data of points used to generate the signal. If None, the weights are equal to one.
        _interp: interpFunc
            Interpolation function to interpolate point quantities to array quantities with the same length as t.
        dataDict: dict
            Dictionnary where all informations related to transformation functions are dumped.
        **kwargs
            Other keyword arguments transmitted to `scipy.integral.solve_ivp`. See https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html 
        """
        assert len(tPoint) == len(rPoint)        
        tPoint = np.array(tPoint)
        rPoint = np.array(rPoint)
        if controlNames is not None:
            assert len(cPoints) == len(controlNames)
        if _wPoints is None:
            _wPoints = np.ones_like(cPoints)
        else: 
            assert len(cPoints) == len(_wPoints)
        # building timeVector & signalVector
        tmin = np.min(tPoint)
        tmax = np.max(tPoint)
        N = int(frequency * (tmax - tmin))
        timeVector = np.linspace(tmin,tmax,N+1)
        if useInvertedReactimeter:
            t = np.copy(timeVector)
            reactivityVector = np.interp(t, tPoint, rPoint)
            n = PK.inverted_population_to_reactivity(t, reactivityVector)
        else:
            t, n, _ = PK.reactivity_to_population(tPoint,rPoint,None,timeVector,**kwargs)
        signalVector = n
        # Building controlVectors & control weights
        controlVectors = []
        _controlWeights = []
        for i in range(0,len(cPoints)):
            controlVectors.append(_interp(t,tPoint,cPoints[i]))
            _controlWeights.append(_interp(t,tPoint,_wPoints[i]))
        super().__init__(timeVector, signalVector, controlVectors, controlNames)
        self.tPoint = np.array(tPoint)
        self.rPoint = np.array(rPoint)
        self.cPoints = np.array(cPoints)
        self.dataDict = {}
        self.PK = PK
        self._controlWeights = np.array(_controlWeights)
        self._interp = _interp

    @property
    def r(self) -> np.ndarray:
        """Shortcut to get the reactivity array"""
        return self._interp(self.t,self.tPoint,self.rPoint)

    def get_cPoint(self) -> np.ndarray:
        """Function that recovers the value of control points associated with `self.tPoint` values.

        Return
        ------
        numpy.ndarray:
            Control points.
        """
        # c = self.get_controlVector()
        # return self._interp(self.tPoint,self.t,c)
        c = self.get_controlVector()
        points = self._interp(self.tPoint,self.t,c)
        cPoint = []
        for element in points:
            if element not in cPoint:
                cPoint.append(element)
        return np.sort(cPoint)
    
    def get_reactivity(self) ->  np.ndarray:
        """A function that apply the reactimeter algorithm to the signal to get the time-dependant reactivity.
        
        Return
        ------
        numpy.ndarray:
            The reactivity obtained by applying the reactimeter algorithm.
        """
        return self.PK.population_to_reactivity(self.t, self.n)

    def get_reactivity_difference(self) -> np.ndarray:
        """A function that apply the reactimeter algorithm to the signal and substract the reactivity with the reference reactivity.
        
        Return
        ------
        numpy.ndarray:
            The difference between the reactivity obtained by applying the reactimeter algorithm and the reference reactivity.
        """
        return self.get_reactivity() - self.r
    
    def factors(self, fPoint, _interp: interpFunc = np.interp) -> np.ndarray:
        """Function that transform the signal by applying correction factors. The 'fPoints' applied must be assciated to the `self.tPoint` values.

        Parameters
        ----------
        fPoint: numpy.ndarray[float]
            Array containing the correction factor data of points used to transform the signal.
        _interp: interpFunc
            Interpolation function to interpolate point quantities to array quantities with the same length as t.

        Return
        ------
        numpy.ndarray:
            the correction factor vector. Each value is associated to a `self.t` value.
        """
        f = _interp(self.t,self.tPoint,fPoint)
        self.dataDict['f'] = f
        self.dataDict['fPoint'] = fPoint
        self.n = self.n * f
        return f

    def gaussian_noise(self, noiseAmplitude):
        """Function that apply a gaussian noise to the signal.

        Parameters
        ----------
        noiseAmplitude: float
            Amplitude of the noise, i.e. the scale of the random normal distribution generated.

        Return
        ------
        numpy.ndarray:
            the noise vector applied.
        """
        nDot = np.shape(self.n)[0]
        noiseArray = np.random.normal(1., noiseAmplitude, (nDot,))
        self.dataDict['noiseArray'] = noiseArray
        self.dataDict['noiseAmplitude'] = noiseAmplitude
        self.n = self.n * noiseArray
        return noiseArray
    
    @classmethod
    def from_parameters(cls, offsetReactivity:float, reactivityStep:float, reactivitySlope:float, nCycle:int, cycleDuration:float, rampDuration:float, factorStep:float = 0., factorSlope:float = 0., PK:PointKinetic = _stdPK, 
                        frequency:float = 100., **kwargs) -> 'Numerical':
        """Function that creates a numerical signal from parameters defining the reactivity and the correction factor applied. 
        The reactivity combines step-reactivity insertion (analog to rods behavior) with constant insertions (analog to temperature feedback behavior).

        Parameters
        ----------
        offsetReactivity: float
            Reactivity offset applied at every point but the first.
        reactivityStep:float
            Increment value of reactivity factor for each step. Represent the action of a rod. 
        reactivitySlope:float
            Reactivity inserted each second. Represent the action of temperature variation over time.
        nCycle:int
            Number of reactivity step during the transient.
        cycleDuration:float
            Time separating two reactivity step.
        rampDuration:float
            Time between the begining and the end of the step reactivity insertion. Represent the time needed to move the rod.
        factorStep:float
            Increment value of correction factor for each step of reactivity.
        factorSlope:float
            Correction factor inserted each second. Represent the action of temperature variation over time.
        PK: PointKinetic
            Point kinetic object to be used in the generation of the signal from point kinetic equations.
        frequency: float
            Frequency of the signal to be generated.
        _interp: interpFunc
            Interpolation function to interpolate point quantities to array quantities with the same length as t.
        """
        tPoint = [0.]
        rPoint = [0.]
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
        myNumeric = cls(tPoint, rPoint, cPoints, PK, ['step', 'slope'], frequency, **kwargs)
        myNumeric.factors(fPoint)
        return myNumeric

