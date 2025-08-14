import numpy as np
import scipy.optimize
import scipy.optimize._optimize

from spartalib_src.graphic import Graphic, plt
from spartalib_src.interactive import Interactive
from spartalib_src.pointkinetic import PointKinetic
from spartalib_src.detector import Detector
from spartalib_src.numerical import Numerical
from spartalib_src._least_squares import least_squares
from spartalib_src.spartalib_typing import *


class SPARTA:
    """A class containing the solvers and inputs used by the linear version of the Signal Processing Algorithm for Reactivity and Transient Analysis (SPARTA).

    Attributes
    ----------
    name: str
        Name of the object. Default is `default_name`.
    t: numpy.ndarray[float]
        The time Vector.
    n: numpy.ndarray[float]
        The signal Vector. Values are asccoiated to `t` values.
    configList: list[np.ndarray[float]]
        The configuration vector. Each element of the list is a time dependant component of the configuration (temperature, rod position etc).
    fittingOrders: list[int]
        The order of the polynomial interpolation used for each component of the configuration. Must be 1 or higher. Each increase in order add 2 degrees of freedom to the problem.
    configNames: list[str]
        The name of each component of the configuration.
    PK: PointKinetic
        Point kinetic object to be used in the generation of the signal from point kinetic equations.
    _ntStep: int
        Number of time steps.
    _nDim: int
        Number of point used for interpolations based on the control vector.
    configPoints: list[np.ndarray[float]]
        The configuration values of the points used for polynomial interpolation.
        
    Methods
    -------
    init_configPoints() -> list[np.ndarray[float]]:
        Returns an initial set of value for each configuration component in order to build the polynomial inetrpolation functions.
    init_xParams() -> np.ndarray[float]:
        Returns an initial set of input parameters for the solver.
    get_rParams(iConfig, xParams) -> np.ndarray[float]:
        Extract the reactivity parameters of a specific configuration component from the input set of parameters.
    get_fParams(iConfig, xParams) -> np.ndarray[float]:
        Extract the factor parameters of a specific configuration component from the input set of parameters.
    get_reactivityPoints(xParams) -> list[np.ndarray[float]]:
        Extract the reactivity parameters from the input set of parameters.
    get_factorPoints(xParams) -> list[np.ndarray[float]]:
        Extract the factor parameters from the input set of parameters.
    create_polynomial(iConfig, fp) -> Callable:
        Create a polynomial interpolation function for reactivity corresponding to the input parameters.
    create_rPolynomial(iConfig, xParams) -> Callable:
        Create a polynomial interpolation function for reactivity corresponding to the input parameters.
    create_fPolynomial(iConfig, xParams) -> Callable:
        Create a polynomial interpolation function for factors corresponding to the input parameters.
    main_loop(ftol, display, interactive, saveToPng) -> tuple[list[np.ndarray[float]]]:
        Run this linear auto corrected reactimeter algorithm using the least_square method.
    display_step(cost, xParams) -> None:
        Display the step values in the terminal.
    interactive_plot(iPlot, xParams, saveToPng, filepath) -> None:
        Updates and display each quantity of interest in an interactive plot.
    """

    def __init__(self, t:np.ndarray[float] ,n:np.ndarray[float], configList:list[np.ndarray[float]], configNames:list[str], fittingOrders:list[float], PK: PointKinetic = PointKinetic(), name: str = 'default_name') -> None:
        """Initialisation function of the SPARTA class.
        
        Parameters
        ----------
        t: numpy.ndarray[float]
            The time Vector.
        n: numpy.ndarray[float]
            The signal Vector. Values are asccoiated to `t` values.
        configList: list[np.ndarray[float]]
            The configuration vector. Each element of the list is a time dependant component of the configuration (temperature, rod position etc).
        configNames: list[str]
            The name of each component of the configuration.
        fittingOrders: list[int]
            The order of the polynomial interpolation used for each component of the configuration. Must be 1 or higher. Each increase in order add 2 degrees of freedom to the problem.
        PK: PointKinetic
            Point kinetic object to be used in the generation of the signal from point kinetic equations.
        name: str
            Name of the object. Default is `default_name`.
        """
        self.name = name
        self.t = np.array(t)
        self.n = np.array(n)
        self.configList = configList
        self.fittingOrders = fittingOrders
        self.configNames = configNames
        assert len(configNames) == len(configList)
        assert len(fittingOrders) == len(configList)
        self.PK = PK
        self._ntStep = int(np.shape(self.t)[0])
        self._nDim = len(configList)
        self.configPoints = self.init_configPoints()
        self.output = None

    def init_configPoints(self) -> list[np.ndarray[float]]:
        """A function that returns an initial set of value for each configuration component in order to build the polynomial interpolation functions."""
        def cost_function(x: float, poles: list[float]) -> float:
            result = np.zeros_like(x)
            for pole in poles:
                result += 1/(x-pole)**2
            return result
        # main
        configPoints = []
        for n in range(0, self._nDim):
            config = self.configList[n]
            order = self.fittingOrders[n]
            configMin, configMax = float(np.min(config)), float(np.max(config))
            poles = [config[0]]
            for i in range(0, order):
                newpole = scipy.optimize.minimize_scalar(cost_function, bounds = (configMin, configMax), args=(poles,)).x
                poles.append(newpole)
            #configPoints.append(np.linspace(configMin, configMax, order+1, endpoint=True))
            assert all(poles.count(x)==1 for x in poles)
            configPoints.append(np.array(poles))
        return configPoints

    def init_xParams(self, rbounds, fbounds, rnolist, fnolist) -> np.ndarray[float]:
        """A function that returns an initial set of input parameters for the solver."""
        params = []
        binf, bsup = [], []
        if rbounds is None:
            rinf, rsup = -np.inf, +np.inf
        else:
            rinf, rsup = rbounds
        if fbounds is None:
            finf, fsup = 0., +np.inf
        else:
            finf, fsup = fbounds
        for n in range(0, self._nDim):
            order = self.fittingOrders[n]
            params += [0.] * order + [1.] * order
            if n not in rnolist:
                binf += [rinf] * order
                bsup += [rsup] * order
            else:
                binf += [-1e-10] * order
                bsup += [+1e-10] * order
            if n not in fnolist:
                binf += [finf] * order
                bsup += [fsup] * order
            else:
                binf += [1-1e-8] * order
                bsup += [1+1e-8] * order
        return np.array(params), np.array(binf), np.array(bsup)

    def get_rParams(self, iConfig: int, xParams: np.ndarray) -> np.ndarray[float]:
        """A function that extract the reactivity parameters of a specific configuration component from the input set of parameters."""
        start=0
        for jConfig in range(0, iConfig):
            start += 2 * self.fittingOrders[jConfig]
        stop = start + self.fittingOrders[iConfig]
        return xParams[start:stop]

    def get_fParams(self, iConfig: int, xParams: np.ndarray) -> np.ndarray[float]:
        """A function that extract the factor parameters of a specific configuration component from the input set of parameters."""
        start=self.fittingOrders[0]
        for jConfig in range(0, iConfig):
            start += self.fittingOrders[jConfig]
            start += self.fittingOrders[jConfig+1]
        stop = start + self.fittingOrders[iConfig]
        return xParams[start:stop]

    def get_reactivityPoints(self, xParams: np.ndarray) -> list[np.ndarray[float]]:
        """A function that extract the reactivity parameters from the input set of parameters."""
        reactivityPoints = []
        for iConfig in range(0, self._nDim):
            cPoint = self.configPoints[iConfig]
            rPoint = np.zeros_like(cPoint)
            rPoint[1:] = self.get_rParams(iConfig, xParams)
            reactivityPoints.append(rPoint)
        return reactivityPoints

    def get_factorPoints(self, xParams: np.ndarray) -> list[np.ndarray[float]]:
        """A function that extract the factor parameters from the input set of parameters."""
        factorPoints = []
        for iConfig in range(0, self._nDim):
            cPoint = self.configPoints[iConfig]
            fPoint = np.ones_like(cPoint)
            fPoint[1:] = self.get_fParams(iConfig, xParams)
            factorPoints.append(fPoint)
        return factorPoints
    
    def get_xParams(self, rPoints: list[np.ndarray], fPoints: list[np.ndarray]):
        """A function that returns a set of input parameters for the solver."""
        params = []
        assert len(rPoints) == len(fPoints)
        for i in range(0, len(rPoints)):
            rPoint, fPoint = rPoints[i], fPoints[i]
            params += list(rPoint[1:]) + list(fPoint[1:])
        return np.array(params)

    def create_polynomial(self, iConfig: int, fp) -> Callable:
        """A function that create a polynomial interpolation function for reactivity corresponding to the input parameters."""
        xp = self.configPoints[iConfig]
        # Build problem: A.xp = fp
        stackList = []
        nPolyDim = np.shape(xp)[0]
        for k in range(0, nPolyDim):
            stackList.append(np.array(xp)**k)
        A = np.transpose(tuple(stackList))
        coefficients = np.dot(np.linalg.inv(A), fp)
        # define function
        def function(x: float|np.ndarray[float]) -> float|np.ndarray[float]:
            result = np.zeros_like(x, dtype=np.float64)
            for k in range(0, nPolyDim):
                result += coefficients[k] * np.array(x, dtype=np.float64)**k
            if np.shape(result)==():
                return np.float64(result)
            else:
                return result
        #return function
        return function

    def create_rPolynomial(self, iConfig: int, xParams: np.ndarray) -> Callable:
        """A function that create a polynomial interpolation function for reactivity corresponding to the input parameters."""
        fp = np.zeros_like(self.configPoints[iConfig])
        fp[1:] = self.get_rParams(iConfig, xParams)
        return self.create_polynomial(iConfig, fp)

    def create_fPolynomial(self, iConfig: int, xParams: np.ndarray) -> Callable:
        """A function that create a polynomial interpolation function for factors corresponding to the input parameters."""
        fp = np.ones_like(self.configPoints[iConfig])
        fp[1:] = self.get_fParams(iConfig, xParams)
        return self.create_polynomial(iConfig, fp)

    def fitting_function(self, xParams: np.ndarray) -> np.ndarray[float]:
        """A function that compute the local flux amplitude for the given reactivity and factor parameters given."""
        r = np.zeros(self._ntStep)
        f = np.ones(self._ntStep)
        for iConfig in range(0, self._nDim):
            config = self.configList[iConfig]
            rPolynomial = self.create_rPolynomial(iConfig, xParams)
            rLinear = rPolynomial(config)
            r += rLinear
            fPolynomial = self.create_fPolynomial(iConfig, xParams)
            fLinear = fPolynomial(config)
            # f = f * fLinear
            f = f + fLinear - 1.
        nf = self.PK.inverted_population_to_reactivity(self.t, r) * f
        return nf, r, f

    def fitting_components(self, xParams: np.ndarray) -> np.ndarray[float]:
        """A function that compute the local flux amplitude for the given reactivity and factor parameters given."""
        rs = []
        fs = []
        for iConfig in range(0, self._nDim):
            config = self.configList[iConfig]
            rPolynomial = self.create_rPolynomial(iConfig, xParams)
            rLinear = rPolynomial(config)
            rs.append(rLinear)
            fPolynomial = self.create_fPolynomial(iConfig, xParams)
            fLinear = fPolynomial(config)
            fs.append(fLinear)
        return rs, fs

    def main_loop(self, ftol=1.e-14, display = False, interactive = False, saveToPng: bool = False, xParams = None, rbounds = None, fbounds = None, rnolist = [], fnolist = [], weightArray = None, **kwargs) -> tuple[list[np.ndarray[float]]]:
        """A function that run this linear auto corrected reactimeter algorithm using the least_square method.

        Parameters
        ----------
        ftol: float
            ftol parameter used by the least_square function. see : https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html#scipy.optimize.least_squares 
        display: bool
            If `True`, displays computational steps in the Terminal. Advised for local runs as they can last several minutes.
        interactive: bool
            If `True`, display an interactive Figure using the `TkAgg` backend of `matplotlib`. Slows down the algorithm. Use it for debugging or simple visualisation of steps.
        saveToPng: bool
            If `True`, save all figures generated with the interactive plot. `interactive` must also be set to `True`.

        Returns
        -------
        np.ndarray
            final reactivity points.
        np.ndarray
            final factor points.
        """
        global loopStep
        def FUN(xParams: np.ndarray, weightArray):
            nf = self.fitting_function(xParams)[0]
            return (nf - self.n)*weightArray
        def CALL(intermediate_result: scipy.optimize._optimize.OptimizeResult):
            global loopStep
            xParams, cost = intermediate_result.x, intermediate_result.cost
            if display:
                self.display_step(cost, xParams)
            if interactive:
                filepath = '%s_loop-%02d.png'%(self.name, loopStep)
                self.interactive_plot(iPlot, xParams, saveToPng, filepath)
            loopStep += 1
        # mainloop
        loopStep = 0
        if interactive:
            iPlot = Interactive()
        if weightArray is None:
            weightArray = np.ones_like(self.t)
        xParams0, binf, bsup = self.init_xParams(rbounds, fbounds, rnolist, fnolist)
        if xParams is None:
            xParams = xParams0
        self.output = least_squares(FUN, xParams, args= (weightArray,), jac = "2-point", ftol=ftol, bounds = (binf, bsup), callback = CALL, **kwargs)
        result = self.output.x
        reactivityPoints = self.get_reactivityPoints(result)
        factorPoints = self.get_factorPoints(result)
        return reactivityPoints, factorPoints
    
    def display_step(self, cost: float, xParams: np.ndarray[float]) -> None:
        """A function to display the step values in the terminal.
        
        Parameters
        ----------
        cost: float
            The step least square evaluation.
        xParams: numpy.ndarray[float]
            The parameters used for the step.
        """
        print(f' ----- STEP: {loopStep}')
        print(f'cost: {cost:.4E}')
        for iConfig in range(0, self._nDim):
            print(f'  -- CV: {iConfig+1}/{len(self.configList)}')
            cPoint = self.configPoints[iConfig]
            indexes = np.argsort(cPoint)
            rPoint = np.zeros_like(self.configPoints[iConfig])
            rPoint[1:] = self.get_rParams(iConfig, xParams)
            fPoint = np.ones_like(self.configPoints[iConfig])
            fPoint[1:] = self.get_fParams(iConfig, xParams)
            [print('cPoint: ',end=' ')]+[print('%+.2e'%(x),end=' ') for x in cPoint[indexes]]
            print()
            [print('rPoint: ',end=' ')]+[print('%+.3f'%(1e5*x),end=' ') for x in rPoint[indexes]]
            print()
            [print('fPoint: ',end=' ')]+[print('%+.3f'%(x),end=' ') for x in fPoint[indexes]]
            print()

    def interactive_plot(self, iPlot: Interactive, xParams:np.ndarray[float], saveToPng: bool = True, filepath: Optional[str] = None) -> None:
        """A function that updates and display each quantity of interest in an interactive plot.

        Parameters
        ----------
        iPlot: Interactive
            The interactive object in which quantities must be plotted.
        xParams: numpy.ndarray[float]
            The parameters used to build the displayed plots.
        saveToPng: bool
            Must be set to `True` to save the figure after plotting.
        filepath: Optional[str]
            Location where the file must be saved.
        """
        reference = np.zeros(self._ntStep)
        factors = np.ones(self._ntStep)
        for i in range(0, self._nDim):
            config = self.configList[i]
            rPolynomial = self.create_rPolynomial(i, xParams)
            rLinear = rPolynomial(config)
            reference += rLinear
            fPolynomial = self.create_fPolynomial(i, xParams)
            fLinear = fPolynomial(config)
            # factors = factors * fLinear
            factors = factors + fLinear - 1.
        nf = self.fitting_function(xParams)[0]
        reactivityPoints = self.get_reactivityPoints(xParams)
        factorPoints = self.get_factorPoints(xParams)
        nmin, nmax = min(np.min(self.n),np.min(nf)), max(np.max(self.n),np.max(nf))
        rmin, rmax = min([1e5*np.min(x) for x in [reference] + reactivityPoints]), max([1e5*np.max(x) for x in [reference] + reactivityPoints])
        fmin, fmax = min([np.min(x) for x in [factors] + factorPoints]), max([np.max(x) for x in [factors] + factorPoints])
        iPlot.time_lim(self.t[0],self.t[-1])
        iPlot.signal_lim(nmin, nmax)
        iPlot.reactivity_lim(rmin, rmax)
        iPlot.factors_lim(fmin, fmax)
        iPlot.clear()
        iPlot.interactive_plot(iPlot.axN, self.t, self.n, color=0, legend='Signal')
        iPlot.interactive_plot(iPlot.axN, self.t, nf, color=1, legend='Fit')
        for i in range(0, self._nDim):
            config = self.configList[i]
            normedC = (config-np.min(config))/(np.max(config)-np.min(config))
            rPolynomial = self.create_rPolynomial(i, xParams)
            rLinear = rPolynomial(config)
            fPolynomial = self.create_fPolynomial(i, xParams)
            fLinear = fPolynomial(config)
            iPlot.interactive_plot(iPlot.axC, self.t, normedC, color=i, legend=str(self.configNames[i]))
            iPlot.interactive_plot(iPlot.axRt, self.t, 1.e5*rLinear, color=i, legend=str(self.configNames[i]))
            iPlot.interactive_plot(iPlot.axRc, normedC, 1.e5*rLinear, color=i, legend=str(self.configNames[i]))
            iPlot.interactive_plot(iPlot.axFt, self.t, fLinear, color=i, legend=str(self.configNames[i]))
            iPlot.interactive_plot(iPlot.axFc, normedC, fLinear, color=i, legend=str(self.configNames[i]))
        iPlot.legend()
        iPlot.display()
        if saveToPng:
            assert filepath is not None
            iPlot.save(filepath)