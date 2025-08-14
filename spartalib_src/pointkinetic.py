import numpy as np
import scipy.integrate
from acralib_src.acralib_typing import *

class PointKinetic:
    """Class to perform point-kinetic computations.

    Instances
    ---------
    L: float
        Default effective mean lifetime of prompt neutron.
    B: float
        Default effective delayed fraction of delayed neutron.
    ai: ArrayLike
        Default abondances of each precursor families.
    li: ArrayLike
        Default decay constants of each precursor families.

    Attributes
    ----------
    L: float
        Effective mean lifetime of prompt neutron.
    B: float
        Effective delayed fraction of delayed neutron.
    ai: ArrayLike
        Abondances of each precursor families.
    li: ArrayLike
        Decay constants of each precursor families.

    Methods
    -------
    reactivity_to_population(timeVector, reactivityVector, sourceVector, t_eval, method, rtol, atol, **kwargs) -> tuple[np.ndarray]
        Applies point-kinetic equations to get the flux amplitude. Based on scipy.integral.solve_ivp function
    population_to_reactivity(timeVector, amplitudeVector) -> np.ndarray
        Applies inverted point-kinetic equations to get the flux amplitude. Based on [Insert future ANS Publication] work.
    inverted_population_to_reactivity(self, timeVector, reactivityVector) -> np.ndarray:
        Applies point-kinetic equations the reactivity vector. Is actually the inverse function of population_to_reactivity.
    reactivities_to_factor(timeVector, reactivity, reference) -> np.ndarray:
        Applies point-kinetic equations to get the flux amplitude associated to the Reactivity and to the reference. Ratio of flux amplitudes is output.


    """
    L = float(25.e-6)
    B = float(679e-5)
    ai = np.array([0.030,0.172,0.103,0.163,0.364,0.018,0.128,0.022])
    li = np.array([0.01246670,0.02829170,0.04252440,0.1330420,0.2924672,0.6664877,1.634781,3.554600])

    def __init__(self, L:float = L, B:float = B, ai:np.ndarray[float] = ai, li:np.ndarray[float] = li) -> None:
        """Initialisation method of the PointKinetic class.

        Parameters
        ----------
        L: float
            Effective mean lifetime of prompt neutron. Default is `25.e-6`.
        B: float
            Effective delayed fraction of delayed neutron. Default is `679e.e-5`.
        ai: numpy.ndarray[float]
            Abondances of each precursor families. Default is `[0.030,0.172,0.103,0.163,0.364,0.018,0.128,0.022]`.
        li: numpy.ndarray[float]
            Decay constants of each precursor families. Default is `[0.01246670,0.02829170,0.04252440,0.1330420,0.2924672,0.6664877,1.634781,3.554600]`.
        """  
        self.L = float(L)
        self.B = float(B)
        self.ai = np.array(ai)
        self.li = np.array(li)
        assert np.shape(ai)[0] == np.shape(li)[0]

    def reactivity_to_population(self, timeVector:np.ndarray[float], reactivityVector:np.ndarray[float], sourceVector:Optional[np.ndarray[float]] = None, t_eval:Optional[np.ndarray[float]] = None, 
                                 method:Literal['RK45','RK23','DOP853','Radau','BDF','LSODA']='Radau', rtol:float = 1.e-3, atol:float = 1.e-6, **kwargs) -> tuple[np.ndarray]:
        """Function that applies point-kinetic equations to get the flux amplitude. Based on scipy.integral.solve_ivp function.

        Parameters
        ----------
        timeVector:numpy.ndarray[float]
            Array containing the time values associated to the reactivity values given in `reactivityVector`.
        reactivityVector:numpy.ndarray[float]
            Array containing the reactivity values to be processed.
        sourceVector:Optional[numpy.ndarray[float]]
            Array containing the values of the external sources for each time value. If `None`, the external sources are neglected.
        t_eval:Optional[numpy.ndarray[float]]
            Array containing the wanted output time values. If None, output timesteps will be chosen automatically by `scipy.integral.solve_ivp`.
        method:Literal['RK45','RK23','DOP853','Radau','BDF','LSODA']
            Method used by `scipy.integral.solve_ivp` to solve the point kinetic equations. Default value is `Radau`.
        rtol:float
            Relative tolerance of the method used by `scipy.integral.solve_ivp`. Default value is `1.e-3`.
        atol:float
            Absolute tolerance of the method used by `scipy.integral.solve_ivp`. Default value is `1.e-6`.
        **kwargs
            Other keyword arguments transmitted to `scipy.integral.solve_ivp`. See https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html 

        Returns
        -------
        numpy.ndarray[float]
            Time array.
        numpy.ndarray[float]
            Flux amplitude array.
        numpy.ndarray[float]
            Precursor array.

        """
        # Initialize local variables
        L = self.L
        B = self.B
        ai = self.ai
        li = self.li
        nGroup = np.shape(ai)[0]
        # Local functions
        def _jacobian(t, y):
            rho = np.interp(t, timeVector, reactivityVector)
            M = np.zeros((nGroup + 1, nGroup + 1))
            M[0,0] = (rho-B)/L
            for i in range(0,nGroup):
                M[i+1,0] = B*ai[i]/L
                M[0,i+1] = +li[i]
                M[i+1,i+1] = -li[i]
            return M
        def _equations(t, y):
            source = np.interp(t, timeVector, sourceVector)
            M = _jacobian(t, y)
            My = np.matmul(M, y)
            My[0] = My[0] + source
            return My
        # Format input variables
        timeVector = np.array(timeVector)
        reactivityVector = np.array(reactivityVector)
        if sourceVector is None:
            sourceVector = np.zeros_like(timeVector)
        else:
            sourceVector = np.array(sourceVector)
        tStart = np.min(timeVector)
        tEnd = np.max(timeVector)
        n0 = 1.
        c0 = n0 * self.B*self.ai/(self.li*self.L)
        y0 = np.insert(c0,0,n0)
        # scipy call
        if method.lower() in ['radau','bdf','lsoda']:
            sol = scipy.integrate.solve_ivp(_equations,(tStart,tEnd),y0,method=method,t_eval=t_eval,jac=_jacobian,rtol=rtol,atol=atol,**kwargs)
        else:
            sol = scipy.integrate.solve_ivp(_equations,(tStart,tEnd),y0,method=method,t_eval=t_eval,rtol=rtol,atol=atol,**kwargs)
        # Return results
        return sol.t, sol.y[0,:], sol.y[1:,:]

    def population_to_reactivity(self, timeVector:np.ndarray[float], amplitudeVector:np.ndarray[float]) -> np.ndarray:
        """Function that applies inverted point-kinetic equations to get the flux amplitude. Based on [Insert future ANS Publication] work.

        Parameters
        ----------
        timeVector: numpy.ndarray[float]
            Array containing the time values associated to the flux amplitude values given in `reactivityVector`.
        amplitudeVector: numpy.ndarray[float]
            Array containing the flux amplitude values to be processed.

        Returns
        -------
        numpy.ndarray[float]
            Reactivity array. The associated time vector is the input `timeVector`.
        """
        # Initialize local variables
        L = self.L
        B = self.B
        ai = self.ai
        li = self.li
        # Format input variables
        timeVector = np.array(timeVector)
        amplitudeVector= np.array(amplitudeVector)
        nDot = np.shape(amplitudeVector)[0]
        Reactivity = np.zeros(nDot)
        ci = B*ai/(li*L)
        # Loop
        for k in range(1,nDot):
            dt = timeVector[k] - timeVector[k-1]
            dn = amplitudeVector[k] - amplitudeVector[k-1]
            dci = (1-np.exp(-li*dt)) * (amplitudeVector[k] * B*ai/(li*L) - ci)
            dc = np.sum(dci)
            Reactivity[k] = L * (dn + dc)/(dt * amplitudeVector[k])
            ci = ci + dci
        # Return results
        return Reactivity

    def inverted_population_to_reactivity(self, timeVector:np.ndarray[float], reactivityVector:np.ndarray[float]) -> np.ndarray:
        """Function that applies point-kinetic equations the reactivity vector. Is actually the inverse function of population_to_reactivity. 

        Parameters
        ----------
        timeVector: numpy.ndarray[float]
            Array containing the time values.
        reactivityVector: numpy.ndarray[float]
            Array containing the reactivity to be processed.
            
        Returns
        -------
        numpy.ndarray[float]
            Flux amplitude array. The associated time vector is the input `timeVector`.
        """
        L = self.L
        B = self.B
        ai = self.ai
        li = self.li
        # Format input variables
        timeVector = np.array(timeVector)
        reactivityVector= np.array(reactivityVector)
        nDot = np.shape(reactivityVector)[0]
        amplitudeVector = np.ones(nDot)
        # ci = B*ai/(li*L)/(B/(B-reactivityVector[0]))
        ci = B*ai/(li*L)
        for k in range(1,nDot):
            dt = timeVector[k] - timeVector[k-1]
            u = (1-np.exp(-li*dt)) * B*ai/(li*L)
            v = (1-np.exp(-li*dt)) * ci
            amplitudeVector[k] = (amplitudeVector[k-1] + np.sum(v)) / (1 + np.sum(u) - reactivityVector[k]*dt/L)
            dci = u*amplitudeVector[k] - v
            ci = ci + dci
        return amplitudeVector
    
    def reactivities_to_factor(self, timeVector:np.ndarray[float], reactivity:np.ndarray[float], reference:np.ndarray[float]) -> np.ndarray:
        """Function that applies point-kinetic equations to get the flux amplitude associated to the Reactivity and to the reference. 
        Then, the factor to apply to get the reference reactivity is deduced from the ratio of the two flux amplitudes.

        Parameters
        ----------
        timeVector: numpy.ndarray[float]
            Array containing the time values.
        reactivity: numpy.ndarray[float]
            Array containing the reactivity to be processed.
        reference: numpy.ndarray[float]
            Array containing the reference reactivity to obtain once the output factor is applied.
            
        Returns
        -------
        numpy.ndarray[float]
            Correction factor array. The associated time vector is the input `timeVector`.
        """
        nf = self.inverted_population_to_reactivity(timeVector, reactivity)
        n = self.inverted_population_to_reactivity(timeVector, reference)
        return nf/n

