import numpy as np
import pandas as pds
from acralib_src.acralib_typing import *

class Detector:
    """[This Class is still under development aand major changes can still occur]
    Specific class to create custom signals compatible with the ACRA_V2 workflow. 
    The signal dataset `self.df` contains at least 3 columns: `time`, `signal` and all the control quantities that can be rod positions, fuel temperature...
    The default columns names for control quantities are `default_0`, `default_1` etc.

    Attributes
    ----------
    df: pandas.core.frame.DataFrame
        Data frame containing the data asscoiated to the signal.
    t: numpy.ndarray[float]
        Array containing the time values. Is recovered from `df`.
    n: numpy.ndarray[float]
        Array containing the signal values. Is recovered from `df`.
    _controlWeights: numpy.ndarray[float]
        Weight associated to each control quantity at each time step.

    """

    def __init__(self, timeVector:np.ndarray[float], signalVector:np.ndarray[float], controlVectors:list[np.ndarray[float]], controlNames:Optional[list[str]] = None) -> None:
        """Initialisation method of the Signal class.

        Parameters
        ----------
        timeVector: numpy.ndarray[float]
            Array of time bins associated to each value of signal and control quantities.
        signalVector: numpy.ndarray[float]
            Array of the signal values at each time point. Must be the same lenght as `timeVector`.
        controlVectors: list[numpy.ndarray[float]]
            Array list of control quantities evalutated at each time point. Each array must be the same lenght as `timeVector`.
        controlNames: Optional[list[str]]
            Names associated to each control quantity. Must be the same lenght as `controlVectors`. If None, the quantity names will be `default_0`, `default_1` etc.
        """  
        assert len(signalVector) == len(timeVector)
        dataDict = {
            'time': timeVector,
            'signal':signalVector,
        }
        nBins = len(timeVector)
        nParam = len(controlVectors)
        for i in range(0, nParam):
            assert len(controlVectors[i]) == len(timeVector)
            array = controlVectors[i]
            if controlNames is None:
                name = f'default_{i}'
            else:
                assert len(controlNames) == len(controlVectors)
                name = controlNames[i]
            dataDict[name] = array
        self.df = pds.DataFrame(data = dataDict)
        self._controlWeights = np.ones((nParam, nBins))
        
    @property
    def t(self) -> np.ndarray:
        """Shortcut to get time array"""
        return self.df['time'].to_numpy()
    
    @property
    def n(self) -> np.ndarray:
        """Shortcut to get the signal array"""
        return self.df['signal'].to_numpy()

    @n.setter
    def n(self, value: np.ndarray) -> None:
        """Clean setter to update the signal"""
        self.df['signal'] = value
    
    def get_controlVector(self) -> np.ndarray:
        """Function that build a control array with the same length that `self.t`. It attributes a scalar value at each time point which represents the state of the core.
        The control array is built based on `_controlWeights` which attributes a weight associated to each control quantity at each time step. 
        Each quantity is normalised between 0. and 1. before being processed.
            For example,I have two control quantities, a rod position ([0.,1.,5.]) and a temperature in Kelvin ([290.,300.,310.]).
            My _controlWeights are ([1.,1.,1.]) and ([0.01,0.1,0.2]). 
            The value of my control vector will be ([2.9,31.0,67.0])
            Once renormalised, The function will return get ([0.00,0.44,1.00])
        It is possible to update the `_controlWeights` values with `apply_characteristics` and `weight_from_phase_space`. 
        
        Returns
        -------
        numpy.ndarray:
            Control vector built.
        """
        nParam, nBins = np.shape(self._controlWeights)
        controlVector = np.zeros(nBins)
        for i in range(0, nParam):
            controlVector += self.df.iloc[:,i+2].to_numpy() * self._controlWeights[i,:]
        minValue = np.min(controlVector)
        maxValue = np.max(controlVector)
        return (controlVector - minValue)/(maxValue - minValue)

    def apply_characteristics(self, controlQuantityName:str, quantityVector:np.ndarray[float], controlVector:np.ndarray[float]):
        """Not developed yet. Will be in a future version"""
        pass

    def weight_from_phase_space(self, quantitiesVector:list[np.ndarray[float]], controlVector):
        """Not developed yet. Will be in a future version"""
        pass
