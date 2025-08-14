import numpy as np
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import warnings
try:
    matplotlib.use('TkAgg')
except:
    warnings.warn("failed to import 'TkAgg' as matplotlib backend. Falling back to default backend. Interactive display may not work.",ImportWarning)

class Interactive:
    """A class to create interactive Graphics when using the ACRA.

    Attributes
    ----------
    fig: matplotlib.figure.Figure
        Figure containing the ACRA step plots.
    axN: matplotlib.figure.Axes
        Axes corresponding to signal plots.
    axC: matplotlib.figure.Axes
        Axes corresponding to control vector plot as a function of time.
    axRt: matplotlib.figure.Axes
        Axes corresponding to reactivity plot as a function of time.
    axFt: matplotlib.figure.Axes
        Axes corresponding to factor plot as a function of time.
    axRc: matplotlib.figure.Axes
        Axes corresponding to reactivity plot as a function of control vector.
    axFc: matplotlib.figure.Axes
        Axes corresponding to factor plot as a function of control vector.

    Methods
    -------
    time_lim(tmin, tmax) -> None:
        Set the time limits on the interactive plot.
    signal_lim(nmin, nmax) -> None:
        Set the signal limits on the interactive plot.
    reactivity_lim(rmin, rmax) -> None:
        Set the reactivity limits on the interactive plot.
    factors_lim(fmin, fmax) -> None:
        Set the factor limits on the interactive plot.
    legend():
        Display the legend of lines drawn on the graphic.
    clear():
        Clears lines from an Axes object.
    interactive_plot(ax, x, y, legend, lw, marker, markersize, ls, color) -> None:
        Draw a line on an interactive plot.
    display():
        Calls functions of matplotlib to display interactive plot.
    save(filepath) -> None:
        Export the interactive graphic to a given location and in a given format.
    """

    COLORS = ['#07519E','#CF1E4C','#8DC340','#F68C20','#2BA8E0']

    def __init__(self, figsize: list[float] = [16.,9.], dpi: int = 100) -> None:
        """Initialisation function of the Interactive class.
        
        Parameters
        ----------
        figsize: list[float]
            Dimension of the figure in inches. Default is [16.,9.].
        dpi: int
            The number of pixels per inches of the figure. Default is 100.
        """
        def _ax(wparams, xmin, xmax, ymin, ymax, xlabel, ylabel, xscale, yscale) -> plt.Axes:
            """Internal function to build Axes object."""
            ax = self.fig.add_axes(wparams)
            if xlabel != '':
                ax.set_xlabel(xlabel)
            if ylabel != '':
                ax.set_ylabel(ylabel)
            ax.set_xscale(xscale)
            ax.set_yscale(yscale)
            ax.set_xlim(xmin,xmax)
            ax.set_ylim(ymin,ymax)
            ax.grid(which='major', lw=0.5, ls='--', color='#121212')
            ax.spines['top'].set_linewidth(0.5)
            ax.spines['right'].set_linewidth(0.5)
            ax.spines['bottom'].set_linewidth(0.5)
            ax.spines['left'].set_linewidth(0.5)
            ax.xaxis.set_tick_params(length=8.,width=0.5)
            ax.yaxis.set_tick_params(length=8.,width=0.5)
            return ax
        numfig = len(plt.get_fignums())
        self.fig = plt.figure(numfig, figsize, dpi)
        self.axN = _ax([0.05,0.6,0.25,0.38], 0., 1., 0., 1., 'Time [s]', 'Detector Signal', 'linear', 'linear')
        self.axC = _ax([0.05,0.1,0.25,0.38], 0., 1., 0., 1., 'Time [s]', 'Control Vector', 'linear', 'linear')
        self.axRt = _ax([0.38,0.6,0.25,0.38], 0., 1., -100., +100., 'Time [s]', 'Reactivity [pcm]', 'linear', 'linear')
        self.axFt = _ax([0.38,0.1,0.25,0.38], 0., 1., 0.8, 1.2, 'Time [s]', 'Correction factor', 'linear', 'linear')
        self.axRc = _ax([0.73,0.6,0.25,0.38], 0., 1., -100., +100., 'Control Vector', 'Reactivity', 'linear', 'linear')
        self.axFc = _ax([0.73,0.1,0.25,0.38], 0., 1., 0.8, 1.2, 'Control Vector', 'Correction factor', 'linear', 'linear')
        plt.ion()
        plt.show()

    def time_lim(self, tmin: float, tmax: float) -> None:
        """A function that set the time limits on the interactive plot.
        
        Parameters
        ----------
        tmin
            Minimum of time displayed.
        tmax
            Maximun of time displayed.
        """
        self.axN.set_xlim(tmin,tmax)
        self.axC.set_xlim(tmin,tmax)
        self.axRt.set_xlim(tmin,tmax)
        self.axFt.set_xlim(tmin,tmax)

    def signal_lim(self, nmin: float, nmax: float) -> None:
        """A function that set the signal limits on the interactive plot.
        
        Parameters
        ----------
        nmin
            Minimum of signal displayed.
        nmax
            Maximun of signal displayed.
        """
        self.axN.set_ylim(nmin, nmax)
        yticks = self.axN.get_yticks()
        nmin = 2*yticks[0] - yticks[1]
        nmax = 2*yticks[-1] - yticks[-2]
        self.axN.set_ylim(nmin, nmax)

    def reactivity_lim(self, rmin: float, rmax: float) -> None:
        """A function that set the reactivity limits on the interactive plot.
        
        Parameters
        ----------
        rmin
            Minimum of reactivity displayed.
        rmax
            Maximun of reactivity displayed.
        """
        self.axRt.set_ylim(rmin,rmax)
        yticks = self.axRt.get_yticks()
        rmin = 2*yticks[0] - yticks[1]
        rmax = 2*yticks[-1] - yticks[-2]
        self.axRt.set_ylim(rmin,rmax)
        self.axRc.set_ylim(rmin,rmax)

    def factors_lim(self, fmin: float, fmax: float) -> None:
        """A function that set the factor limits on the interactive plot.
        
        Parameters
        ----------
        fmin
            Minimum of factor displayed.
        fmax
            Maximun of factor displayed.
        """
        self.axFt.set_ylim(fmin,fmax)
        yticks = self.axFt.get_yticks()
        fmin = 2*yticks[0] - yticks[1]
        fmax = 2*yticks[-1] - yticks[-2]
        self.axFt.set_ylim(fmin,fmax)
        self.axFc.set_ylim(fmin,fmax)

    def legend(self):
        """A function that display the legend of lines drawn on the graphic."""
        if len(self.axN.get_lines()) > 0:
            self.axN.legend()
        if len(self.axC.get_lines()) > 0:
            self.axC.legend()
        if len(self.axRt.get_lines()) > 0:
            self.axRt.legend()
        if len(self.axFt.get_lines()) > 0:
            self.axFt.legend()
        if len(self.axRc.get_lines()) > 0:
            self.axRc.legend()
        if len(self.axFc.get_lines()) > 0:
            self.axFc.legend()

    def clear(self):
        """A function that clears lines from an Axes object."""
        for ax in [self.axN , self.axC , self.axRt, self.axFt, self.axRc, self.axFc]:
            for line in ax.get_lines():
                line.remove()

    def interactive_plot(self, ax: plt.Axes, x: np.ndarray, y: np.ndarray, legend: str = '', lw: float = 1.0, marker: str = '', markersize: float = 5., ls: str = '-', color: int | str = 0) -> None:
        """A function that draw a line on an interactive plot.

        Parameters
        ----------
        x: np.ndarray
            x-coordinate of points used to draw the line.
        y: np.ndarray
            y-coordinate of points used to draw the line.
        legend: str
            Legend of the line. Default is no legend.
        lw: float
            Width of the line. Default is 1.
        marker: str
            Marker used in line points. Default is no marker.
        markersize: float
            Size of the marker if it exists. Default is 5.
        ls: str
            Style of the line. Default is '-' (plain).
        color: int | str
            The color of the line. If int, take the color at the int position from the COLORS list instance. If str, treat as an hexadecimal color value.

        """
        if type(color) == int:
            color = Interactive.COLORS[color%len(Interactive.COLORS)]
        line = ax.plot(x, y, lw=lw, ls=ls, marker=marker, markersize=markersize,  color=color)[0]
        if legend!='':
            line.set_label(legend)

    def display(self):
        """A function that calls functions of matplotlib to display interactive plot."""
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def save(self, filepath: str) -> None:
        """A function that export the interactive graphic to a given location and in a given format.
        
        Parameters
        ----------
        filepath: str
            Location where the file must be saved.
        """
        self.fig.savefig(filepath)