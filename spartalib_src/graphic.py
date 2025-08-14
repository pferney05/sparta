import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import warnings

from typing import Optional

class Graphic:
    """Basic class built on matplotlib to produce simple figures.

    Attributes
    ----------
    fig: matplotlib.figure.Figure
        Matplotlib `Figure` object the user can manipulate.
    ax: matplotlib.axes.Axes
        Matplotlib `Axes` object where lines will be drawn.

    Instances
    ---------
    COLORS: list[str]
        Hexadecimal values of INL used colors to stay as corporate as one can be. Order is 'Blue', 'Red', 'Green', 'Orange', 'Cyan'.

    Methods
    -------
    line_plot(x, y, legend, lw, marker, markersize, ls, color) -> None
        Draw a line going through points defined by `x` and `y` arrays.
    legend() -> None
        Display the legend of lines drawn on the graphic.
    save(filepath) -> None
        Export the graphic to a given location and in a given format.
    from_linelist(title, xLabel, yLabel, x, yList, xBounds, yBounds, legends) -> Graph
        Class method. Create a linear graphic automatically from a list of lines.
    show() -> None:
        Static method. Shortcut to `matplotlib.pyplot.show()` command.

    """

    COLORS = ['#07519E','#CF1E4C','#8DC340','#F68C20','#2BA8E0']

    def __init__(self, xmin: float = 0., xmax: float = 1., ymin: float = 0., ymax: float = 1., xlabel: str = '', ylabel: str = '', title: str = '', 
                 xscale: str = 'linear', yscale: str = 'linear', figsize: list[float] = [6.4,4.8], dpi: int = 150) -> None:
        """Initialisation method of the Graph class.

        Parameters
        ----------
        xmin: float
            Minimum value displayed on the x-axis.
        xmax: float
            Maximum value displayed on the x-axis.
        ymin: float
            Minimum value displayed on the y-axis.
        ymax: float
            Maximum value displayed on the y-axis.
        xlabel: str
            Label of the x-axis of the figure. Default is no label.
        ylabel: str
            Label of the y-axis of the figure. Default is no label.
        title: str
            Title that will appear on the top of the figure. Default is None and the figure will be displayed larger.
        xscale: str
            Option given in input to matplotlib `set_xscale` Axes method.
        yscale: str
            Option given in input to matplotlib `set_yscale` Axes method.
        figsize: list[float]
            Dimension of the figure in inches. Default is [6.4, 4.8].
        dpi: int
            The number of pixels per inches of the figure.
        """
        numfig = len(plt.get_fignums())
        self.fig = plt.figure(numfig, figsize, dpi)
        if title != '':
            self.ax: plt.Axes = self.fig.add_axes([0.14,0.14,0.80,0.80])
            self.ax.set_title(title)
        else:
            self.ax: plt.Axes = self.fig.add_axes([0.14,0.12,0.80,0.84])
        if xlabel != '':
            self.ax.set_xlabel(xlabel)
        if ylabel != '':
            self.ax.set_ylabel(ylabel)
        self.ax.set_xscale(xscale)
        self.ax.set_yscale(yscale)
        self.ax.set_xlim(xmin,xmax)
        self.ax.set_ylim(ymin,ymax)
        self.ax.grid(which='major', lw=0.5, ls='--', color='#121212')
        self.ax.spines['top'].set_linewidth(0.5)
        self.ax.spines['right'].set_linewidth(0.5)
        self.ax.spines['bottom'].set_linewidth(0.5)
        self.ax.spines['left'].set_linewidth(0.5)
        self.ax.xaxis.set_tick_params(length=8.,width=0.5)
        self.ax.yaxis.set_tick_params(length=8.,width=0.5)
    
    def line_plot(self, x: np.ndarray, y: np.ndarray, legend: str = '', lw: float = 1.0, marker: str = '', markersize: float = 5., ls: str = '-', color: int | str = 0) -> None:
        """A function that draw a line going through points defined by `x` and `y` arrays.

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
            color = Graphic.COLORS[color%len(Graphic.COLORS)]
        line = self.ax.plot(x, y, lw=lw, ls=ls, marker=marker, markersize=markersize,  color=color)[0]
        if legend!='':
            line.set_label(legend)

    def legend(self) -> None:
        """A function that display the legend of lines drawn on the graphic."""
        self.ax.legend()

    def save(self, filepath: str) -> None:
        """A function that export the graphic to a given location and in a given format.
        
        Parameters
        ----------
        filepath: str
            Location where the file must be saved.
        """
        self.fig.savefig(filepath)

    @classmethod
    def from_linelist(cls, title: str, xLabel: str, yLabel: str, x: np.ndarray, yList: list[np.ndarray], xBounds: Optional[list[float]] = None, yBounds: Optional[list[float]] = None, 
                      legends: list[str] = None, colors: Optional[list[str]] = None) -> 'Graphic':
        """A function that creates a linear graphic automatically from a list of lines.
        
        Parameters
        ----------
        title: str
            Title that will appear on the top of the figure.
        xlabel: str
            Label of the x-axis of the figure.
        ylabel: str
            Label of the y-axis of the figure.
        x: np.ndarray
            x-coordinate of points used to draw the line.
        yList: list[np.ndarray]
            List of y-coordinate of points used to draw the line.
        xBounds: list[float]
            List of xmin, xmax values provided to the figure builder. If None, is computed automatically.
        yBounds: list[float]
            List of ymin, ymax values provided to the figure builder. If None, is computed automatically.
        legends: list[str]
            Legends of lines. Default is no legends.

        Returns
        -------
        Graphic
            The graphic created.
        """
        if colors is not None:
            assert len(colors) == len(yList)
        if legends is not None:
            assert len(legends) == len(yList)
        if xBounds is None:
            xmin = np.min(x)
            xmax = np.max(x)
        else:
            xmin, xmax = xBounds
        if yBounds is None:
            yArr = np.array(yList)
            ymin = np.min(yArr)
            ymax = np.max(yArr)
        else:
            ymin, ymax = yBounds     
        myGraph = cls(xmin, xmax, ymin, ymax, xLabel, yLabel, title)
        for i, y in enumerate(yList):
            if colors is not None:
                color = colors[i]
            else:
                color = i
            if legends is not None:
                legend = legends[i]
            else:
                legend = ''
            myGraph.line_plot(x, y, legend, color=color)
        if legends is not None:
            myGraph.legend()
        return myGraph

    @staticmethod
    def show() -> None:
        """A function that call `matplotlib.pyplot.show()`. Show all figures."""
        plt.show()