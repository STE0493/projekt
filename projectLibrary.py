import numpy as np
import matplotlib.pyplot as plt
from typing import Union
import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, interact_manual
from numpy.typing import NDArray
import warnings

class Set:
    """
    A class representing a fractal set
    The plot axis are r and i - real(horizontal) and imaginary(vertical)

    Necessary packages:
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.cm
        from typing import Union
        import ipywidgets as widgets
        from ipywidgets import interact, interactive, fixed, interact_manual      
        from numpy.typing import NDArray
        import warnings

    Attributes:
        Cr(int or float): real part of complex number C (representing the center point of the plot)
        Ci(int or float): imaginary part of complex number C (representing the center point of the plot)
        zoom(int): desired zoom of final plot
        n(int): resolution of final image (n*n)
        k(int): number of iterations
        colormap(str): colormap used in plot
        Cr_min(int or float): real axis minimum value
        Cr_max(int or float): real axis maximum value
        Ci_min(int or float): imaginary axis minimum value
        Ci_max(int or float): imaginary axis maximum value
        C(complex): array of all complex numbers in given bounds 
        matrix(np.ndarray): final (modified) divergence matrix

        Attributes for creating an interactive plot using widgets:
        self.colormap_text
        self.Cr_slider
        self.Ci_slider
        self.zoom_slider
        self.k_slider
        self.n_slider
        self.my_interact_manual

    Methods:
        restart(Cr, Ci, zoom, n, k, colormap)
            (re)assignes parameters of class
        plot_set()
            plots the set
        create_widgets()
            creates widgets to use for interactive plot of the set
    """

    def __init__(self, Cr: Union[int,float], Ci: Union[int,float], zoom: int, n: int, k: int, colormap: str):
        """
        Assignes attributes of Set class
        
        Recommended initial parameters are:
            Cr=0, Ci=0, zoom=0, n=1000, k=100, colormap='prism'

        Parameters:
            Cr(int or float): real part of complex number C
            Ci(int or float): imaginary part of complex number C
            zoom(int): desired zoom of final plot
            n(int): resolution of final image (n*n)
            k(int): number of iterations
            colormap(str): colormap used in plot
            zoom_expo_max(int)

        Returns:
            none
        """
        #assign parameters
        self.restart(Cr, Ci, zoom, n, k, colormap)

        #ignore unimportant warnings
        warnings.filterwarnings("ignore")

    def restart(self, Cr: Union[int,float], Ci: Union[int,float], zoom: int, n: int, k: int, colormap: str):
        """
        (Re)ssignes attributes of Set class

        Parameters:
            Cr(int or float): real part of complex number C
            Ci(int or float): imaginary part of complex number C
            zoom(int): desired zoom of final plot
            n(int): resolution of final image (n*n)
            k(int): number of iterations
            colormap(str): colormap used in plot
            zoom_expo_max(int)

        Returns:
            none
        """

        #make sure imput float numbers are in correct form
        Cr = round(Cr,2)
        Ci = round(Ci,2)

        self.n = n
        self.k = k
        self.colormap = colormap

        #convert input data to axis limits
        self.Cr_min = -(10**4 - zoom)*0.0002 + Cr
        self.Cr_max = (10**4 - zoom)*0.0002 + Cr
        self.Ci_min = -(10**4 - zoom)*0.0002 - Ci
        self.Ci_max = (10**4 - zoom)*0.0002 - Ci
        #create meshgrid of all complex numbers in Cr and Ci bounds
        Cr = np.linspace(self.Cr_min, self.Cr_max, self.n)
        Ci = np.linspace(self.Ci_min, self.Ci_max, self.n)
        C_real, C_imag = np.meshgrid(Cr, Ci)
        #final array of all complex numbers C within given bounds
        self.C = C_real + 1j*C_imag

        
    def plot_set(self):
        """
        Plots the set

        Parameters:
            none
            
        Returns:
            none
        """

        #set figure size
        fig, ax = plt.subplots(figsize=(20, 12))
        #set axis properties
        tics = [0, 200, 400, 600, 800, 1000]
        ax.set_xticks(tics)
        step_Cr = (self.Cr_max-self.Cr_min)/5
        step_Ci = (self.Ci_max-self.Ci_min)/5
        ax.set_xticklabels([round(self.Cr_min,2), round(self.Cr_min+step_Cr,2), round(self.Cr_min+2*step_Cr,2), round(self.Cr_min+3*step_Cr,2), round(self.Cr_min+4*step_Cr,2), round(self.Cr_max,2)])
        ax.set_yticks(tics)
        ax.set_yticklabels([-round(self.Ci_min,2), -round(self.Ci_min+step_Ci,2), -round(self.Ci_min+2*step_Ci,2), -round(self.Ci_min+3*step_Ci,2), -round(self.Ci_min+4*step_Ci,2), -round(self.Ci_max,2)])
        ax.set_xlabel(r'$r$', fontsize=15)
        ax.set_ylabel(r'$i$', fontsize=15)
        ax.set_title('SET',fontsize=20)
        #plot finished modified divergence matrix
        ax.imshow(self.matrix, cmap=self.colormap)

    def create_widgets(self, Cr_slider_min: Union[int,float], Cr_slider_max: Union[int,float], Ci_slider_min: Union[int,float], Ci_slider_max: Union[int,float], k_slider_min: int, k_slider_max: int, n_slider_min: int, n_slider_max: int):
        """
        Creates widgtes for the interactive plot of the set

        Parameters:
            all parameters represent the minimum and maximum values of given widgets
            Cr_slider_min(int, float)
            Cr_slider_max(int, float)
            Ci_slider_min(int, float)
            Ci_slider_max(int, float)
            k_slider_min(int)
            k_slider_max(int)
            n_slider_min(int)
            n_slider_max(int)
            
        Returns:
            none
        """

        #create desired widget elements
        self.colormap_text = widgets.Dropdown(options=["prism","flag","gist_ncar","hsv", "gist_rainbow"], description='colormap')
        self.Cr_slider = widgets.FloatSlider(min=Cr_slider_min, max=Cr_slider_max, value=0.0, step=0.01, description='r coordinates')
        self.Ci_slider = widgets.FloatSlider(min=Ci_slider_min, max=Ci_slider_max, value=0.0, step=0.01, description='i coordinates')
        self.zoom_slider = widgets.IntSlider(min=0, max=9999, value=0, step=1, description='zoom')
        self.k_slider = widgets.IntSlider(min=k_slider_min, max=k_slider_max, value=100, step=1, description='iterations')
        self.n_slider = widgets.IntSlider(min=n_slider_min, max=n_slider_max, value=1000, step=1, description='resolution')
        #create interactive plot of the Julia set
        self.my_interact_manual = interact_manual.options(manual_name='apply changes')
    
class Mandelbrot(Set):
    """
    A class representing the Mandelbrot set inheriting from the Set class

    To generate the Mandelbrot set we start with a number z0=0 and repeatedly apply the function $z=z^2 + C$, where C
        are all points in the complex plane. The points either diverge or converge
        (commonly determined by comparing the numbers to 2). The Mandelbrot set is visualised by coloring the points
        based on whether they escape to infinity or stay bounded.

    Recommended use:
        set = Mandelbrot(0, 0, 0, 1000, 100, 'prism')
        set.interactive_plot()

    Addition attributes:
        None

    Methods:
        run(Cr, Ci, zoom, n, k, colormap)
            assignes given values to Mandelbrot class and calls calculate() and plot_set() methods
        calculate()
            calculates divegence matrix
        interactive_plot()
            creates interactive plot of the Julia set
    """

    def run(self, Cr: Union[int,float], Ci: Union[int,float], zoom: int, n: int, k: int, colormap: str):
        """
        (Re)assignes attributes of Mandelbrot class

        Parameters:
            Cr(int or float): real part of complex number C
            Ci(int or float): imaginary part of complex number C
            zoom(int): desired zoom of final plot
            n(int): resolution of final image (n*n)
            k(int): number of iterations
            colormap(str): colormap used in plot
            
        Returns:
            none
        """
        #reassign parameters
        self.restart(Cr, Ci, zoom, n, k, colormap)

        #calculate divergence matrix
        self.matrix = self.calculate()
        #plot Mandelbrot set
        self.plot_set()

    def calculate(self) -> NDArray[np.float64]:
        """
        Calculates divergence matrix of Mandelbrot set

        Parameters:
            none
            
        Returns:
            modified divergency matrix of Mandelbrot set
        """
        #create matrix to store results of calculations in CalculateMatrix()
        self.matrix = np.zeros((self.n,self.n))

        #as per the Mandelbrot set definition, z0 is always 0
        z = 0
        #iterate for given number of times (k)
        for i in range(self.k):
            #compute next number as per the Mandelbrot set deifnition
            z = z**2 + self.C
            #create divergence matrix (True, False values) by comparing all numbers to 2
            m = np.sqrt(z.real**2 + z.imag**2)<2
            #convert True-False values to 1-0 values
            m = (np.abs(z) < 2).astype(int)
            #add modified divergence matrix to the final matrix
            self.matrix = self.matrix + m

        return self.matrix
            
        
    def interactive_plot(self, Cr_slider_min: Union[int,float], Cr_slider_max: Union[int,float], Ci_slider_min: Union[int,float], Ci_slider_max: Union[int,float], k_slider_min: int, k_slider_max: int, n_slider_min: int, n_slider_max: int):
        """
        Creates interactive plot of the Mandelbrot set using ipywidgets

        Parameters:
            all parameters represent the minimum and maximum values of given widgets
            Cr_slider_min(int, float) - recommended -2
            Cr_slider_max(int, float) - recommended 2
            Ci_slider_min(int, float) - recommended -2
            Ci_slider_max(int, float) - recommended 2
            k_slider_min(int) - recommended 1
            k_slider_max(int) - recommended 1000
            n_slider_min(int) - recommended 1
            n_slider_max(int) - recommended 2000
            
        Returns:
            none
        """
        self.create_widgets(Cr_slider_min, Cr_slider_max, Ci_slider_min, Ci_slider_max, k_slider_min, k_slider_max, n_slider_min, n_slider_max)
        self.my_interact_manual(self.run, Cr = self.Cr_slider, Ci = self.Ci_slider, zoom = self.zoom_slider, n = self.n_slider, k = self.k_slider, colormap = self.colormap_text)


class Julia(Set):
    """
    A class representing the Julia set inheriting from the Set class
    
    To generate the Julia set we repeatedly apply the function $C=C^2 + Z$, where Z is a chosen complex constant.
        The points C either diverge or converge (commonly determined by comparing the numbers to 2).
        The Julia set is visualised by coloring the points based on whether they escape to infinity or stay bounded.
        By changing the complex constant Z we get different patterns.

    Recommended use:
        set = Julia(0, 0, 0, 1000, 100, 'prism')
        set.set_constant(-0.4, 0.6)
        set.interactPlot()

    Additional attributes:
        Z(complex): complex constant Z

    Methods:
        set_constant()
            sets the constant Z of the Julia set
        run(Zr, Zi, n, k, zoom, Cr, Ci, colormap)
            assignes given values to Julia class and calls calculate() and plot_set() methods
        calculate()
            calculates the (modified) divegence matrix of Julia set
        interactive_plot()
            creates interactive plot of the Julia set
    """
    def set_constant(self, Zr: Union[int,float], Zi: Union[int,float]):
        """
        Sets constant Z of Julia class

        Parameters:
            Zr(int or float): real number of complex constant Z
            Zi(int or float): real number of complex constant Z

        Returns:
            none
        """

        #make sure imput float numbers are in correct form
        Zr = round(Zr,2)
        Zi = round(Zi,2)

        #define complex constant Z
        self.Z = Zr+ + 1j*Zi

    def run(self, Cr: Union[int,float], Ci: Union[int,float], n: int, k: int, zoom: int, Zr: Union[int,float], Zi: Union[int,float], colormap: str):
        """
        (Re)assignes attributes of Julia class

        Parameters:
            Cr(int or float): real part of complex number C (representing the center point of the plot)
            Ci(int or float): imaginary part of complex number C (representing the center point of the plot)
            n(int): resolution of final image (n*n)
            k(int): number of iterations
            zoom(int): desired zoom of final plot (range 0-10000)
            Zr(int or float): real part of complex constant Z
            Zi(int or float): imaginary part of complex constant Z
            colormap(str): colormap used in plot
            
        Returns:
            none
        """
        #reassign parameters
        self.restart(Cr, Ci, zoom, n, k, colormap)
        #reassign constant
        self.set_constant(Zr, Zi)

        #calculate divergence matrix
        self.matrix = self.calculate()
        #plot Mandelbrot set
        self.plot_set()

    def calculate(self) -> NDArray[np.float64]:
        """
        Calculates divergence matrix (self.matrix) of Julia set

        Parameters:
            none
            
        Returns:
            modified divergency matrix of Julia set
        """

        #create matrix to store results of calculations in CalculateMatrix()
        self.matrix = np.zeros((self.n,self.n))

        #iterate for given number of times (k)
        for i in range(self.k):
            #compute next number as per the Julia set deifnition
            self.C = self.C**2 + self.Z
            #create divergence matrix (True/False values) by comparing all numbers to 2
            m = np.sqrt(self.C.real**2 + self.C.imag**2)<2
            #convert True-False values to 1-0 values
            m = (np.abs(self.C) < 2).astype(int)
            #add modified divergence matrix to the final matrix
            self.matrix = self.matrix + m

        return self.matrix
        
    def interactive_plot(self, Cr_slider_min: Union[int, float], Cr_slider_max: Union[int, float], Ci_slider_min: Union[int, float], Ci_slider_max: Union[int, float], k_slider_min: int, k_slider_max: int, n_slider_min: int, n_slider_max: int):
        """
        Creates interactive plot of the Julia set using ipywidgets
        
        Parameters:
            all parameters represent the minimum and maximum values of given widgets
            Cr_slider_min(int, float) - recommended -2
            Cr_slider_max(int, float) - recommended 2
            Ci_slider_min(int, float) - recommended -2
            Ci_slider_max(int, float) - recommended 2
            k_slider_min(int) - recommended 1
            k_slider_max(int) - recommended 1000
            n_slider_min(int) - recommended 1
            n_slider_max(int) - recommended 2000
            
        Returns:
            none
        """
        self.create_widgets(Cr_slider_min, Cr_slider_max, Ci_slider_min, Ci_slider_max, k_slider_min, k_slider_max, n_slider_min, n_slider_max)
        Zr_slider =  widgets.FloatText(value=-0.4, step=0.01, description='Z-real')
        Zi_slider =  widgets.FloatText(value=0.6, step=0.01, description='Z-imaginary')

        self.my_interact_manual(self.run, Cr = self.Cr_slider, Ci = self.Ci_slider, n = self.n_slider, k=self.k_slider, zoom = self.zoom_slider, Zr=Zr_slider, Zi=Zi_slider, colormap = self.colormap_text)
