import numpy as np
import matplotlib.pyplot as plt
from typing import Union
import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, interact_manual
from numpy.typing import NDArray

class Mandelbrot:
    """
    A class representing the Mandelbrot set
    The plot axis are r and i - real(horizontal) and imaginary(vertical)

    To generate the Mandelbrot set we start with a number z0=0 and repeatedly apply the function $z=z^2 + C$, where C
        are all points in the complex plane. The points either diverge or converge
        (commonly determined by comparing the numbers to 2). The Mandelbrot set is visualised by coloring the points
        based on whether they escape to infinity or stay bounded.

    Recommended use:
        set = Mandelbrot(0, 0, 0, 1000, 100, 'prism')
        set.interactive_plot()

    Necessary packages:
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.cm
        from typing import Union
        import ipywidgets as widgets
        from ipywidgets import interact, interactive, fixed, interact_manual

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

    Methods:
        run(Cr, Ci, zoom, n, k, colormap)
            assignes given values to Mandelbrot class and calls calculate() and plotMandelbrot() methods
        calculate()
            calculates divegence matrix
        plot_set()
            plots the Mandelbrot set
        interactive_plot()
            creates interactive plot of the Mandelbrot set
    """
    def __init__(self, Cr: Union[int,float], Ci: Union[int,float], zoom: int, n: int, k: int, colormap: str):
        """
        Assignes attributes of Mandelbrot class
        
        Recommended initial parameters are:
            Cr=0, Ci=0, zoom=0, n=1000, k=100, colormap='prism'

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

        #make sure imput float numbers are in correct form
        Cr = round(Cr,2)
        Ci = round(Ci,2)

        self.n = n
        self.k = k
        self.colormap = colormap

        #convert input data to axis limits
        self.Cr_min = -(10000 - zoom)*0.0002 + Cr
        self.Cr_max = (10000 - zoom)*0.0002 + Cr
        self.Ci_min = -(10000 - zoom)*0.0002 - Ci
        self.Ci_max = (10000 - zoom)*0.0002 - Ci
        #create meshgrid of all complex numbers in Cr and Ci bounds
        Cr = np.linspace(self.Cr_min, self.Cr_max, self.n)
        Ci = np.linspace(self.Ci_min, self.Ci_max, self.n)
        C_real, C_imag = np.meshgrid(Cr, Ci)
        #final array of all complex numbers C within given bounds
        self.C = C_real + 1j*C_imag
        
        #create matrix to store results of calculations in CalculateMatrix()
        self.matrix = np.zeros((n,n))

        #ignore unimportant warnings
        np.warnings.filterwarnings("ignore")

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

        self.n = n
        #convert input data to axis limits
        self.Cr_min = -(10000 - zoom)*0.0002 + Cr
        self.Cr_max = (10000 - zoom)*0.0002 + Cr
        self.Ci_min = -(10000 - zoom)*0.0002 - Ci
        self.Ci_max = (10000 - zoom)*0.0002 - Ci
        #create meshgrid of all complex numbers in Cr and Ci bounds
        Cr = np.linspace(self.Cr_min, self.Cr_max, self.n)
        Ci = np.linspace(self.Ci_min, self.Ci_max, self.n)
        C_real, C_imag = np.meshgrid(Cr, Ci)
        #final array of all complex numbers C within given bounds
        self.C = C_real + 1j*C_imag
        self.k = k
        self.colormap = colormap
        #create matrix to store results of calculations in CalculateMatrix()
        self.matrix = np.zeros((n,n))

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
            
    def plot_set(self):
        """
        Plots the Mandelbrot set

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
        ax.set_title('MANDELBROT SET',fontsize=20)
        #plot finished modified divergence matrix
        ax.imshow(self.matrix, cmap=self.colormap)
        
    def interactive_plot(self):
        """
        Creates interactive plot of the Mandelbrot set using ipywidgets

        Parameters:
            none
            
        Returns:
            none
        """

        #create desired widget elements
        colormap_text = widgets.Dropdown(options=["prism","flag","gist_ncar","hsv", "gist_rainbow"], description="colormap:")
        Cr_slider = widgets.FloatSlider(min=-2.0, max=2.0, value=0.0, step=0.01, description="r coordinates")
        Ci_slider = widgets.FloatSlider(min=-2.0, max=2.0, value=0.0, step=0.01, description="i coordinates")
        zoom_slider = widgets.IntSlider(min=0, max=9999, value=0, step=1, description="zoom")
        k_slider = widgets.IntSlider(min=1, max=1000, value=100, step=1, description="iterations")
        #create interactive plot of the Mandelbrot set
        my_interact_manual = interact_manual.options(manual_name="apply changes")
        my_interact_manual(self.run, Cr = Cr_slider, Ci = Ci_slider, zoom = zoom_slider, n = fixed(1000), k = k_slider, colormap = colormap_text)


class Julia:
    """
    A class representing the Julia set
    The plot axis are r and i - real(horizontal) and imaginary(vertical)
    
    To generate the Julia set we repeatedly apply the function $Z=Z^2 + C$, where C is a chosen complex constant.
        The points Z either diverge or converge (commonly determined by comparing the numbers to 2).
        The Julia set is visualised by coloring the points based on whether they escape to infinity or stay bounded.
        By changing the complex constant C we get different patterns.

    Recommended use:
        set = Julia(Zr=0, Zi=0, n=1000, k=100, zoom=0, Cr=-0.4, Ci=0.6, colormap='prism')
        set.interactPlot()

    Necessary packages:
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.cm
        from typing import Union
        import ipywidgets as widgets
        from ipywidgets import interact, interactive, fixed, interact_manual

    Attributes:
        n(int): resolution of final image (n*n)
        k(int): number of iterations
        zoom(int): desired zoom of final plot (range 0-10000)
        C(complex): complex constant C
        colormap(str): colormap used in plot
        Zr_min(int or float): real axis minimum value
        Zr_max(int or float): real axis maximum value
        Zi_min(int or float): imaginary axis minimum value
        Zi_max(int or float): imaginary axis maximum value
        Z(complex): array of all imaginary numbers in given bounds 
        matrix(np.ndarray): final (modified) divergence matrix

    Methods:
        run(Zr, Zi, n, k, zoom, Cr, Ci, colormap)
            assignes given values to Julia class and calls calculate() and plot_set() methods
        calculate()
            calculates the (modified) divegence matrix of Julia set
        plot_set()
            plots the Julia set
        interactive_plot()
            creates interactive plot of the Julia set
    """
    def __init__(self, Zr: Union[int,float], Zi: Union[int,float], n: int, k: int, zoom: int, Cr: Union[int,float], Ci: Union[int,float], colormap: str):
        """
        Constructs attributes of Julia class

        Recommended initial parameters are:
            Zr=0, Zi=0, n=1000, k=100, zoom=0, Cr=-0.4, Ci=0.6, colormap='prism'

        Parameters:
            Zr(int or float): real part of imaginary number Z
            Zi(int or float): imaginary part of imaginary number Z
            n(int): resolution of final image (n*n)
            k(int): number of iterations
            zoom(int): desired zoom of final plot (range 0-10000)
            Cr(int or float): real number of complex constant C
            Ci(int or float): real number of complex constant C
            colormap(str): colormap used in plot

        Returns:
            none
        """

        #make sure imput float numbers are in correct form
        Zr = round(Zr,2)
        Zi = round(Zi,2)
        Cr = round(Cr,2)
        Ci = round(Ci,2)

        self.n = n
        self.k = k
        self.colormap = colormap

        #convert input data to axis limits
        self.Zr_min = -(10000 - zoom)*0.0002 + Zr
        self.Zr_max = (10000 - zoom)*0.0002 + Zr
        self.Zi_min = -(10000 - zoom)*0.0002 - Zi
        self.Zi_max = (10000 - zoom)*0.0002 - Zi
        #create meshgrid of all complex numbers in Zr and Zi bounds
        Zr = np.linspace(self.Zr_min, self.Zr_max, self.n)
        Zi = np.linspace(self.Zi_min, self.Zi_max, self.n)
        Z_real, Z_imag = np.meshgrid(Zr, Zi)
        #final array of all complex numbers Z within given bounds
        self.Z = Z_real + 1j*Z_imag

        #define complex constant C
        self.C = Cr+ + 1j*Ci

        #create matrix to store results of calculations in calculate()
        self.matrix = np.zeros((n,n))

        #ignore unimportant warnings
        np.warnings.filterwarnings("ignore")

    def reload(self, Zr: Union[int,float], Zi: Union[int,float], n: int, k: int, zoom: int, Cr: Union[int,float], Ci: Union[int,float], colormap: str):
        """
        (Re)assignes attributes of Julia class

        Parameters:
            Zr(int or float): real part of complex number Z (representing the center point of the plot)
            Zi(int or float): imaginary part of complex number Z (representing the center point of the plot)
            n(int): resolution of final image (n*n)
            k(int): number of iterations
            zoom(int): desired zoom of final plot (range 0-10000)
            Cr(int or float): real part of complex constant C
            Ci(int or float): imaginary part of complex constant C
            colormap(str): colormap used in plot
            
        Returns:
            none
        """

        self.n = n
        self.k = k
        self.colormap = colormap

        #convert input data to axis limits
        self.Zr_min = -(10000 - zoom)*0.0002 + Zr
        self.Zr_max = (10000 - zoom)*0.0002 + Zr
        self.Zi_min = -(10000 - zoom)*0.0002 - Zi
        self.Zi_max = (10000 - zoom)*0.0002 - Zi
        #create meshgrid of all complex numbers in Zr and Zi bounds
        Zr = np.linspace(self.Zr_min, self.Zr_max, self.n)
        Zi = np.linspace(self.Zi_min, self.Zi_max, self.n)
        Z_real, Z_imag = np.meshgrid(Zr, Zi)
        #final array of all complex numbers Z within given bounds
        self.Z = Z_real + 1j*Z_imag

        #define complex constant C
        self.C = Cr+ + 1j*Ci

        #create matrix to store results of calculations in calculate()
        self.matrix = np.zeros((n,n))

        #calculate divergence matrix
        self.matrix = self.calculate()
        #plot Mandelbrot set
        self.plot_set()

    def calculate(self) ->np.float64:
        """
        Calculates divergence matrix (self.matrix) of Julia set

        Parameters:
            none
            
        Returns:
            modified divergency matrix of Julia set
        """

        #iterate for given number of times (k)
        for i in range(self.k):
            #compute next number as per the Julia set deifnition
            self.Z = self.Z**2 + self.C
            #create divergence matrix (True/False values) by comparing all numbers to 2
            m = np.sqrt(self.Z.real**2 + self.Z.imag**2)<2
            #convert True-False values to 1-0 values
            m = (np.abs(self.Z) < 2).astype(int)
            #add modified divergence matrix to the final matrix
            self.matrix = self.matrix + m

        return self.matrix
    
    def plot_set(self):
        """
        Plots the Julia set

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
        step_Zr = (self.Zr_max-self.Zr_min)/5
        step_Zi = (self.Zi_max-self.Zi_min)/5
        ax.set_xticklabels([round(self.Zr_min,2), round(self.Zr_min+step_Zr,2), round(self.Zr_min+2*step_Zr,2), round(self.Zr_min+3*step_Zr,2), round(self.Zr_min+4*step_Zr,2), round(self.Zr_max,2)])
        ax.set_yticks(tics)
        ax.set_yticklabels([-round(self.Zi_min,2), -round(self.Zi_min+step_Zi,2), -round(self.Zi_min+2*step_Zi,2), -round(self.Zi_min+3*step_Zi,2), -round(self.Zi_min+4*step_Zi,2), -round(self.Zi_max,2)])
        ax.set_xlabel(r'$r$', fontsize=15)
        ax.set_ylabel(r'$i$', fontsize=15)
        ax.set_title('JULIA SET',fontsize=20)
        #plot finished modified divergence matrix
        ax.imshow(self.matrix, cmap=self.colormap)
        
    def interactive_plot(self):
        """
        Creates interactive plot of the Julia set using ipywidgets

        Parameters:
            none
            
        Returns:
            none
        """

        #create desired widget elements
        colormap_text = widgets.Dropdown(options=["prism","flag","gist_ncar","hsv", "gist_rainbow"], description="colormap:")
        Zr_slider = widgets.FloatSlider(min=-2, max=2, value=0.0, step=0.01, description="r coordinates")
        Zi_slider = widgets.FloatSlider(min=-2, max=2, value=0.0, step=0.01, description="i coordinates")
        zoom_slider = widgets.IntSlider(min=0, max=9999, value=0, step=1, description="zoom")
        Cr_slider =  widgets.FloatText(min=-5.0, max=5.0, value=-0.4, step=0.01, description="real part of C")
        Ci_slider =  widgets.FloatText(min=-5.0, max=5.0, value=0.6, step=0.01, description="imaginary part of C")
        k_slider = widgets.IntSlider(min=1, max=1000, value=100, step=1, description="iterations")
        #create interactive plot of the Julia set
        my_interact_manual = interact_manual.options(manual_name="apply changes")
        my_interact_manual(self.reload, Zr = Zr_slider, Zi = Zi_slider, n = fixed(1000), k=k_slider, zoom = zoom_slider, Cr=Cr_slider, Ci=Ci_slider, colormap = colormap_text)

        
    