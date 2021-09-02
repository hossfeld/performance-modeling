# -*- coding: utf-8 -*-
"""
The module provides a class for finite discrete distributions which are utilized for discrete-time analysis.
For example, discrete-time GI/GI/1 systems can be analyzed with functions of the module.

(c) Tobias Hossfeld  (Aug 2021)

This module is part of the following book. The book is to be cited whenever the script is used (copyright CC BY-SA 4.0):

>Tran-Gia, P. & Hossfeld, T. (2021). 
>Performance Modeling and Analysis of Communication Networks - A Lecture Note.
>Würzburg University Press. <br>
>https://doi.org/10.25972/WUP-978-3-95826-153-2


Example
-------
We can easily define some discrete distribution and do computations with the corresponding random variables.
In the example, we consider the sum of two random variables, which requires the convolution of the corresponding probability mass functions.
The r.v. A follows a discrete uniform distribution in the range [0;10], while the r.v. B follows a negative binomial distribution,
which is defined through the mean and the coefficient of variation.

>>> import discreteTimeAnalysis as dt
>>> A = dt.DU(a=0, b=10) % A ~ DU(0,10)
>>> EX, cX = 2.0, 1.5   % mean EX and coefficient of variation cX
>>> B = dt.NEGBIN(EX, cX) % negative binomial distribution
>>> C = A + B % sum of random variables requires convolution of PMFs
>>> C.plotCDF(label='A+B') % plot the CDF of the sum of A+B


Operators
---------
Overloaded operators


Notes
-----
The theory behind the module is described in the book in Chapter 6. 
The text book is published as open access book and can be downloaded at
<https://modeling.systems>


"""

import numpy as np
import matplotlib.pyplot as plt
import math
#%%    
class DiscreteDistribution:
    """The class implements finite discrete distributions representing discrete random variables.

    A discrete distribution reflects a random variable \( X \) and is defined 
    by its probability mass function (PMF). The random variable can take discrete values
    which are defined by the numpy array `xk` (sample space). The probability that the random variable
    takes a certain value is \( P(X=k)=p_k \). The probabilities are stored in the
    numpy array `pk`.
    

    Attributes
    ----------
    xk : numpy array
        Values of the distribution (sample space).
    pk : numpy array
        Probabilities corresponding to the sample space.
    name : string
        Arbitrary name of that distribution. 

    """    
    
    def __init__(self, xk, pk, name='discrete distr.'):         
        """A discrete distribution is initialized with value range `xk`and probabilities `pk`.
        
        For the initialization of a discrete random variable, the sample space `xk` and the corresponding
        probabilities `pk` are required. Both parameters are then stored as class attributes in form
        of numpy array (one-dimensional). In addition, an arbitrary `name` can be passed to the
        distribution which is used when printing an instance of the class, see e.g. 
        `DiscreteDistribution.describe`.

        Parameters
        ----------
        xk : numpy array or list
            Values of the distribution.
        pk : numpy array or list
            Probabilities corresponding to the values: \( P(X=xk)=pk \).
        name : string, optional (default 'discrete distr.')
            Name of the distribution for string representation.

        """        
        assert len(xk)==len(pk) # same length        
        
        self.xmin = np.min(xk)
        self.xmax = np.max(xk)
        
        # adjust to vector xk without gaps
        self.xk = np.arange(self.xmin, self.xmax+1, dtype='int')
        self.pk = np.zeros( len(self.xk) )
        self.pk[xk-self.xmin] = pk
        self.name = name
        
    def mean(self):
        """Returns the mean value of the distribution \( E[X] \).
    
    
        Returns
        -------
        float
            Mean value.
            
        """        
        return np.sum(self.xk*self.pk)
    
    def var(self):
        """Returns the variance of the distribution \( VAR[X] \).
    
    
        Returns
        -------
        float
            Variance of the distribution.
            
        """                
        return np.sum(self.xk**2*self.pk)-self.mean()**2
    
    def std(self):
        """Returns the standard deviation of the distribution \( {STD}[X]=\sqrt{VAR[X]} \).
    
    
        Returns
        -------
        float
            Standard deviation of the distribution.
            
        """                
        return math.sqrt(self.var())
    
    def cx(self):
        """Returns the coefficient of the variation of the distribution \( c_X = STD[X]/E[X] \).
    
    
        Returns
        -------
        float
            Coefficient of variation of the distribution.
            
        """               
        return self.std()/self.mean()
    
    def mode(self):
        """Returns the mode of the distribution.
    
    
        Returns
        -------
        float
            Mode of the distribution.
            
        """                
        return self.xk[np.argmax(self.pk)]
    
    def quantile(self, q=0.95):
        """Returns the q-quantile of the distribution.
    
        Parameters
        ----------
        q : float, optional (default 0.95)
            The parameter indicates that the q-quantile is derived. The default value is `q=0.95`
            for the 95%-quantile. It must be ensured that \( 0< q < 1\).
    
        Returns
        -------
        float
            q-Quantile (default 95%) of the distribution.
            
        """                
        return self.xk[np.argmax(self.pk.cumsum()>q)]
    
    def describe(self):
        """Prints basic characteristics of the distribution.        
        
        This method prints basic characteristics of the distribution.
        
        Example
        -------
        >>> A.describe()
            interarrival_time: EX=5.5000, cX=0.5222, mode=1 
                        
        """               
        print(f'{self.name}: EX={self.mean():.4f}, cX={self.cx():.4f}, mode={self.mode()} ')

    def checkDistribution(self):
        """Returns if the distribution is valid.         
    
        Returns
        -------
        bool
            Return true if the distribution is valid. 
            Returns false if e.g. the values of `xk` are not increasing or the sum of probabilities `pk` is less than 1.
            
        """                
        increasing = np.all(np.diff(self.xk) > 0) # xk: strictly monotonic increasing
        sumOne = abs(np.sum(self.pk)-1)<1e-8 # xk: error
        return increasing and sumOne
            
    def conv(A,B,name=None):
        """Returns the sum of the two distributions.
        
        Returns the sum of the distribution `A` and the distribution `B`. Note that \( A+B=B+A \).
        
        
        Parameters
        ----------
        A : DiscreteDistribution
            The first distribution of the sum.
        B : DiscreteDistribution
            The second distribution of the sum.
            
        Example
        -------
        >>> A = DU()
        >>> A.conv(A) # returns A+A
        >>> DiscreteDistribution.conv(A,A) # returns A+A
        >>> A+A # returns A+A
    
        Returns
        -------
        DiscreteDistribution
            Sum of the distributions: `A+B`.
            
        See also
        --------
        Whatsoever
            
        """                
        pk = np.convolve(A.pk, B.pk)
        xk = np.arange(A.xmin+B.xmin, A.xmax+B.xmax+1)
        return DiscreteDistribution(xk,pk,name=name)    
    
    def convNeg(A,B,name='A-B'):
        """Discrete distribution is initialized with value range `xk` and probabilities `pk`.

        Parameters
        ----------
        xk : numpy array
            Values of the distribution.
        pk : numpy array
            Probabilities corresponding to the values: \( P(X=xk)=pk \).
        name : string, optional (default 'discrete distr.')
            Name of the distribution for string representation.

        """                
        pk = np.convolve(A.pk, B.pk[::-1])
        xk = np.arange(A.xmin-B.xmax, A.xmax-B.xmin+1)
        return DiscreteDistribution(xk,pk,name=name)
        
    def pi_op(A, m=0, name=None):        
        """Discrete distribution is initialized with value range `xk`and probabilities `pk`.

        Parameters
        ----------
        xk : numpy array
            Values of the distribution.
        pk : numpy array
            Probabilities corresponding to the values: \( P(X=xk)=pk \).
        name : string, optional (default 'discrete distr.')
            Name of the distribution for string representation.

        """                
        s = f'pi_{m}({A.name})' if name is None else name     
        if m <= A.xmin:
            A.name = s
            return A
        elif m >= A.xmax:
            return  DiscreteDistribution([m],[1],name=s)
        else:
            #s = f'pi_{m}({A.name})' if name is None else name        
            k = np.searchsorted(A.xk,m)
            xk = np.arange(m, A.xmax+1)
            pk = np.zeros(len(xk))
            pk[0] = np.sum(A.pk[0:k+1])
            pk[1:] = A.pk[k+1:]
            return DiscreteDistribution(xk,pk,name=s)        
    
    def pi0(A, name=None):
        """Discrete distribution is initialized with value range `xk`and probabilities `pk`.

        Parameters
        ----------
        xk : numpy array
            Values of the distribution.
        pk : numpy array
            Probabilities corresponding to the values: \( P(X=xk)=pk \).
        name : string, optional (default 'discrete distr.')
            Name of the distribution for string representation.

        """                
        s = f'pi0({A.name})' if name is None else name
        return DiscreteDistribution.pi_op(A, m=0, name=s)
    
    # remove trailing zero probs, shorten vector
    # the distribution A is changed
    def trim(A):
        """Discrete distribution is initialized with value range `xk`and probabilities `pk`.

        Parameters
        ----------
        xk : numpy array
            Values of the distribution.
        pk : numpy array
            Probabilities corresponding to the values: \( P(X=xk)=pk \).
        name : string, optional (default 'discrete distr.')
            Name of the distribution for string representation.

        """                
        m = A.pk!=0        
        kmin = m.argmax()
        kmax = m.size - m[::-1].argmax()-1
        
        #A.xmin = np.min(xk)
        #A.xmax = np.max(xk)        
        
        A.xk = A.xk[kmin:kmax+1]
        A.pk = A.pk[kmin:kmax+1]
        
        A.xmin = A.xk[0]
        A.xmax = A.xk[-1]
        return 
    
    # remove trailing zero probs, shorten vector
    # the distribution A is changed
    def trimEps(A, eps=1e-8):
        """Discrete distribution is initialized with value range `xk`and probabilities `pk`.

        Parameters
        ----------
        xk : numpy array
            Values of the distribution.
        pk : numpy array
            Probabilities corresponding to the values: \( P(X=xk)=pk \).
        name : string, optional (default 'discrete distr.')
            Name of the distribution for string representation.

        """                
        m = A.pk>eps #!=0        
        kmin = m.argmax()
        kmax = m.size - m[::-1].argmax()-1
        
        #A.xmin = np.min(xk)
        #A.xmax = np.max(xk)        
        
        A.xk = A.xk[kmin:kmax+1]
        A.pk = A.pk[kmin:kmax+1]
        
        A.xmin = A.xk[0]
        A.xmax = A.xk[-1]
        return 
    
    
    # this is an unnormalized distribution: 
    # conditional distribution if normalized
    # sigmaLT = sigma^m: takes the lower part ( k < m ) of a distribution
    def sigmaLT(self, m=0, name=None, normalized=True):        
        """Discrete distribution is initialized with value range `xk`and probabilities `pk`.

        Parameters
        ----------
        xk : numpy array
            Values of the distribution.
        pk : numpy array
            Probabilities corresponding to the values: \( P(X=xk)=pk \).
        name : string, optional (default 'discrete distr.')
            Name of the distribution for string representation.

        """                
        #assert m<xk[-1]
        s = f'sigma^{m}({self.name})' if name is None else name     
                
        if m<=self.xk[0]:
            if normalized: 
                raise ValueError('sigmaLT: m < min(xk)')
            else:
                return DiscreteDistribution([m], [0], name=s)
        if m>self.xk[-1]:
            return DiscreteDistribution(self.xk, self.pk, name=s)
            
        last = np.searchsorted(self.xk, m, side='right')-1        
        
        xk=self.xk[:last]
        if normalized:
            prob_Dist_U_lt_m = self.pk[:last].sum() 
            pk=self.pk[:last] / prob_Dist_U_lt_m
        else:
            pk=self.pk[:last]                
        return DiscreteDistribution(xk, pk, name=s)
    
    # this is an unnormalized distribution: 
    # conditional distribution if normalized
    def sigmaGEQ(self, m=0, name=None, normalized=True):
        """Discrete distribution is initialized with value range `xk`and probabilities `pk`.

        Parameters
        ----------
        xk : numpy array
            Values of the distribution.
        pk : numpy array
            Probabilities corresponding to the values: \( P(X=xk)=pk \).
        name : string, optional (default 'discrete distr.')
            Name of the distribution for string representation.

        """                
        s = f'sigma_{m}({self.name})' if name is None else name     
        #assert m>=self.xk[0]
        if m>self.xk[-1]:
            if normalized: 
                raise ValueError('sigmaGEQ: m > max(xk)')
            else:
                return DiscreteDistribution([m], [0], name=s)                    
        
        first = np.searchsorted(self.xk, m, side='left')
        
        xk=self.xk[first:]
        if normalized:
            prob_Dist_U_geq_m = self.pk[first:].sum() 
            pk=self.pk[first:] / prob_Dist_U_geq_m
        else:
            pk=self.pk[first:]                
        return DiscreteDistribution(xk, pk, name=s)
    
    def pmf(self, xi):
        """Discrete distribution is initialized with value range `xk`and probabilities `pk`.

        Parameters
        ----------
        xk : numpy array
            Values of the distribution.
        pk : numpy array
            Probabilities corresponding to the values: \( P(X=xk)=pk \).
        name : string, optional (default 'discrete distr.')
            Name of the distribution for string representation.

        """                
        #myxk = np.arange(self.xmin-1, self.xmax+2)
        #mypk = np.hstack((0, self.pk, 0))
        if type(xi) is not np.ndarray:
            if type(xi) is list:
                xi = np.array(xi)
            else:
                xi = np.array([xi])
        
        i = np.where( (xi>=self.xmin) & (xi<=self.xmax) )[0]
        mypk = np.zeros(len(xi))
        
        if len(i)>0:            
            mypk[i] = self.pk[np.searchsorted(self.xk, xi[i], side='left')]
        return mypk
    
    def plotCDF(self,  addZero=True, **kwargs):
        """Discrete distribution is initialized with value range `xk`and probabilities `pk`.

        Parameters
        ----------
        xk : numpy array
            Values of the distribution.
        pk : numpy array
            Probabilities corresponding to the values: \( P(X=xk)=pk \).
        name : string, optional (default 'discrete distr.')
            Name of the distribution for string representation.

        """                
        if addZero and self.xk[0]>=0:
            x = np.insert(self.xk,0,0)
            y = np.insert(self.pk,0,0)
        else:
            x, y = self.xk, self.pk
        
        x = np.append(x, x[-1]+1)
        Y = np.append(y.cumsum(), 1)
        
        plt.step(x, Y, '.-', where='post', **kwargs)
        
    def plotPMF(self,  **kwargs):
        """Discrete distribution is initialized with value range `xk`and probabilities `pk`.

        Parameters
        ----------
        xk : numpy array
            Values of the distribution.
        pk : numpy array
            Probabilities corresponding to the values: \( P(X=xk)=pk \).
        name : string, optional (default 'discrete distr.')
            Name of the distribution for string representation.

        """                
        plt.plot(self.xk, self.pk, '.-', **kwargs)        
        
    def conditionalRV(self, condition):
        """Discrete distribution is initialized with value range `xk`and probabilities `pk`.

        Parameters
        ----------
        xk : numpy array
            Values of the distribution.
        pk : numpy array
            Probabilities corresponding to the values: \( P(X=xk)=pk \).
        name : string, optional (default 'discrete distr.')
            Name of the distribution for string representation.

        """                
        return self
        
    
    # A+B
    def __add__(self, other): 
        """Discrete distribution is initialized with value range `xk`and probabilities `pk`.

        Parameters
        ----------
        xk : numpy array
            Values of the distribution.
        pk : numpy array
            Probabilities corresponding to the values: \( P(X=xk)=pk \).
        name : string, optional (default 'discrete distr.')
            Name of the distribution for string representation.

        """                
        return DiscreteDistribution.conv(self,other,name=f'{self.name}+{other.name}')
    
    # A-C
    def __sub__(self, other): 
        """Discrete distribution is initialized with value range `xk`and probabilities `pk`.

        Parameters
        ----------
        xk : numpy array
            Values of the distribution.
        pk : numpy array
            Probabilities corresponding to the values: \( P(X=xk)=pk \).
        name : string, optional (default 'discrete distr.')
            Name of the distribution for string representation.

        """                
        return DiscreteDistribution.convNeg(self,other,name=f'{self.name}-{other.name}')
    
    # A<B: based on means
    def __lt__(self, other):        
        """Discrete distribution is initialized with value range `xk`and probabilities `pk`.

        Parameters
        ----------
        xk : numpy array
            Values of the distribution.
        pk : numpy array
            Probabilities corresponding to the values: \( P(X=xk)=pk \).
        name : string, optional (default 'discrete distr.')
            Name of the distribution for string representation.

        """                
        return self.mean() < other.mean()
    
    def __le__(self, other):        
        """Discrete distribution is initialized with value range `xk`and probabilities `pk`.

        Parameters
        ----------
        xk : numpy array
            Values of the distribution.
        pk : numpy array
            Probabilities corresponding to the values: \( P(X=xk)=pk \).
        name : string, optional (default 'discrete distr.')
            Name of the distribution for string representation.

        """                
        return self.mean() <= other.mean()
    
    def __eq__(self, other):        
        """Discrete distribution is initialized with value range `xk`and probabilities `pk`.

        Parameters
        ----------
        xk : numpy array
            Values of the distribution.
        pk : numpy array
            Probabilities corresponding to the values: \( P(X=xk)=pk \).
        name : string, optional (default 'discrete distr.')
            Name of the distribution for string representation.

        """                
        if len(self.xk) != len(other.xk): return False
        return np.all(self.xk==other.xk) and np.all(self.pk==other.pk)
    
    def __repr__(self):
        """Discrete distribution is initialized with value range `xk`and probabilities `pk`.

        Parameters
        ----------
        xk : numpy array
            Values of the distribution.
        pk : numpy array
            Probabilities corresponding to the values: \( P(X=xk)=pk \).
        name : string, optional (default 'discrete distr.')
            Name of the distribution for string representation.

        """                
        return self.__str__()

    def __str__(self):
        if len(self.xk)<10:
            return f'{self.name}: xk={np.array2string(self.xk,separator=",")}, pk={np.array2string(self.pk,precision=3, separator=",")}'
        else:
            return f'{self.name}: xk={self.xmin},...,{self.xmax}, pk={self.pk[0]:g},...,{self.pk[-1]:g}'
    
def pi_op(A, m=0, name=None):    
    """Example function with types documented in the docstring.

    `PEP 484`_ type annotations are supported. If attribute, parameter, and
    return types are annotated according to `PEP 484`_, they do not need to be
    included in the docstring:

    Parameters
    ----------
    param1 : int
        The first parameter.
    param2 : str, optional (default '5')
        The second parameter.

    Returns
    -------
    bool
        True if successful, False otherwise.
    """ 
    return DiscreteDistribution.pi_op(A,m,name)

def pi0(A, name=None):
    """Example function with types documented in the docstring.

    `PEP 484`_ type annotations are supported. If attribute, parameter, and
    return types are annotated according to `PEP 484`_, they do not need to be
    included in the docstring:

    Parameters
    ----------
    param1 : int
        The first parameter.
    param2 : str, optional (default '5')
        The second parameter.

    Returns
    -------
    bool
        True if successful, False otherwise.
    """ 
    return DiscreteDistribution.pi0(A, name=name)    

def conv(A,B,name='A+B'):
    """Example function with types documented in the docstring.

    `PEP 484`_ type annotations are supported. If attribute, parameter, and
    return types are annotated according to `PEP 484`_, they do not need to be
    included in the docstring:

    Parameters
    ----------
    param1 : int
        The first parameter.
    param2 : str, optional (default '5')
        The second parameter.

    Returns
    -------
    bool
        True if successful, False otherwise.
    """    
    pk = np.convolve(A.pk, B.pk)
    xk = np.arange(A.xmin+B.xmin, A.xmax+B.xmax+1)
    return (xk, pk)


def plotCDF(A, addZero=True, **kwargs):
    """Example function with types documented in the docstring.

    `PEP 484`_ type annotations are supported. If attribute, parameter, and
    return types are annotated according to `PEP 484`_, they do not need to be
    included in the docstring:

    Parameters
    ----------
    param1 : int
        The first parameter.
    param2 : str, optional (default '5')
        The second parameter.

    Returns
    -------
    bool
        True if successful, False otherwise.
    """     
    A.plotCDF(addZero, **kwargs)
    

def gg1_waitingTime(W0, C, epsMean=1e-4, epsProb=1e-16):
    """Example function with types documented in the docstring.

    `PEP 484`_ type annotations are supported. If attribute, parameter, and
    return types are annotated according to `PEP 484`_, they do not need to be
    included in the docstring:

    Parameters
    ----------
    param1 : int
        The first parameter.
    param2 : str, optional (default '5')
        The second parameter.

    Returns
    -------
    bool
        True if successful, False otherwise.
    """     
    diff = 1
    Wn = W0
    i = 0
    while diff>epsMean:
        Wn1 = pi0(Wn+C)
        diff = np.abs(Wn.mean()-Wn1.mean())
        Wn = Wn1
        Wn.name = None
        Wn.trimEps(eps=epsProb)
        i += 1
    return Wn, i
        
    
def kingman(rho, cA, cB, EB):
    """Example function with types documented in the docstring.

    `PEP 484`_ type annotations are supported. If attribute, parameter, and
    return types are annotated according to `PEP 484`_, they do not need to be
    included in the docstring:

    Parameters
    ----------
    param1 : int
        The first parameter.
    param2 : str, optional (default '5')
        The second parameter.

    Returns
    -------
    bool
        True if successful, False otherwise.
    """     
    return rho/(1-rho)*EB*(cA**2+cB**2)/2

#%% Bernoulli distribution

# Binomial distribution

# poisson distribution


#%% NEGBIN files
from scipy.stats import nbinom, geom

def getNegBinPars(mu,cx):
    """Example function with types documented in the docstring.

    `PEP 484`_ type annotations are supported. If attribute, parameter, and
    return types are annotated according to `PEP 484`_, they do not need to be
    included in the docstring:

    Parameters
    ----------
    param1 : int
        The first parameter.
    param2 : str, optional (default '5')
        The second parameter.

    Returns
    -------
    bool
        True if successful, False otherwise.
    """     
    z = cx**2*mu-1    
    return mu/z, 1- z/(cx**2*mu)


def NEGBIN(EX,cx, eps=1e-8):
    """Example function with types documented in the docstring.

    `PEP 484`_ type annotations are supported. If attribute, parameter, and
    return types are annotated according to `PEP 484`_, they do not need to be
    included in the docstring:

    Parameters
    ----------
    param1 : int
        The first parameter.
    param2 : str, optional (default '5')
        The second parameter.

    Returns
    -------
    bool
        True if successful, False otherwise.
    """     
    r,p = getNegBinPars(EX,cx)
    rv = nbinom(r,p)
    cut = int(rv.isf(eps))
    #print(f'cut at {cut}')
    x = np.arange(cut)
    pk = rv.pmf(x)
    return DiscreteDistribution(x, pk/pk.sum())

def DET(EX, name=None):
    """Example function with types documented in the docstring.

    `PEP 484`_ type annotations are supported. If attribute, parameter, and
    return types are annotated according to `PEP 484`_, they do not need to be
    included in the docstring:

    Parameters
    ----------
    param1 : int
        The first parameter.
    param2 : str, optional (default '5')
        The second parameter.

    Returns
    -------
    bool
        True if successful, False otherwise.
    """     
    return DiscreteDistribution([EX], [1.0], name=name)
    
def DU(a=1, b=10, name=None):    
    """Example function with types documented in the docstring.

    `PEP 484`_ type annotations are supported. If attribute, parameter, and
    return types are annotated according to `PEP 484`_, they do not need to be
    included in the docstring:

    Parameters
    ----------
    param1 : int
        The first parameter.
    param2 : str, optional (default '5')
        The second parameter.

    Returns
    -------
    bool
        True if successful, False otherwise.
    """     
    xk = np.arange(a,b+1)
    n = b-a+1
    pk = 1.0/n
    return DiscreteDistribution(xk, np.array([pk]*n), name=name)    

def GEOM(EX, m=0, eps=1e-8, name=None):
    """Example function with types documented in the docstring.

    `PEP 484`_ type annotations are supported. If attribute, parameter, and
    return types are annotated according to `PEP 484`_, they do not need to be
    included in the docstring:

    Parameters
    ----------
    param1 : int
        The first parameter.
    param2 : str, optional (default '5')
        The second parameter.

    Returns
    -------
    bool
        True if successful, False otherwise.
    """     
    p = 1.0/(EX+1-m)    
    rv = geom(p, loc=m-1)
    cut = int(rv.isf(eps))    
    x = np.arange(cut)
    pk = rv.pmf(x)
    return DiscreteDistribution(x, pk/pk.sum(), name=name)

#%% substitute distribution
def substiteDistribution(EX, cX, name=None):
    """Example function with types documented in the docstring.

    `PEP 484`_ type annotations are supported. If attribute, parameter, and
    return types are annotated according to `PEP 484`_, they do not need to be
    included in the docstring:

    Parameters
    ----------
    param1 : int
        The first parameter.
    param2 : str, optional (default '5')
        The second parameter.

    Returns
    -------
    bool
        True if successful, False otherwise.
    """     
    if cX==0:
        return DET([EX], [1.0], name=name)
    elif cX==1:
        return GEOM(EX, name=name)
    
#%% mixture distribution
def MIX(A,  w, name=None):
    """Example function with types documented in the docstring.

    `PEP 484`_ type annotations are supported. If attribute, parameter, and
    return types are annotated according to `PEP 484`_, they do not need to be
    included in the docstring:

    Parameters
    ----------
    param1 : int
        The first parameter.
    param2 : str, optional (default '5')
        The second parameter.

    Returns
    -------
    bool
        True if successful, False otherwise.
    """     
    xkMin = min(list(map(lambda Ai: Ai.xk[0], A)))
    xkMax = max(list(map(lambda Ai: Ai.xk[-1], A)))
    
    xk = np.arange( xkMin, xkMax+1)
    pk = np.zeros(len(xk))
    for (Ai, wi) in zip(A,w):
        iA = np.searchsorted(xk, Ai.xk, side='left')
        pk[iA] += Ai.pk*wi
    return DiscreteDistribution(xk, pk, name=name)

