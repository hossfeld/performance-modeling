# -*- coding: utf-8 -*-
"""
StateTransitionGraph: A directed graph class for modeling continuous-time Markov chains (CTMCs).

This module defines the `StateTransitionGraph` class, an extension of `networkx.DiGraph` 
with specialized methods and attributes for analyzing, simulating, and visualizing 
continuous-time Markov chains (CTMCs).

Key Features
------------
- Symbolic and numerical parameter support using SymPy.
- Automatic generation and storage of the rate matrix (Q-matrix).
- Simulation of state transitions and computation of steady-state probabilities.
- Enhanced graph visualization with transition rates and labels.
- Support for dynamic import of constants via a custom module (default: "Constants").

Typical Use Case
----------------
1. Define a set of model parameters (e.g., lambda, mu, n).
2. Create a `StateTransitionGraph` instance with those parameters.
3. Add states and transitions with symbolic rates.
4. Compute the rate matrix and steady-state probabilities.
5. Run simulations and visualize the CTMC behavior.


Example
-------
>>> parameters = {'lambda': lam, 'mu': 1.0, 'n': 5}
>>> G = StateTransitionGraph(parameters)

>>> # Define transitions
>>> for i in range(parameters['n']):
>>>     G.addTransition(i, i + 1, G.sym("lambda"), tid=const.ARRIVAL)
>>>     G.addTransition(i + 1, i, (i + 1) * G.sym("mu"), tid=const.DEPARTURE)

>>> # Set node properties
>>> G.setAllStateDefaultProperties()
>>> G.setStateColor(G.n, const.COLOR_NODE_BLOCKING)

>>> # Compute steady-state probabilities
>>> probs = G.calcSteadyStateProbs()

>>> # Plot bar chart of state probabilities
>>> import matplotlib.pyplot as plt
>>> plt.figure(1, clear=True)
>>> plt.bar(probs.keys(), probs.values())
>>> plt.xlabel('states')
>>> plt.ylabel('probability')

>>> # Layout for transition graph
>>> pos = {s: (s, 0) for s in G.states()}
>>> G.drawTransitionGraph(pos=pos, bended=True, label_pos=0.5, num=2)

>>> # Simulate and animate system dynamics
>>> states, times = G.simulateMarkovModel(startNode=0, numEvents=2000)

See Also
--------
networkx.DiGraph : The base class that is extended.
sympy.symbols   : Used to define symbolic transition rates.
matplotlib.pyplot : Used internally for graph visualization.


Author
------
Tobias Hossfeld <tobias.hossfeld@uni-wuerzburg.de>

Created
-------
5 April 2025 

"""


import networkx as nx  # For the magic
import matplotlib.pyplot as plt  # For plotting
import numpy as np
from scipy import linalg
#import itertools
import sympy as sp
import warnings
import random
import importlib
from scipy.linalg import expm
from IPython.display import display, clear_output
#import Constants as const
#from collections import namedtuple
#%%
class StateTransitionGraph(nx.DiGraph):
    """
    Initialize the StateTransitionGraph for continuous time Markov chains (CTMC). 
    Extends the networkx.Graph class to include additional methods and properties 
    for the analysis, visualization, and simulation of CTMCs. 
    The analysis allows deriving the state probabilities in the steady-state 
    and in the transient phase.

    This constructor sets up symbolic and numeric parameters for the state transition graph,
    imports a constants module dynamically, and initializes internal data structures for
    rate matrix computation and steady-state probability analysis.

    Parameters
    ----------
    - `parameters` : dict  
      A dictionary mapping parameter names to their numerical values.     
        Example: {'lambda': 1.0, 'mu': 1.0}. 
        For each parameter key:
        - A symbolic variable is created using sympy.symbols. Can be accessed as `G.sym("key")` or `G.sym_key`
        - A variable is created which can be accessed via `G.key` (and the numerical value is assigned)
        - For the key="lambda", `G.lam` and `G.sym_lambda` are created
    
        The parameters can be accessed via property `parameters`
    - `constants` : str, optional
        The name of a Python module (as string) to import dynamically and assign to `self.const`.
        This can be used to store global or user-defined constants.
        Default is "Constants".
    - `*args` : tuple
        Additional positional arguments passed to the base `networkx.DiGraph` constructor.
    - `**kwargs` : dict
        Additional keyword arguments passed to the base `networkx.DiGraph` constructor.

    Attributes
    ----------
    - `_rate_matrix` : ndarray or None  
      The continuous-time rate matrix `Q` of the CTMC. Computed when calling `createRateMatrix()`.
    - `_state_probabilities` : dict or None  
      Dictionary of steady-state probabilities, computed via `calcSteadyStateProbs()`.
    - `_n2i` : dict or None  
      Mapping from node identifiers to matrix row/column indices used for rate matrix construction.
    - `_sym` : dict  
      Mapping of parameter names (strings) to their symbolic `sympy` representations.
    - `_subs` : dict  
      Mapping from symbolic parameters to their numerical values (used for substitutions).
    - `_parameters` : dict  
      Copy of the original parameter dictionary passed at initialization.
    - `_transitionProbabilitesComputed` : bool  
      Internal flag indicating whether transition probabilities for the embedded Markov chain
      have been computed.
    - `const` : module  
      Dynamically imported module (default `"Constants"`), used for storing user-defined constants 
      such as colors or transition IDs.
        
    Inherits
    --------
        networkx.Graph: The base graph class from NetworkX.
    """

    def __init__(self, parameters, constants="Constants", *args, **kwargs):
        """
        Initialize the state transition graph with symbolic parameters.

        This constructor sets up the symbolic and numerical environment for modeling
        a CTMC. Parameters are stored as both symbolic expressions and numerical values,
        and convenient attribute access is provided for both forms.

        Parameters
        ----------
        - `parameters` : dict  
          A dictionary mapping parameter names to their numerical values.     
            Example: {'lambda': 1.0, 'mu': 1.0}. 
            For each parameter key:
            - A symbolic variable is created using sympy.symbols. Can be accessed as `G.sym("key")` or `G.sym_key`
            - A variable is created which can be accessed via `G.key` (and the numerical value is assigned)
            - For the key="lambda", `G.lam` and `G.sym_lambda` are created
        
            The parameters can be accessed via property `parameters`
        - `constants` : str, optional
            The name of a Python module (as string) to import dynamically and assign to `self.const`.
            This can be used to store global or user-defined constants.
            Default is "Constants".
        - `*args` : tuple
            Additional positional arguments passed to the base `networkx.DiGraph` constructor.
        - `**kwargs` : dict
            Additional keyword arguments passed to the base `networkx.DiGraph` constructor.
        """        
        super().__init__(*args, **kwargs)
        self._rate_matrix = None  # Rate Matrix Q of this transition graph
        self._state_probabilities = None  # steady state proabilities of this CTMC
        self._n2i = None # internal index for generating the rate matrix Q: dictionary (node to index)
        
        #parameters = {'lambda': 1.0, 'mu':1.0, 'n':5}
        self._parameters = parameters
        self._sym = {key: sp.symbols(key) for key in parameters}
        self._subs = {} # contains symbolic parameter as key and provides numerical value for computation.
        for k in parameters:
            self._subs[ self._sym[k] ] = parameters[k] 

        # access to parameters and symbols using dot-notation, e.g. G.lam, G.sym_lambda, or G.sym("lambda")
        for key, value in parameters.items():
            if key=='lambda':
                setattr(self, 'lam', value)
            else:
                setattr(self, key, value)
            
            setattr(self, "sym_"+key, self._sym[key])
            
        self._transitionProbabilitesComputed = False
        
        self._const = importlib.import_module(constants)

        
        
    def addState(self, state, color=None, label=None):
        """
        Add a state (node) to the graph with optional color and label attributes.
    
        If the node already exists, its attributes are updated with the provided values.
    
        Parameters
        ----------
        - state : hashable
            The identifier for the state. Can be a string, int, or a tuple. If no label
            is provided, the label will be auto-generated as:
            - comma-separated string for tuples (e.g., `(1, 2)` -> `'1,2'`)
            - stringified value for other types
    
        - color : str, optional
            The color assigned to the state (used for visualization).
            If not specified, defaults to `self.const.COLOR_NODE_DEFAULT`.
    
        - label : str, optional
            A label for the node (used for display or annotation). If not provided,
            it will be auto-generated from the state identifier.
    
        Notes
        -----
        - This method wraps `self.add_node()` from NetworkX.
        - If the node already exists in the graph, this method does not raise an error;
          it will update the node’s existing attributes.
        """
        if label is None:
            if isinstance(state, tuple):
                label = ','.join(map(str, state)) # if state is a tuple
            else: 
                label = str(state)
            #self.nodes[state]["label"] = label
        
        if color is None:
            color = self._const.COLOR_NODE_DEFAULT
        
        # no error if the node already exists.Attributes get updated if you pass them again.
        self.add_node(state, color=color, label=label)
        
    def setStateProperties(self, state, color=None, label=None, **kwargs):
        """
        Update visual or metadata properties for an existing state (node) in the graph.
    
        This method updates the node's attributes such as color, label, and any additional
        custom key-value pairs. The node must already exist in the graph.
    
        Parameters
        ----------
        - state : hashable
            Identifier of the state to update. Must be a node already present in the graph.
    
        - color : str, optional
            Color assigned to the node, typically used for visualization. If not provided,
            the default node color (`self._const.COLOR_NODE_DEFAULT`) is used.
    
        - label : str, optional
            Label for the node. If not provided, it is auto-generated from the state ID
            (especially for tuples).
    
        - **kwargs : dict
            Arbitrary additional key-value pairs to set as node attributes. These can include
            any custom metadata.
    
        Raises
        ------
        KeyError
            If the specified state is not present in the graph.
    
        Notes
        -----
        - This method ensures that required attributes (like `color` and `label`) are set for visualization. 
        - Existing node attributes are updated or extended with `kwargs`.
        """        
        if state not in self:
            raise KeyError(f"State '{state}' does not exist in the graph.")
        
        # mandatory for plotting
        self.addState(state, color, label)
        
        # # Add any additional attributes
        self.nodes[state].update(kwargs)
        
    def setAllStateDefaultProperties(self, **kwargs):
        """
        Set default visual and metadata properties for all states (nodes) in the graph.
    
        This method applies `setStateProperties()` to every node, ensuring each state
        has at least the required default attributes (e.g., color and label).
        Any additional keyword arguments are passed to `setStateProperties()`.
    
        Parameters
        ----------
        **kwargs : dict, optional
            Optional keyword arguments to apply uniformly to all nodes,
            such as color='gray', group='core', etc.
    
        Notes
        -----
        - This method is useful for initializing node attributes before plotting
          or performing other graph-based computations.
        - Nodes without existing attributes will receive defaults via `addState()`.
        """
        for state in self.nodes():
            self.setStateProperties(state, **kwargs)

        
    #def addTransition(self, origin_state, list_destination_state, sym_rate, color=plt.cm.tab10(5), tid = None):        
    def addTransition(self, origin_state, destination_state, sym_rate, tid = None, color=None, label=None):     
        """
        Add a directed transition (edge) between two states with a symbolic transition rate.
    
        This method adds a directed edge from `origin_state` to `destination_state`,
        stores both the symbolic and numerical rate, and sets optional display properties
        such as color, label, and transition ID. If the edge already exists, a warning is issued.
    
        Parameters
        ----------
        - origin_state : hashable
            The source node (state) of the transition.
    
        - destination_state : hashable
            The target node (state) of the transition.
    
        - sym_rate : sympy.Expr
            A symbolic expression representing the transition rate.
            This will be numerically evaluated using internal substitutions (`self._subs`).
    
        - tid : str or int, optional
            A transition identifier (e.g. for grouping or styling).
            If provided, the color will default to `self._const.COLOR_TRANSITIONS[tid]` if not set explicitly.
    
        - color : str, optional
            Color for the edge (used in visualization). Defaults to the transition-specific color
            if `tid` is provided.
    
        - label : str, optional
            LaTeX-formatted string for labeling the edge. If not provided, a default label is
            generated using `sympy.latex(sym_rate)`.
    
        Notes
        -----
        - The numeric rate is computed by substituting known parameter values into `sym_rate`.
        - The following attributes are stored on the edge:
            - `rate` (float): numeric value of the transition rate
            - `sym_rate` (sympy.Expr): original symbolic expression
            - `label` (str): display label
            - `color` (str): edge color
            - `tid` (str or int): transition ID
        - If the edge already exists, a `UserWarning` is issued but the edge is still overwritten.
        """        
        if tid is not None:
            if color is None:
                color = self._const.COLOR_TRANSITIONS[ tid ]
                
        rate = sym_rate.subs( self._subs )
        
        if label is None: string = "$"+sp.latex(sym_rate)+"$"
                
        if self.has_edge(origin_state, destination_state):
            theedge = self[origin_state][destination_state]
            warningstr = f"Edge already exists: {origin_state} -> {destination_state} with rate {theedge['label']}"
            warnings.warn(warningstr, category=UserWarning)
        self.add_edge(origin_state, destination_state, rate=float(rate), label=string, 
                   color=color, sym_rate=sym_rate, tid=tid)
    
    def sym(self, key):
        """
        Retrieve the symbolic representation of a model parameter.
    
        Parameters
        ----------
        key : str
            The name of the parameter (e.g., 'lambda', 'mu', 'n').
    
        Returns
        -------
        sympy.Symbol
            The symbolic variable corresponding to the given key.
    
        Raises
        ------
        KeyError
            If the key does not exist in the internal symbolic mapping (`_sym`).
    
        Examples
        --------
        >>> G.sym('lambda')
        lambda
    
        Notes
        -----
        - The symbolic mapping is initialized during object construction based on the `parameters` dictionary.
        - To access the symbol using dot-notation, use `.sym_key` attributes created during initialization.
        """
        return self._sym[key]

        
    @property
    def parameters(self):
        """
        Return the model parameters as a dictionary.
    
        This includes the original numerical values provided during initialization.
    
        Returns
        -------
        dict
            Dictionary of parameter names and their corresponding numeric values.
    
        Examples
        --------
        >>> G.parameters
        {'lambda': 1.0, 'mu': 0.5, 'n': 3}
    
        Notes
        -----
        - This property does not return symbolic values. Use `self.sym(key)` for symbolic access.
        - Parameter symbols can also be accessed using attributes like `sym_lambda` or the `sym()` method.
        """
        return self._parameters

    

    @property
    def rate_matrix(self):
        """
        Return the continuous-time rate matrix (Q matrix) of the state transition graph.
    
        This matrix defines the transition rates between all pairs of states in the
        continuous-time Markov chain (CTMC) represented by this graph.
    
        Returns
        -------
        numpy.ndarray or None
            The rate matrix `Q`, where `Q[i, j]` is the transition rate from state `i` to `j`.
            Returns `None` if the rate matrix has not been computed yet.
    
        Notes
        -----
        - The matrix is typically generated via a method like `computeRateMatrix()` or similar.
        - Internal state indexing is handled by `_n2i` (node-to-index mapping).
        - The rate matrix is stored internally as `_rate_matrix`.
        """
        return self._rate_matrix



    @property
    def state_probabilities(self):
        """
        Return the steady-state probabilities of the continuous-time Markov chain (CTMC).
    
        These probabilities represent the long-run proportion of time the system spends
        in each state, assuming the CTMC is irreducible and positive recurrent.
    
        Returns
        -------
        dict or None
            A dictionary mapping each state to its steady-state probability.
            Returns `None` if the probabilities have not yet been computed.
    
        Notes
        -----
        - The probabilities are stored internally after being computed via a dedicated method
          `calcSteadyStateProbs()`.
        - The result is stored in the `_state_probabilities` attribute.
        """
        return self._state_probabilities

     
    
    def printEdges(self, state):
        """
        Print all outgoing edges from a given state in the graph.
    
        For each edge from the specified `state`, this method prints the destination
        state and the corresponding transition label (typically the symbolic rate).
    
        Parameters
        ----------
        state : hashable
            The source node (state) from which outgoing edges will be printed.            
    
        Returns
        -------
        None
            This method prints output directly to the console and does not return a value.
    
        Notes
        -----
        - The transition label is retrieved from the edge attribute `'label'`.
        - Assumes `state` exists in the graph; otherwise, a KeyError will be raised.
        """
        print(f"from {state}")
        node = self[state]
        for e in node:
            rate = node[e]['label']
            print(f" -> {e} with {rate}")

            
    def createRateMatrix(self):
        """
        Construct the continuous-time Markov chain (CTMC) rate matrix Q from the graph.
    
        This method generates the generator matrix `Q` corresponding to the structure and
        rates defined in the directed graph. Each node represents a system state, and each
        directed edge must have a `'rate'` attribute representing the transition rate from
        one state to another.
    
        The matrix `Q` is constructed such that:
        - `Q[i, j]` is the transition rate from state `i` to state `j`
        - Diagonal entries satisfy `Q[i, i] = -sum(Q[i, j]) for j ≠ i`, so that each row sums to 0
    
        The resulting matrix and the internal node-to-index mapping (`_n2i`) are stored
        as attributes of the graph.
    
        Returns
        -------
        None
            The method updates internal attributes:
            - `_rate_matrix` : numpy.ndarray
                The CTMC rate matrix Q
            - `_n2i` : dict
                Mapping from node identifiers to matrix row/column indices
    
        Notes
        -----
        - Assumes all edges in the graph have a `'rate'` attribute.
        - The node order used in the matrix is consistent with the order from `list(self.nodes)`.
        - Use `self.rate_matrix` to access the generated matrix after this method is called.
        """
        n2i = {}
        nodes = list(self.nodes)
        for i,node in enumerate(nodes):
            n2i[node] = i         
        Q = np.zeros((len(nodes),len(nodes))) # rate matrix
        
        for edge in self.edges:
            i0 = n2i[edge[0]]
            i1 = n2i[edge[1]]
            Q[i0,i1] = self[edge[0]][edge[1]]["rate"] 
            
        np.fill_diagonal(Q, -Q.sum(axis=1))
        self._rate_matrix = Q
        self._n2i = n2i
        #return Q, n2i
    
    def calcSteadyStateProbs(self, verbose=False):        
        """
        Compute the steady-state probabilities of the continuous-time Markov chain (CTMC).
    
        This method calculates the stationary distribution of the CTMC defined by the rate
        matrix of the state transition graph. The computation solves the linear system
        `XQ = 0` with the constraint that the probabilities X sum to 1, using a modified
        version of the rate matrix `Q`.
    
        Parameters
        ----------
        verbose : bool, optional
            If True, prints the rate matrix, the modified matrix, the right-hand side vector,
            and the resulting steady-state probabilities.
    
        Returns
        -------
        probs : dict
            A dictionary mapping each state (node in the graph) to its steady-state probability.
    
        Notes
        -----
        - The graph must already contain valid transition rates on its edges.
        - If the rate matrix has not yet been created, `createRateMatrix()` is called automatically.
        - The method modifies the last column of the rate matrix to impose the normalization constraint
          `sum(X) = 1`.
        - The inverse of the modified matrix is computed directly, so the graph should be small enough
          to avoid numerical instability or performance issues.
        - The resulting probabilities are stored internally in `_state_probabilities` and can be accessed
          via the `state_probabilities` property.
        """
        
        # compute transition matrix if not already done
        if self._rate_matrix is None:  self.createRateMatrix()
        Q2 =  self._rate_matrix.copy()
        if verbose:
            print("Rate matrix Q")
            print(f'Q=\n{Q2}\n')
        
        Q2[:, -1] = 1
        if verbose:        
            print(f'Matrix is changed to\nQ2=\n{Q2}\n')

        b = np.zeros(len(Q2))
        b[-1] = 1
        if verbose:
            print("Solve X = b @ Q2")
            print(f'b=\n{b}\n')
        
        # state probabilities
        X = b @ linalg.inv(Q2) # compute the matrix inverse
            
        # Generate a matrix with P(X,Y)
        matrix = {}
        for n in self.nodes:
            matrix[n] = X[ self._n2i[n] ]
        
        if verbose:
            print("Steady-state probabilities X")
            print(f'X=\n{matrix}\n')
        self._state_probabilities = matrix
        return matrix
        
    def calcTransitionProbabilities(self):
        """
        Compute and store transition probabilities for each node in the graph of the embedded discrete Markov chain.
    
        This method processes each node in the graph, treating outgoing edges as
        transitions in an embedded discrete-time Markov chain derived from a continuous-time
        Markov chain (CTMC). Transition *rates* on outgoing edges are normalized into
        *probabilities*, and relevant data is stored in the node attributes.
    
        For each node, the following attributes are set:
        - `"transitionProbs"` : np.ndarray
            Array of transition probabilities to successor nodes.
        - `"transitionNodes"` : list
            List of successor node identifiers (ordered as in the probability array).
        - `"sumOutgoingRates"` : float
            Sum of all outgoing transition rates. Used to model the exponential waiting time
            in continuous-time simulation (`T ~ Exp(sumOutgoingRates)`).
    
        Notes
        -----
        - The graph must be a directed graph where each edge has a `'rate'` attribute.
        - Nodes with no outgoing edges are considered absorbing states.
        - Useful for discrete-event simulation and Markov chain Monte Carlo (MCMC) modeling.
        - Sets an internal flag `_transitionProbabilitesComputed = True` upon completion.
    
        Raises
        ------
        KeyError
            If any outgoing edge lacks a `'rate'` attribute.
        """
        for node in self:
            successors = list(self.successors(node))
            rates = np.array([self[node][nbr]['rate'] for nbr in successors])
            probs = rates / rates.sum()  # Normalize to probabilities
            self.nodes[node]["transitionProbs"] = probs # probabilities
            self.nodes[node]["transitionNodes"] = successors # list of nodes
            self.nodes[node]["sumOutgoingRates"] = rates.sum() # time in state ~ EXP(sumOutgoingRates)
            numel = len(successors)
            if numel==0: 
                print(f"Absorbing state: {node}")
        self._transitionProbabilitesComputed = True
                
    def simulateMarkovModel(self, startNode=None, numEvents=10):
        """
        Simulate a trajectory through the state transition graph as a continuous-time Markov chain (CTMC).
        
        This method performs a discrete-event simulation of the CTMC by generating a random walk
        through the graph, starting at `startNode` (or a default node if not specified). Transitions
        are chosen based on the normalized transition probabilities computed from edge rates, and
        dwell times are sampled from exponential distributions based on the total outgoing rate
        from each state.
        
        Parameters
        ----------
        - startNode : hashable, optional
            The node at which the simulation starts. If not specified, the first node in the graph
            (based on insertion order) is used.
        
        - numEvents : int, optional
            The number of transition events (steps) to simulate. Default is 10.
        
        Returns
        -------
        states : list
            A list of visited states, representing the sequence of nodes traversed in the simulation.
        
        times : numpy.ndarray
            An array of dwell times in each state, sampled from exponential distributions
            with rate equal to the sum of outgoing transition rates from each state.
        
        Notes
        -----
        - This method automatically calls `calcTransitionProbabilities()` if transition probabilities
          have not yet been computed.
        - If an absorbing state (i.e., a node with no outgoing edges) is reached, the simulation stops early.
        - The cumulative time can be obtained by `np.cumsum(times)`.
        - State durations follow `Exp(sumOutgoingRates)` distributions per CTMC theory.
        """
        if not self._transitionProbabilitesComputed: self.calcTransitionProbabilities()
        
        # simulate random walk through Graph = Markov Chain
        if startNode is None:
            startNode = next(iter(self))
        

        states = [] # list of states
        times = np.zeros(numEvents) # time per state

        node = startNode
        for i in range(numEvents):
            outgoingRate = self.nodes[node]["sumOutgoingRates"]  # time in state ~ EXP(sumOutgoingRates)
            times[i] = random.expovariate(outgoingRate) # exponentially distributed time in state
            states.append(node)
            
            # get next node    
            probs = self.nodes[node]["transitionProbs"] # probabilities
            successors = self.nodes[node]["transitionNodes"] # list of nodes        
            numel = len(successors)
            if numel==0: 
                print(f"Absorbing state: {node}")
                break
            elif numel==1:
                nextNode = successors[0]
            else:
                nextNode = successors[ np.random.choice(numel, p=probs) ]
            # edge = G[node][nextNode]
            node = nextNode
            
        return states, times
    
    def calcProbabilitiesSimulationRun(self, states, times):
        """
        Estimate empirical state probabilities from a simulation run of the CTMC.
    
        This method computes the fraction of total time spent in each state during a simulation.
        It uses the sequence of visited states and corresponding sojourn times to estimate the
        empirical (time-weighted) probability distribution over states.
    
        Parameters
        ----------
        - states : list of tuple
            A list of state identifiers visited during the simulation.    
        - times : numpy.ndarray
            An array of sojourn times corresponding to each visited state, same length as `states`.
    
        Returns
        -------
        simProbs : dict
            A dictionary mapping each unique state to its estimated empirical probability,
            based on total time spent in that state relative to the simulation duration.
    
        Notes
        -----
        - Internally, states are mapped to flat indices for efficient counting using NumPy.
        - This method assumes that `states` contains only valid node identifiers from the graph.
        - If `_n2i` (node-to-index mapping) has not been initialized, it will be computed here.
        - This is especially useful for validating analytical steady-state probabilities
          against sampled simulation results. Or in case of state-space explosions.
        """        
        if self._n2i is None:
            n2i = {}
            nodes = list(self.nodes)
            for i,node in enumerate(nodes):
                n2i[node] = i     
            self._n2i = n2i
                
        
        # Map tuples to flat indices
        data = np.array(states)
        max_vals = data.max(axis=0) + 1
        indices = np.ravel_multi_index(data.T, dims=max_vals)

        # Weighted bincount
        bincount = np.bincount(indices, weights=times)

        # Decode back to tuples
        tuple_indices = np.array(np.unravel_index(np.nonzero(bincount)[0], shape=max_vals)).T
        timePerState = bincount[bincount > 0]

        simProbs = {}
        totalTime = times.sum()
        for t, c in zip(tuple_indices, timePerState):
            simProbs[tuple(t)] = c / totalTime
            #print(f"Tuple {tuple(t)} → weighted count: {c}")
        return simProbs
    
    
    def _draw_networkx_edge_labels(
        self,
        pos,
        edge_labels=None, label_pos=0.75,
        font_size=10, font_color="k", font_family="sans-serif", font_weight="normal",
        alpha=None, bbox=None,
        horizontalalignment="center", verticalalignment="center",
        ax=None, rotate=True, clip_on=True, rad=0):
        """
        Draw edge labels for a directed graph with optional curved edge support.
    
        This method extends NetworkX's default edge label drawing by supporting
        curved (bended) edges through quadratic Bézier control points. It places
        labels dynamically at a specified position along the curve and optionally
        rotates them to follow the edge orientation.
    
        Parameters
        ----------
        pos : dict
            Dictionary mapping nodes to positions (2D coordinates). Required.
    
        edge_labels : dict, optional
            Dictionary mapping edge tuples `(u, v)` to label strings. If None,
            edge data is used and converted to strings automatically.
    
        label_pos : float, optional
            Position along the edge to place the label (0=head, 1=tail).
            Default is 0.75 (closer to the target node).
    
        font_size : int, optional
            Font size of the edge labels. Default is 10.
    
        font_color : str, optional
            Font color for the edge labels. Default is 'k' (black).
    
        font_family : str, optional
            Font family for the edge labels. Default is 'sans-serif'.
    
        font_weight : str, optional
            Font weight for the edge labels (e.g., 'normal', 'bold'). Default is 'normal'.
    
        alpha : float, optional
            Opacity for the edge labels. If None, labels are fully opaque.
    
        bbox : dict, optional
            Matplotlib-style bbox dictionary to style the label background.
            Default is a white rounded box.
    
        horizontalalignment : str, optional
            Horizontal alignment of the text. Default is 'center'.
    
        verticalalignment : str, optional
            Vertical alignment of the text. Default is 'center'.
    
        ax : matplotlib.axes.Axes, optional
            Axes on which to draw the labels. If None, uses the current axes.
    
        rotate : bool, optional
            Whether to rotate the label to match the edge direction. Default is True.
    
        clip_on : bool, optional
            Whether to clip labels at the plot boundary. Default is True.
    
        rad : float, optional
            Curvature of the edge (used to determine Bézier control point).
            Positive values bend counterclockwise; negative values clockwise. Default is 0 (straight).
    
        Returns
        -------
        dict
            Dictionary mapping edge tuples `(u, v)` to the corresponding matplotlib Text objects.
    
        Notes
        -----
        - If `rotate=True`, label text is automatically aligned with the direction of the edge.
        - Curvature (`rad`) enables visualization of bidirectional transitions using bent edges.
        - This method is intended for internal use and supports bended edge label placement to match custom edge rendering.
    
        See Also
        --------
        draw
        draw_networkx
        draw_networkx_nodes
        draw_networkx_edges
        draw_networkx_labels
        """                
        if ax is None:
            ax = plt.gca()
        if edge_labels is None:
            labels = {(u, v): d for u, v, d in self.edges(data=True)}
        else:
            labels = edge_labels
        text_items = {}
        for (n1, n2), label in labels.items():
            (x1, y1) = pos[n1]
            (x2, y2) = pos[n2]
            (x, y) = (
                x1 * label_pos + x2 * (1.0 - label_pos),
                y1 * label_pos + y2 * (1.0 - label_pos),
            )
            pos_1 = ax.transData.transform(np.array(pos[n1]))
            pos_2 = ax.transData.transform(np.array(pos[n2]))
            #linear_mid = 0.5*pos_1 + 0.5*pos_2
            linear_mid = (1-label_pos)*pos_1 + label_pos*pos_2
            d_pos = pos_2 - pos_1
            rotation_matrix = np.array([(0,1), (-1,0)])
            ctrl_1 = linear_mid + rad*rotation_matrix@d_pos
            ctrl_mid_1 = 0.5*pos_1 + 0.5*ctrl_1
            ctrl_mid_2 = 0.5*pos_2 + 0.5*ctrl_1
            bezier_mid = 0.5*ctrl_mid_1 + 0.5*ctrl_mid_2
            (x, y) = ax.transData.inverted().transform(bezier_mid)

            if rotate:
                # in degrees
                angle = np.arctan2(y2 - y1, x2 - x1) / (2.0 * np.pi) * 360
                # make label orientation "right-side-up"
                if angle > 90:
                    angle -= 180
                if angle < -90:
                    angle += 180
                # transform data coordinate angle to screen coordinate angle
                xy = np.array((x, y))
                trans_angle = ax.transData.transform_angles(
                    np.array((angle,)), xy.reshape((1, 2))
                )[0]
            else:
                trans_angle = 0.0
            # use default box of white with white border
            if bbox is None:
                bbox = dict(boxstyle="round", ec=(1.0, 1.0, 1.0), fc=(1.0, 1.0, 1.0))
            if not isinstance(label, str):
                label = str(label)  # this makes "1" and 1 labeled the same

            t = ax.text(
                x, y, label,
                size=font_size, color=font_color, family=font_family, weight=font_weight,
                alpha=alpha,
                horizontalalignment=horizontalalignment, verticalalignment=verticalalignment,
                rotation=trans_angle, transform=ax.transData,
                bbox=bbox, zorder=1,clip_on=clip_on)
            text_items[(n1, n2)] = t

        ax.tick_params(axis="both", which="both",bottom=False, left=False, labelbottom=False, labelleft=False)

        return text_items    

    
    def drawTransitionGraph(self, pos=None, 
                        bended=False, node_size=1000, num=1, rad=-0.2,
                        edge_labels=None, node_shapes='o', figsize=(14, 7),
                        clear=True, fontsize=10,
                        fontcolor='black', label_pos=0.75):        
        """
        Visualize the state transition graph with nodes and directed edges.
    
        This method draws the graph using `matplotlib` and `networkx`, including labels,
        node colors, and optional curved (bended) edges for better visualization of
        bidirectional transitions. It supports layout customization and is useful for
        understanding the structure of the Markov model.
    
        Parameters
        ----------
        pos : dict, optional
            A dictionary mapping nodes to positions. If None, a Kamada-Kawai layout is used.
    
        bended : bool, optional
            If True, edges are drawn with curvature using arcs. Useful for distinguishing
            bidirectional transitions. Default is False.
    
        node_size : int, optional
            Size of the nodes in the plot. Default is 1000.
    
        num : int, optional
            Figure number for matplotlib (useful for managing multiple figures). Default is 1.
    
        rad : float, optional
            The curvature radius for bended edges. Only used if `bended=True`. Default is -0.2.
    
        edge_labels : dict, optional
            Optional dictionary mapping edges `(u, v)` to labels. If None, the `'label'`
            attribute from each edge is used.
    
        node_shapes : str, optional
            Shape of the nodes (e.g., `'o'` for circle, `'s'` for square). Default is `'o'`.
    
        figsize : tuple, optional
            Size of the matplotlib figure. Default is (14, 7).
    
        clear : bool, optional
            If True, clears the figure before plotting. Default is True.
    
        fontsize : int, optional
            Font size used for node labels. Default is 10.
    
        fontcolor : str, optional
            Font color for node labels. Default is `'black'`.
    
        label_pos : float, optional
            Position of edge labels along the edge (0=start, 1=end). Default is 0.75.
    
        Returns
        -------
        fig : matplotlib.figure.Figure
            The created matplotlib figure object.
    
        ax : matplotlib.axes.Axes
            The axes object where the graph is drawn.
    
        pos : dict
            The dictionary of node positions used for drawing.
    
        Notes
        -----
        - Node labels and colors must be set beforehand using `addState()` or `setStateProperties()`.
        - Edge attributes must include `'label'` and `'color'`.
        - This method modifies the current matplotlib figure and is intended for interactive or inline use.
        """        
        node_labels = {node: data["label"] for node, data in self.nodes(data=True)}        
        node_colors = {node: data["color"] for node, data in self.nodes(data=True)}        
        node_colors = [data["color"] for node, data in self.nodes(data=True)]
        #node_shapes = {(hw,sw): "o" if hw+sw<G.N else "s" for hw,sw in G.nodes()}        
            
        edge_cols = {(u,v): self[u][v]["color"] for u,v in self.edges}
            
        if edge_labels is None:
            edge_labels = {(u,v): self[u][v]["label"] for u,v in self.edges}
            
        if pos is None:
            pos = nx.kamada_kawai_layout(self)    
            
        plt.figure(figsize=figsize, num=num, clear=clear)                        
        nx.draw_networkx_nodes(self, pos, node_color=node_colors, node_shape = node_shapes, node_size=node_size)                    
        
        nx.draw_networkx_labels(self, pos, labels=node_labels, font_size=fontsize, font_color=fontcolor)  # Draw node labels
        if bended: 
            nx.draw_networkx_edges(self, pos,  width=1, edge_color=[edge_cols[edge] for edge in self.edges], 
                                   node_size = node_size, 
                                   arrows = True, arrowstyle = '-|>',
                                   connectionstyle=f"arc3,rad={rad}")        
            self._draw_networkx_edge_labels(pos, ax=plt.gca(), edge_labels=edge_labels,rotate=False, rad = rad, label_pos=label_pos)
        else:
            nx.draw_networkx_edges(self, pos,  width=1, edge_color=[edge_cols[edge] for edge in self.edges], 
                               node_size = node_size, 
                               #min_target_margin=17, arrowsize=15,
                               arrows = True, arrowstyle = '-|>')                           
            nx.draw_networkx_edge_labels(self, pos, edge_labels, label_pos=label_pos) #, verticalalignment='center',)        
        plt.axis('off');    
        
        return plt.gcf(), plt.gca(), pos
    
    def animateSimulation(self, expectedTimePerState=1, inNotebook=False, **kwargs):
        """
        Animate a simulation trajectory through the state transition graph.
        The animation of the CTMC simulation run either in a Jupyter notebook or as a regular script.
    
        This method visualizes the path of a simulated or predefined sequence of states by
        temporarily highlighting each visited state over time. The time spent in each state
        is scaled relative to the average sojourn time to produce a visually smooth animation.
    
        Parameters
        ----------
        - `expectedTimePerState` : float, optional
            Approximate duration (in seconds) for an average state visit in the animation.
            The actual pause duration for each state is scaled proportionally to its sojourn time.
            Default is 1 second.
            
        - `inNotebook` : bool, optional
            If True, uses Jupyter-compatible animation (with display + clear_output).
            If False, uses standard matplotlib interactive animation. Default is True.
    
        - `**kwargs` : dict
            Additional keyword arguments including:
                - states : list
                    A list of visited states (nodes) to animate.
                - times : list or np.ndarray
                    Sojourn times in each state (same length as `states`).
                - startNode : hashable
                    Optional start node for automatic simulation if `states` and `times` are not provided.
                - numEvents : int
                    Number of events (state transitions) to simulate.
                - Additional drawing-related kwargs are passed to `drawTransitionGraph()`.
    
        Raises
        ------
        ValueError
            If neither a `(states, times)` pair nor `(startNode, numEvents)` are provided.
    
        Returns
        -------
        None
            The animation is shown interactively using matplotlib but no data is returned.
    
        Notes
        -----
        - If `states` and `times` are not supplied, a simulation is run via `simulateMarkovModel()`.
        - The function uses matplotlib's interactive mode (`plt.ion()`) to animate transitions.
        - The average sojourn time across all states is used to normalize animation speed.
        - Each visited state is highlighted in red for its corresponding (scaled) dwell time.
        - Drawing arguments like layout, font size, or color can be customized via kwargs.
        """        
        # expTimePerState: in seconds
        if "states" in kwargs and "times" in kwargs:
            states = kwargs["states"]
            times = kwargs["times"]
        else:
            # Run default simulation if data not provided
            if "startNode" in kwargs and "numEvents" in kwargs:
                states, times = self.simulateMarkovModel(startNode=None, numEvents=10)
            else:
                raise ValueError("Must provide either ('states' and 'times') or ('startNode' and 'numEvents').")
        
        #avg_OutGoingRate = np.mean([self.nodes[n].get("sumOutgoingRates", 0) for n in self.nodes])
        avg_Time = np.mean([1.0/self.nodes[n].get("sumOutgoingRates", 0) for n in self.nodes])
        
            
        # Remove simulation-related keys before drawing
        draw_kwargs = {k: v for k, v in kwargs.items() if k not in {"states", "times", "startNode", "numEvents"}}
                
        fig, ax, pos = self.drawTransitionGraph(**draw_kwargs)
        plt.tight_layout()
        
        if not inNotebook:
            plt.ion()
                                
        for node, time in zip(states,times):
            if inNotebook:
                clear_output(wait=True)
                artist = nx.draw_networkx_nodes(self, pos, nodelist=[node], ax=ax,
                                                node_color=self._const.COLOR_NODE_ANIMATION_HIGHLIGHT, node_size=1000)
                display(fig)
            else:
                artist = nx.draw_networkx_nodes(self, pos, nodelist=[node],ax=ax,
                                       node_color=self._const.COLOR_NODE_ANIMATION_HIGHLIGHT, node_size=1000)
                plt.draw()                        
            plt.pause(time/avg_Time*expectedTimePerState)
            
            # Remove highlighted node    
            artist.remove()
        
        pass
    
        
    
    def states(self, data=False):
        """
        Return a view of the graph's states (nodes), optionally with attributes.
    
        This method is functionally identical to `self.nodes()` in NetworkX, but is
        renamed to `states()` to reflect the semantics of a state transition graph or
        Markov model, where nodes represent system states.
    
        Parameters
        ----------
        data : bool, optional
            If True, returns a view of `(node, attribute_dict)` pairs.
            If False (default), returns a view of node identifiers only.
    
        Returns
        -------
        networkx.classes.reportviews.NodeView or NodeDataView
            A view over the graph’s nodes or `(node, data)` pairs, depending on the `data` flag.
    
        Examples
        --------
        >>> G.states()
        ['A', 'B', 'C']
    
        >>> list(G.states(data=True))
        [('A', {'color': 'red', 'label': 'Start'}), ('B', {...}), ...]
    
        Notes
        -----
        - This is a convenience wrapper for semantic clarity in models where nodes represent states.
        - The view is dynamic: changes to the graph are reflected in the returned view.
        """
        return self.nodes(data=data)

    
    def setStateColor(self, state, color):
        """
        Set the color attribute of a specific state (node).
    
        Parameters
        ----------
        state : hashable
            The identifier of the state whose color is to be updated.
    
        color : str
            The new color to assign to the state (used for visualization).
    
        Returns
        -------
        None
        """
        self.nodes[state]["color"] = color
        
        
    def setStateLabel(self, state, label):
        """
        Set the label attribute of a specific state (node).
    
        Parameters
        ----------
        state : hashable
            The identifier of the state whose label is to be updated.
    
        label : str
            The new label to assign to the state (used for visualization or annotation).
    
        Returns
        -------
        None
        """
        self.nodes[state]["label"] = label

    def prob(self, state):
        """
        Return the steady-state probability of a specific state.
    
        Parameters
        ----------
        state : hashable
            The identifier of the state whose probability is to be retrieved.
    
        Returns
        -------
        float
            The steady-state probability of the specified state.
    
        Raises
        ------
        KeyError
            If the state is not found in the steady-state probability dictionary.
        """
        return self.state_probabilities[state]
        
    
    def probs(self):
        """
        Return the steady-state probabilities for all states.
    
        Returns
        -------
        dict
            A dictionary mapping each state to its steady-state probability.
    
        Notes
        -----
        - The probabilities are computed via `calcSteadyStateProbs()` and stored internally.
        - Use this method to access the full steady-state distribution.
        """
        return self.state_probabilities

    def getEmptySystemProbs(self, state=None):
        """
        Create an initial state distribution where all probability mass is in a single state.
    
        By default, this method returns a distribution where the entire probability mass
        is placed on the first node in the graph. Alternatively, a specific starting state
        can be provided.
    
        Parameters
        ----------
        state : hashable, optional
            The state to initialize with probability 1. If None, the first node
            in the graph (based on insertion order) is used.
    
        Returns
        -------
        dict
            A dictionary representing the initial distribution over states.
            The selected state has probability 1, and all others have 0.
    
        Examples
        --------
        >>> X0 = G.getEmptySystem()
        >>> sum(X0.values())  # 1.0
    
        Notes
        -----
        - This method is useful for setting up deterministic initial conditions
          in transient simulations of a CTMC.
        - The return format is compatible with methods like `transientProbs()`.
        """
        if state is None:
            first_node = next(iter(self.nodes))
        X0 = {s: 1 if s == first_node else 0 for s in self}
        return X0

        

    def transientProbs(self, initialDistribution, t):
        """
        Compute the transient state probabilities at time `t` for the CTMC.
    
        This method solves the forward equation of a continuous-time Markov chain (CTMC):
        X(t) = X0 · expm(Q·t), where `Q` is the generator (rate) matrix and `X0` is the
        initial state distribution. It returns a dictionary mapping each state to its
        probability at time `t`.
    
        Parameters
        ----------
        - `initialDistribution` : dict 
            A dictionary mapping states (nodes) to their initial probabilities at time `t=0`.
            The keys must match the nodes in the graph, and the values should sum to 1.    
            
        - `t` : float
            The time at which the transient distribution is to be evaluated.
    
        Returns
        -------
        dict
            A dictionary mapping each state (node) to its probability at time `t`.
    
        Notes
        -----
        - The rate matrix `Q` is generated automatically if not yet computed.
        - The state order in the rate matrix corresponds to the internal `_n2i` mapping.
        - Internally, the dictionary `initialDistribution` is converted into a vector
          aligned with the state indexing.
        - The transient solution is computed using `scipy.linalg.expm` for matrix exponentiation.
    
        Raises
        ------
        ValueError
            If the input dictionary has missing states or probabilities that do not sum to 1.
            (This is not enforced but recommended for correctness.)
        """        
        if self._rate_matrix is None:  self.createRateMatrix()
        Q = self._rate_matrix
        X0 = np.zeros(len(initialDistribution))
        
        for i,n in enumerate(self.nodes):
            X0[ self._n2i[n] ] = initialDistribution[n]
                
        
        X = X0 @ expm(Q*t)    
        
        matrix = {}
        for n in self.nodes:
            matrix[n] = X[ self._n2i[n] ]
                
        return matrix
        
    def symSolveMarkovModel(self):
        """
        Symbolically solve the global balance equations of the CTMC.
    
        This method constructs and solves the system of symbolic equations for the
        steady-state probabilities of each state in the CTMC. It uses symbolic transition
        rates stored on the edges to form balance equations for each node, along with
        the normalization constraint that all probabilities sum to 1.
    
        Returns
        -------
        solution : dict
            A symbolic solution dictionary mapping SymPy symbols (e.g., x_{A}) to
            expressions in terms of model parameters.
    
        num_solution : dict
            A numerical dictionary mapping each state (node) to its steady-state probability,
            computed by substituting the numeric values from `self._subs` into the symbolic solution.
    
        Notes
        -----
        - For each node `s`, a symbolic variable `x_{label}` is created using the node's label attribute.
        - One balance equation is created per state: total inflow = total outflow.
        - An additional constraint `sum(x_i) = 1` ensures proper normalization.
        - The symbolic system is solved with `sympy.solve`, and the results are simplified.
        - Numeric values are computed by substituting known parameter values (`self._subs`) into the symbolic solution.
    
        Examples
        --------
        >>> sym_sol, num_sol = G.symSolveMarkovModel()
        >>> sym_sol  # symbolic expressions for each state
        >>> num_sol  # evaluated numerical values for each state
        """
        X={}
        to_solve = []
        for s in self.nodes():
            X[s] = sp.Symbol(f'x_{{{self.nodes[s]["label"]}}}')
            to_solve.append(X[s])
        #%
        eqs = []
        for s in self.nodes():
             succ = self.successors(s) # raus
             out_rate = 0
             for z in list(succ):
                 #out_rate += X[s]*self.edges[s, z]["sym_rate"]                 
                 out_rate += X[s]*self[s][z]["sym_rate"]
             
             in_rate = 0
             for r in list(self.predecessors(s)):
                 #in_rate += X[r]*self.edges[r,s]["sym_rate"] 
                 in_rate += X[r]*self[r][s]["sym_rate"] 
             eqs.append( sp.Eq(out_rate, in_rate) )
        #% 
        Xsum = 0    
        for s in X:
            Xsum += X[s]
        
        eqs.append( sp.Eq(Xsum, 1) )    
        #%
        solution = sp.solve(eqs, to_solve)
        simplified_solution = sp.simplify(solution)
        #print(simplified_solution)
        num_solution = simplified_solution.subs(self._subs)
        #num_dict =  {str(var): str(sol) for var, sol in simplified_solution.items()}
        num_dict = {s: num_solution[X[s]] for s in X}                
        
        return simplified_solution, num_dict 
    
