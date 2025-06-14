"""Class for symbolic expression object or program."""

import array
# from msilib.schema import Error
import warnings
from textwrap import indent

import numpy as np
from sympy.parsing.sympy_parser import parse_expr
from sympy import pretty
import time

from .functions import PlaceholderConstant
from .const import make_const_optimizer,ScipyMinimize
from .utils import cached_property
# import kd.model.discover.utils as U
from . import utils as U
from .stridge import STRidge,Regulations,Node,build_tree,unsafe_execute_torch,unsafe_execute


def _finish_tokens(tokens):

    """
    Complete a possibly unfinished string of tokens.

    Parameters
    ----------
    tokens : list of integers
        A list of integers corresponding to tokens in the library. The list
        defines an expression's pre-order traversal.

    Returns
    _______
    tokens : list of ints
        A list of integers corresponding to tokens in the library. The list
        defines an expression's pre-order traversal. "Dangling" programs are
        completed with repeated "x1" until the expression completes.

    """

    n_objects = Program.n_objects
    # import pdb;pdb.set_trace()
    arities = np.array([Program.library.arities[t] for t in tokens])
    # Number of dangling nodes, returns the cumsum up to each point
    # Note that terminal nodes are -1 while functions will be >= 0 since arities - 1
    dangling = 1 + np.cumsum(arities - 1)

    if -n_objects in (dangling - 1):
        # Chop off tokens once the cumsum reaches 0, This is the last valid point in the tokens
        expr_length = 1 + np.argmax((dangling - 1) == -n_objects)
        tokens = tokens[:expr_length]
    else:
        # Extend with valid variables until string is valid
        # NOTE: This only appends onto the end of a set of tokens, even in the multi-object case!
        assert n_objects == 1, "Is max length constraint turned on? Max length constraint required when n_objects > 1."
        tokens = np.append(tokens, np.random.choice(Program.library.input_tokens, size=dangling[-1]))
        
    return tokens


def from_str_tokens(str_tokens, skip_cache=False):
    """
    Memoized function to generate a Program from a list of str and/or float.
    See from_tokens() for details.

    Parameters
    ----------
    str_tokens : str | list of (str | float)
        Either a comma-separated string of tokens and/or floats, or a list of
        str and/or floats.

    skip_cache : bool
        See from_tokens().

    Returns
    -------
    program : Program
        See from_tokens().
    """

    # Convert str to list of str
    if isinstance(str_tokens, str):
        str_tokens = str_tokens.split(",")

    # Convert list of str|float to list of tokens
    if isinstance(str_tokens, list):
        traversal = []
        constants = []
        for s in str_tokens:
            if s in Program.library.names:
                t = Program.library.names.index(s)
            elif U.is_float(s):
                assert "const" not in str_tokens, "Currently does not support both placeholder and hard-coded constants."
                t = Program.library.const_token
                constants.append(float(s))
            else:
                raise ValueError("Did not recognize token {}.".format(s))
            traversal.append(t)
        traversal = np.array(traversal, dtype=np.int32)
    else:
        raise ValueError("Input must be list or string.")

    # Generate base Program (with "const" for constants)
    p = from_tokens(traversal, skip_cache=skip_cache)

    # Replace any constants
    p.set_constants(constants)

    return p


def from_tokens(tokens, skip_cache=False, on_policy=True, finish_tokens=True):

    """
    Memoized function to generate a Program from a list of tokens.

    Since some tokens are nonfunctional, this first computes the corresponding
    traversal. If that traversal exists in the cache, the corresponding Program
    is returned. Otherwise, a new Program is returned.

    Parameters
    ----------
    tokens : list of integers
        A list of integers corresponding to tokens in the library. The list
        defines an expression's pre-order traversal. "Dangling" programs are
        completed with repeated "x1" until the expression completes.

    skip_cache : bool
        Whether to bypass the cache when creating the program (used for
        previously learned symbolic actions in DSP).
        
    finish_tokens: bool
        Do we need to finish this token. There are instances where we have
        already done this. Most likely you will want this to be True. 

    Returns
    _______
    program : Program
        The Program corresponding to the tokens, either pulled from memoization
        or generated from scratch.
    """

    '''
        Truncate expressions that complete early; extend ones that don't complete
    '''
  
    if finish_tokens:
        tokens = _finish_tokens(tokens)

    # For stochastic Tasks, there is no cache; always generate a new Program.
    # For deterministic Programs, if the Program is in the cache, return it;
    # otherwise, create a new one and add it to the cache.
    # import pdb;pdb.set_trace()
    if skip_cache or Program.task.stochastic:
        p = Program(tokens, on_policy=on_policy)
    else:
        key = tokens.tostring() 
        try:
            p = Program.cache[key]
            if on_policy:
                p.on_policy_count += 1
            else:
                p.off_policy_count += 1
        except KeyError:
            p = Program(tokens, on_policy=on_policy)
            Program.cache[key] = p

    return p


class Program(object):
    """
    The executable program representing the symbolic expression.

    The program comprises unary/binary operators, constant placeholders
    (to-be-optimized), input variables, and hard-coded constants.

    Parameters
    ----------
    tokens : list of integers
        A list of integers corresponding to tokens in the library. "Dangling"
        programs are completed with repeated "x1" until the expression
        completes.

    Attributes
    ----------
    traversal : list
        List of operators (type: Function) and terminals (type: int, float, or
        str ("const")) encoding the pre-order traversal of the expression tree.

    tokens : np.ndarry (dtype: int)
        Array of integers whose values correspond to indices

    const_pos : list of int
        A list of indicies of constant placeholders along the traversal.

    float_pos : list of float
        A list of indices of constants placeholders or floating-point constants
        along the traversal.

    sympy_expr : str
        The (lazily calculated) SymPy expression corresponding to the program.
        Used for pretty printing _only_.

    complexity : float
        The (lazily calcualted) complexity of the program.

    r : float
        The (lazily calculated) reward of the program.

    count : int
        The number of times this Program has been sampled.

    str : str
        String representation of tokens. Useful as unique identifier.
    """

    # Static variables
    task = None             # Task
    library = None          # Library
    const_optimizer = None  # Function to optimize constants
    cache = {}
    n_objects = 1          # Number of executable objects per Program instance

    default_terms=[]
    # Cython-related static variables
    have_cython = None      # Do we have cython installed
    execute = None          # Link to execute. Either cython or python
    cyfunc = None           # Link to cyfunc lib since we do an include inline

    function_terms = {} #cache the bst reward sample function terms
    reward_list = []
    def __init__(self, tokens=None, on_policy=True):
        """
        Builds the Program from a list of of integers corresponding to Tokens.
        """
        
        # Can be empty if we are unpickling 
        if tokens is not None:
            self._init(tokens, on_policy)
    
    def _init(self, tokens, on_policy=True):

        self.traversal = [Program.library[t] for t in tokens]
        self.default_terms = [[Program.library[t]] for term in Program.default_terms for t in term]
        # import pdb;pdb.set_trace()
        self.set_stridge()
        
        self.const_pos = [i for i, t in enumerate(self.traversal) if isinstance(t, PlaceholderConstant)]
        self.len_traversal = len(self.traversal)

        if self.have_cython and self.len_traversal > 1:
            self.is_input_var = array.array('i', [t.input_var is not None for t in self.traversal])

        self.invalid = False
        self.str = tokens.tostring()
        self.tokens = tokens

        self.on_policy_count = 1 if on_policy else 0
        self.off_policy_count = 0 if on_policy else 1
        self.originally_on_policy = on_policy # Note if a program was created on policy
        if Program.n_objects > 1:
                # Fill list of multi-traversals
            danglings = -1 * np.arange(1, Program.n_objects + 1)
            self.traversals = [] # list to keep track of each multi-traversal
            i_prev = 0
            arity_list = [] # list of arities for each node in the overall traversal
            for i, token in enumerate(self.traversal):
                arities = token.arity
                arity_list.append(arities)
                dangling = 1 + np.cumsum(np.array(arity_list) - 1)[-1]
                if (dangling - 1) in danglings:
                    trav_object = self.traversal[i_prev:i+1]
                    self.traversals.append(trav_object)
                    i_prev = i+1
                    """
                    Keep only what dangling values have not yet been calculated. Don't want dangling to go down and up (e.g hits -1, goes back up to 0 before hitting -2)
                    and trigger the end of a traversal at the wrong time
                    """
                    danglings = danglings[danglings != dangling - 1] 
    
    def set_stridge(self):
        self.STRidge = STRidge(self.traversal.copy(),self.default_terms,
                            noise_level=Program.task.noise_level, 
                            max_depth=Program.task.max_depth,
                            cut_ratio=Program.task.cut_ratio,
                            spatial_error=Program.task.spatial_error,
                            const =Program.task.add_const)    
                    
    def __getstate__(self):
        
        have_r = "r" in self.__dict__
        have_evaluate = "evaluate" in self.__dict__
        possible_const = have_r or have_evaluate
        
        state_dict = {'tokens' : self.tokens, # string rep comes out different if we cast to array, so we can get cache misses.
                      'have_r' : bool(have_r),
                      'r' : float(self.r) if have_r else float(-np.inf), 
                      'have_evaluate' : bool(have_evaluate),
                      'evaluate' : self.evaluate if have_evaluate else float(-np.inf), 
                      'const' : array.array('d', self.get_constants()) if possible_const else float(-np.inf), 
                      'on_policy_count' : bool(self.on_policy_count),
                      'off_policy_count' : bool(self.off_policy_count),
                      'originally_on_policy' : bool(self.originally_on_policy),
                      'invalid' : bool(self.invalid), 
                      'error_node' : array.array('u', "" if not self.invalid else self.error_node), 
                      'error_type' : array.array('u', "" if not self.invalid else self.error_type)}    
        
        # In the future we might also return sympy_expr and complexity if we ever need to compute in parallel 

        return state_dict
                
    def __setstate__(self, state_dict):
        
        # Question, do we need to init everything when we have already run, or just some things?
        self._init(state_dict['tokens'], state_dict['originally_on_policy'])
        
        have_run = False
        
        if state_dict['have_r']:
            setattr(self, 'r', state_dict['r'])
            have_run = True
            
        if state_dict['have_evaluate']:
            setattr(self, 'evaluate', state_dict['evaluate'])
            have_run = True 
        
        if have_run:
            self.set_constants(state_dict['const'].tolist())
            self.invalid = state_dict['invalid']
            self.error_node = state_dict['error_node'].tounicode()
            self.error_type = state_dict['error_type'].tounicode()
            self.on_policy_count = state_dict['on_policy_count']
            self.off_policy_count = state_dict['off_policy_count']

    def execute_STR(self,u, x,ut,  test=False, wf=False):
        if wf:
            y_hat, w_best, self.invalid,self.error_node,self.error_type, y_right= self.STRidge.wf_calculate(u,x,ut, \
            test, Program.execute_function)
        else:
            y_hat, w_best, self.invalid,self.error_node,self.error_type, y_right= self.STRidge.calculate(u,x,ut, \
                test, Program.execute_function)

        return y_hat, y_right, w_best
    
    def execute_terms(self,u,x):
        results =  self.STRidge.evaluate_terms(u,x, execute_function =Program.execute_function )
        #  todo return  n*d
        
        return results
        
    def execute_multi_STR(self,u,x,ut, test = False):
        self.cached_terms, cached_vals = None, None  
        y_hat, w_best, self.invalid,self.error_node,self.error_type, y_right=  self.STRidge.calculate(u,\
            x,ut, test, Program.execute_function, cached = (self.cached_terms, cached_vals))
        
        return y_hat, y_right, w_best
    
    def sample_cached_terms(self):
        
        self.cached_terms, cached_vals = self.task.sample_cached_terms()
        return cached_vals
    
    def execute_BFGS(self,u, x, ut,):
        # default coef for every terms is 1
        # self.set_stridge()
        # import pdb;pdb.set_trace()
        y_hat, w_best, self.invalid,self.error_node,self.error_type, y_right = self.STRidge.evaluate(self.traversal,\
                                                                                                      u,x,ut, Program.execute_function )
        return y_hat, y_right,w_best
    
    def execute(self, u,x,ut,wf = False,):
        if isinstance(Program.const_optimizer, ScipyMinimize):
            self.cached_terms=None
            return self.execute_BFGS(u,x,ut)

        else:
            if isinstance(ut, list):
                # multi-dataset for subgrid force or multi state variables
                return self.execute_multi_STR(u,x,ut) 
            else:
                # single stridge
                self.cached_terms=None
                return self.execute_STR(u, x,ut,wf = wf)
                
    def execute_test(self, u, x, ut):
        results = self.STRidge.calculate_RHS_terms(u,x, Program.execute_function)
        return results
    
    def execute_stability_test(self):     
        return self.task.stability_test(self)
    
    def switch_tokens(self, token_type='torch'):
        if token_type == 'torch':
            torch_tokens = Program.library.torch_tokens # dicts key == numpy_op+t, value = torch tokens
            if len(torch_tokens)==0:
                # print("AD Utilized")
                return
            np_tokens_name = Program.library.np_names
            if "_t" in np_tokens_name[0]:
                # use ad to generate meta data 
                return 
            new_traversal = []
            for t in self.traversal:
                if t.name in np_tokens_name:
                    new_traversal.append(torch_tokens[t.name+'_t'])
                else:
                    new_traversal.append(t)
        else:
            assert token_type == 'subgrid'
            subgrid_tokens = Program.library.subgrid_tokens # dicts key == numpy_op+t, value = torch tokens
            if len(subgrid_tokens)==0:
                return
            new_traversal = []
            for t in self.traversal:
                if t.name in subgrid_tokens:
                    # import pdb;pdb.set_trace()
                    new_traversal.append(subgrid_tokens[t.name])
                else:
                    new_traversal.append(t)
            
        self.traversal = new_traversal
        self.set_stridge()
                
    @cached_property
    def terms_values(self):
        #return term tokens and their values
        tokens, values = self.task.terms_values(self)
        return  tokens, values
        
    @cached_property
    def mse(self):
        return self.task.mse
    
    @cached_property
    def cv(self):
        return self.task.cv
    
    @cached_property
    def evaluate(self):
        """Evaluates and returns the evaluation metrics of the program."""

        # Program must be optimized before computing evaluate
        if "r_ridge" not in self.__dict__:
            print("WARNING: Evaluating Program before computing its reward." \
                  "Program will be optimized first.")
            self.optimize_inner_constant()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            return self.task.evaluate(self)

    def optimize_hardcoded_constant(self):
        pass

    def optimize_inner_constant(self):
        """
        Optimizes PlaceholderConstant tokens against the reward function. The
        optimized values are stored in the traversal.
        """
        # original settings: pre sum up the consts 
        if len(self.const_pos) == 0:
            return
        
        # Define the objective function: negative reward
        def f(consts):
            self.set_constants(consts)
            # import pdb;pdb.set_trace()
            r = self.task.mse_function(self)
            # Constant optimizer minimizes the objective function

            # Need to reset to False so that a single invalid call during
            # constant optimization doesn't render the whole Program invalid.
            self.invalid = False

            return r

        # Do the optimization
        x0 = np.ones(len(self.const_pos)) # Initial guess
        optimized_constants = Program.const_optimizer(f, x0)

        # Set the optimized constants
        self.set_constants(optimized_constants)
        # reset time
        # Program.const_optimizer.reset()
        

    def get_constants(self):
        """Returns the values of a Program's constants."""

        return [t.value for t in self.traversal if isinstance(t, PlaceholderConstant)]

    def set_constants(self, consts):
        """Sets the program's constants to the given values"""

        for i, const in enumerate(consts):
            assert U.is_float, "Input to program constants must be of a floating point type"
            # Create a new instance of PlaceholderConstant instead of changing
            # the "values" attribute, otherwise all Programs will have the same
            # instance and just overwrite each other's value.
            self.traversal[self.const_pos[i]] = PlaceholderConstant(const)


    @classmethod
    def set_n_objects(cls, n_objects):
        Program.n_objects = n_objects

    @classmethod
    def clear_cache(cls):
        """Clears the class' cache"""

        cls.cache = {}
        STRidge.clear_cache()


    @classmethod
    def set_task(cls, task):
        """Sets the class' Task"""

        Program.task = task
        Program.library = task.library
        
    @classmethod
    def reset_task(cls, model, generate_type):
        
        '''
        generate_type = AD or FD
        '''
        Program.task.generate_meta_data(model,generate_type)  
         
    @classmethod
    def set_default_terms(cls, default_terms):
        Program.default_terms = default_terms

    @classmethod
    def set_const_optimizer(cls, name, **kwargs):
        """Sets the class' constant optimizer"""
        const_optimizer = make_const_optimizer(name, **kwargs)
        Program.const_optimizer = const_optimizer


    @classmethod
    def set_complexity(cls, name):
        """Sets the class' complexity function"""

        all_functions = {
            # No complexity
            None : lambda p : 0.0,

            # Length of sequence
            "length" : lambda p : len(p.traversal),

            # Sum of token-wise complexities
            "token" : lambda p : sum([t.complexity for t in p.traversal]),

        }

        assert name in all_functions, "Unrecognzied complexity function name."
        Program.complexity_function = lambda p : all_functions[name](p)

    @classmethod
    def set_execute(cls, protected,use_torch):
        """Sets which execute method to use"""

        from discover.execute import python_execute, python_execute_torch
        execute_function        = python_execute
        Program.have_cython     = False
        Program.use_torch = use_torch
        
        Program.execute_function = unsafe_execute if not use_torch else unsafe_execute_torch 
        Program.protected = False

    @cached_property
    def r_ridge(self):
        """Evaluates and returns the reward of the program"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # warnings.filterwarnings('ignore', 'Intel MKL ERROR')
            self.optimize_inner_constant()
            # 调用我们修改后的 reward_function，并接收所有4个返回值
            result, w, y_hat, y_right = self.task.reward_function(self)
            self.w = w
            self.y_hat_rhs = y_hat   # 存储预测的RHS
            self.Theta = y_right # 存储Theta矩阵
            return result
                
    @cached_property
    def complexity(self):
        """Evaluates and returns the complexity of the program"""

        return Program.complexity_function(self)



    @cached_property
    def sympy_expr(self):
        """
        Returns the attribute self.sympy_expr.

        This is actually a bit complicated because we have to go: traversal -->
        tree --> serialized tree --> SymPy expression
        """

        if Program.n_objects == 1:
            tree = self.traversal.copy()
            tree = build_tree(tree)
            tree = convert_to_sympy(tree)
            try:
                expr = parse_expr(tree.__repr__()) # SymPy expression
            except:
                expr = tree.__repr__()
            return [expr]
        else:
            exprs = []
            for i in range(len(self.traversals)):
                tree = self.traversals[i].copy()
                tree = build_tree(tree)
                tree = convert_to_sympy(tree)
                try:
                    expr = parse_expr(tree.__repr__()) # SymPy expression
                except:
                    expr = tree.__repr__()
                exprs.append(expr)
            return exprs

    def pretty(self):
        """Returns pretty printed string of the program"""
        return [pretty(self.sympy_expr[i]) for i in range(Program.n_objects)] 


    def print_stats(self):
        """Prints the statistics of the program
        
            We will print the most honest reward possible when using validation.
        """

        print("\tReward: {}".format(self.r_ridge))
        print("\tMSE:{}".format(self.evaluate['nmse_test']))
        print("\tTraversal: {}".format(self))

        print("\tExpression:")
        # print("{}\n".format(indent(self.pretty()[0], '\t  ')))
        print(self.str_expression)


    def __repr__(self):
        """Prints the program's traversal"""
        
        return ','.join([repr(t) for t in self.traversal])

    @property   
    def str_expression(self):
        out = ""
        # else:
        # if 'w_test' in self.evaluate:
        #     self.w = self.evaluate['w_test']
        for number, traversal in zip(self.w, self.STRidge.terms):
            # traversal_str =','.join([repr(t) for t in traversal])
            # import pdb;pdb.set_trace()
            out+="%.4f"%number +" * "+ repr(traversal) +' + '
        prefix = ""
        if self.cached_terms is not None:
            for traversal in self.cached_terms:
                prefix += repr(traversal) +' + '
        if  len(self.w)-len(self.STRidge.terms)==1:
            out+= "%.4f"%self.w[-1]+' + '
        return prefix + out[:-3]
    
    @property
    def funcion_expression(self):
        out = " |".join([repr(function ) for function  in self.STRidge.terms])
        return  out
    
    @property
    def coefficents(self):
        out = "|".join([str(round(w,4)) for w  in self.w])
        return  out



###############################################################################
# Everything below this line is currently only being used for pretty printing #
###############################################################################






def convert_to_sympy(node):
    """Adjusts trees to only use node values supported by sympy"""

    if node.val == "div":
        node.val = "Mul"
        new_right = Node("Pow")
        new_right.children.append(node.children[1])
        new_right.children.append(Node("-1"))
        node.children[1] = new_right

    elif node.val == "sub":
        node.val = "Add"
        new_right = Node("Mul")
        new_right.children.append(node.children[1])
        new_right.children.append(Node("-1"))
        node.children[1] = new_right

    elif node.val == "inv":
        node.val = Node("Pow")
        node.children.append(Node("-1"))

    elif node.val == "neg":
        node.val = Node("Mul")
        node.children.append(Node("-1"))

    elif node.val == "n2":
        node.val = "Pow"
        node.children.append(Node("2"))

    elif node.val == "n3":
        node.val = "Pow"
        node.children.append(Node("3"))

    elif node.val == "n4":
        node.val = "Pow"
        node.children.append(Node("4"))

    for child in node.children:
        convert_to_sympy(child)

    return node
