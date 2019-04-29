from spacepy import pycdf
import re
import numpy as np
import pdb
import matplotlib.pyplot as plt

CDF_VARS = {}
FILE_VARS = {}
SUPPORT_VARS = {}


def gattrnames(filename):
    """
    Return the names of all global attributes contained within a CDF file.
    
    Parameters
    ==========
    filename : str
        Name of the CDF file for which names are returned.
        
    Returns
    =======
    names : list of str
        Name of each global attribute within the CDF file.
    """
    with pycdf.CDF(filename) as f:
        names = [name for name in f.attrs]
    return names


def gattrvalue(filename, gattrname):
    """
    Return the value of a global attribute contained within a CDF file.
    
    Parameters
    ==========
    filename : str
        Name of a CDF file.
    gattrname : str
        Name of the global attribute within the CDF file for which value is desired.
        
    Returns
    =======
    value : list
        All global attribute entries associated with the global attribute. Not all
        entries necessarily have a value. In the case of no value, the list entry
        is None.
    """
    with pycdf.CDF(filename) as f:
        value = f.attrs[gattrname][:]
    return value


def varnames(filename):
    """
    Return the names of all variables contained within a CDF file.
    
    Parameters
    ==========
    filename : str
        Name of the CDF file for which names are returned.
        
    Returns
    =======
    names : list of str
        Name of each variable within the CDF file.
    """
    with pycdf.CDF(filename) as f:
        names = [name for name in f]
    return names


def vattrnames(filename, varname):
    """
    Return the names of all variable attributes associated with a
    given variable contained within a CDF file.
    
    Parameters
    ==========
    filename : str
        Name of a CDF file.
    varname : str
        Name of the variable within the CDF file for which variable
        attribute names are requested.
        
    Returns
    =======
    names : list of str
        Name of each variable attribute associated with the given variable.
    """
    with pycdf.CDF(filename) as f:
        names = [name for name in f[varname].attrs]
    return names


def vattrvalue(filename, varname, vattrname):
    """
    Return the value of a variable attribute contained within a CDF file.
    
    Parameters
    ==========
    filename : str
        Name of a CDF file.
    varname : str
        Name of a variable within the CDF.
    vattrname : str
        Name of the variable attribute for which the value is requested.
        
    Returns
    =======
    value : list
        All values associated with the variable attribute.
    """
    with pycdf.CDF(filename) as f:
        value = f[varname].attrs[vattrname]
    return value


def read(filenames, variables):
    """
    Read variables from a CDF file.
    
    Parameters
    ==========
    filenames : str, list
        Name of the CDF file(s) to be read.
    variables : str, list
        Name of the variable(s) to be read. A (list of) regular expression may
        be given. Any variables matching the regex will be read.
    
    Returns
    =======
    vars : tuple
        A tsdata objects for each variable.
    """
    global CDF_VARS
    global FILE_VARS
    global SUPPORT_VARS
    
    # Turn strings into lists
    if isinstance(filenames, str):
        filenames = [filenames]
    if isinstance(variables, str):
        variables = [variables]
    
    # Reset global variables
    _read_reset
    
    # Read variables from files
    for fname in filenames:
        FILE_VARS = {}
        
        with pycdf.CDF(fname) as f:
            for vname in f:
                
                # Skip if the user does not ask for it
                if not re.search('|'.join('(?:{0})'.format(v) for v in variables), vname):
                    continue
                
                # Skip if it has already been read
                if vname in FILE_VARS:
                    continue
                
                # Read the variable
                var = _read_var(f, vname)
    
    # Remove support variables
    vars = tuple(CDF_VARS[varname] for varname in CDF_VARS if varname not in SUPPORT_VARS)
    
    # Empty the global variables
    _read_reset
    
    return vars


def _read_reset():
    """
    Reset the global variables used by read()
    """
    global CDF_VARS
    global FILE_VARS
    global SUPPORT_VARS
    
    CDF_VARS = {}
    FILE_VARS = {}
    SUPPORT_VARS = {}


def _read_var(cdf, varname):
    """
    Read data and collect metadata from a CDF variable.
    
    Parameters
    ==========
    cdf: : object
        A spacepy.pycdf.CDF object of the CDF file being read.
    varname : str
        The name of the CDF variable to be read
    """
    global CDF_VARS
    global FILE_VARS
    
    # Data has already been read from this CDF file
    if varname in FILE_VARS:
        var = FILE_VARS[varname]
    
    else:
        # Read the data
        var = cdfdata(cdf[varname][...])
        var.name = varname
        var.rec_vary = cdf[varname].rv
        
        # Append to existing data
        if varname in CDF_VARS and cdf[varname].rv:
            d0 = CDF_VARS[varname]
            np.append(var.y, d0, 0)
        
        # Mark as read
        #  - Prevent infinite loop. Must save the variable in the registry
        #  so that variable attributes do not try to read the same variable
        #  again.
        CDF_VARS[varname] = var
        FILE_VARS[varname] = var
        
        # Read the metadata
        var = _read_var_attrs(cdf, var)
    
    return var


def _read_var_attrs(cdf, var):
    """
    Read metadata from a CDF variable.
    
    Parameters
    ==========
    cdf: : object
        A spacepy.pycdf.CDF object of the CDF file being read.
    var : object
        The tsdata object that will take on CDF variable attributes
        as object attributes.
    
    Returns
    ==========
    var : object
        The input tsdata object with new attributes.
    """
    global SUPPORT_VARS
    
    # Add each attribute as a 
    for vattr in cdf[var.name].attrs:
    
        # Follow pointers
        attrvalue = cdf[var.name].attrs[vattr]
        if isinstance(attrvalue, str) and attrvalue in cdf:
            varname = attrvalue
            attrvalue = _read_var(cdf, varname)
            SUPPORT_VARS[varname] = attrvalue
        
        # Set the attribute value
        if vattr == 'DELTA_PLUS_VAR':
            var.delta_plus = attrvalue
        elif vattr == 'DELTA_MINUS_VAR':
            var.delta_minus = attrvalue
        elif vattr == 'DEPEND_0':
            pdb.set_trace()
            var.x0 = attrvalue
        elif vattr == 'DEPEND_1':
            var.x1 = attrvalue
        elif vattr == 'DEPEND_2':
            var.x2 = attrvalue
        elif vattr == 'DEPEND_3':
            var.x3 = attrvalue
        elif vattr == 'LABL_PTR_1':
            var.label = attrvalue
        elif vattr == 'LABL_PTR_2':
            var.label2 = attrvalue
        elif vattr == 'LABL_PTR_3':
            var.label3 = attrvalue
        else:
            setattr(var, vattr, attrvalue)
    
    return var


class cdfdata:
    
    def __init__(self, y):
        self.y = y
    
    # def _plot_xaxes(self, h):
    #
    #     # Map FIELDNAM to plot title
    #     try plt.title(self.FIELDNAM):
    #         pass
    #     except AttributeError:
    #         pass
    #     except:
    #         raise
    #
    #     # Map LABLAXIS to the y-axis title
    #     try plt.ylabel(self.LABLAXIS):
    #         pass
    #     except AttributeError:
    #         pass
    #     except:
    #         raise
    #
    #     # Map SCLTYPE to log y-axis
    #
    # def plot(self):
    #
    #     if hasattr(self, 'x3'):
    #         a = 1
    #     elif hasattr(self, 'x2'):
    #         b = 1
    #     elif hasattr(self, 'x1'):
    #         c = 1
    #     elif hasattr(self, 'x0'):
    #         h = plt.plot(self.x0.y, self.y)
    #
    #
    #     else:
    #         e = 1
    #
    #     plt.show()
    
    """
    @property
    def x0(self):
        return self.x0
    
    @x0.setter
    def x0(self, value):
        if len(value) != np.ma.size(self.y, 0):
            ValueError('x0 is incorrect size.')
        else:
            self._x0 = value
    
    @property
    def x1(self):
        return self.x1
    
    @x1.setter
    def x1(self, value):
        if len(value) != np.ma.size(self.y, 1):
            ValueError('x1 is incorrect size.')
        else:
            self._x1 = value
    
    @property
    def x2(self):
        return self.x2
    
    @x2.setter
    def x2(self, value):
        if len(value) != np.ma.size(self.y, 2):
            ValueError('x2 is incorrect size.')
        else:
            self._x2 = value
    
    @property
    def x3(self):
        return self.x3
    
    @x3.setter
    def x3(self, value):
        if len(value) != np.ma.size(self.y, 3):
            ValueError('x3 is incorrect size.')
        else:
            self._x3 = value
    """