# Instructions for subclassing numpy.ndarray are taken from (9/8/2017)
#   https://docs.scipy.org/doc/numpy-1.13.0/user/basics.subclassing.html
import numpy as np
from spacepy import pycdf
from matplotlib import pyplot as plt
from matplotlib import image as img
import pytz
import re

class MrArray(np.ndarray):
    
    _count = 0
    _cache = []
    
    def __new__(cls, x, t=[], cache=False, name='MrArray'):
        # Convert input to ndarray (if it is not one already)
        # Cast it as an instance of MrArray
        obj = np.asarray(x).view(cls)
        
        # Add new attributes
        obj.x = np.array(x)
        obj.t = np.array(t, dtype='datetime64[s]')
        
        # Name and cache the object
        obj.name = name
        if cache:
            obj.__class__.cache(obj)
        
        # Finally, we must return the newly created object:
        return obj

    
    def __array_finalize__(self, obj):
        # ``self`` is a new object resulting from
        # ndarray.__new__(InfoArray, ...), therefore it only has
        # attributes that the ndarray.__new__ constructor gave it -
        # i.e. those of a standard ndarray.
        #
        # We could have got to the ndarray.__new__ call in 3 ways:
        # From an explicit constructor - e.g. InfoArray():
        #    obj is None
        #    (we're in the middle of the InfoArray.__new__
        #    constructor, and self.info will be set when we return to
        #    InfoArray.__new__)
        
        if obj is None: return
        # From view casting - e.g arr.view(InfoArray):
        #    obj is arr
        #    (type(obj) can be InfoArray)
        # From new-from-template - e.g infoarr[:3]
        #    type(obj) is InfoArray
        #
        # Note that it is here, rather than in the __new__ method,
        # that we set the default value for 'info', because this
        # method sees all creation of default objects - with the
        # InfoArray.__new__ constructor, but also with
        # arr.view(InfoArray).
        self.x = getattr(obj, 'x', None)
        self.t = getattr(obj, 't', None)
        self.name  = getattr(obj, 'name', 'MrArray')
        # We do not need to return anything
    
    
    def __del__(self):
        try:
            self.remove()
        except ValueError:
            pass
    
    
    def __getitem__(self, dt):
        # use duck typing
        # is it an integer or sequence?
        try:
            obj = super(MrArray, self).__getitem__(dt)
            
            # Scalar index
            if np.isscalar(dt):
                pass
                
            # Index or slice multiple dimensions
            # If scalar indices were given, obj will be a scalar (no attributes)
            elif isinstance(dt, tuple):
                try:
                    obj.t = self.t[dt[0]]
                except AttributeError:
                    pass
            
            # Index or slice single dimension
            else:
                obj.t = self.t[dt]
            
            return obj
        
        except (TypeError, IndexError) as e:
            raise
        
        # it must be datetime
        # make sure it is a numpy datetime
        dt = np.datetime64(dt,'s')
        
        # Return exact match
        idx = dt == self.t
        if np.any(idx):
            return self[idx][0]
        
        # Nearest neighbor interpolation
        idx = np.argwhere(dt < self.t)
        idx = idx[0][0]
        x_lo = self[idx - 1, ...]
        x_hi = self[idx, ...]
        t_lo = self.t[idx - 1]
        t_hi = self.t[idx]
        
        return x_lo + (x_hi - x_lo) * ((dt - t_lo) / (t_hi - t_lo))
    
    
    def __check_time(self, t):
        return np.all(self.t == t)
    
    
#    def __mul__(self, value):
#        a = 1
    
    
    def cache(self, index=None, clobber=False):
        # Store an instance in the array cache. Do not allow
        # arrays to have the same name.
        names = []
        names = [arr[1].name for arr in enumerate(self.__class__._cache)]
        
        # Check if the name already exists. When clobbering,
        # delete the existing value. Otherwise, if the name
        # exists, find the highest number in <name>_# that
        # makes <name> unique.
        if self.name in names:
            
            if clobber:
                idx = names.index(self.name)
                del self.__class__._cache[idx]
                
            else:
                rex = re.compile(self.name + '(_(\d+))?$')
                num = [re.match(rex, x).group(2) for x in names if re.match(rex, x)]
                num = max([0 if i is None else int(i) for i in num])
                self.name += '_' + str(num+1)
        
        # Add to the variable cache
        if index is None:
            self.__class__._cache.append(self)
        else:
            self.__class__._cache.insert(index, self)
    
    
    def image(self, axes=[], colorbar=True, show=True):
        # Create the figure
        if not axes:
            fig, axes = plt.subplots(nrows=1, ncols=1)
        
        # Convert time to seconds and reshape to 2D arrays
        dt2sec = np.vectorize(lambda x: x.total_seconds())
        x0 = dt2sec(self.x0.x - self.x0.x[0])
        x1 = self.x1
        if x0.ndim == 1:
            x0 = np.repeat(x0[:, np.newaxis], self.shape[1], axis=1)
        if x1.ndim == 1:
            x1 = np.repeat(x1[np.newaxis, :], self.shape[0], axis=0)
        
        # Format the image
        #   TODO: Use the delta_plus and delta_minus attributes to create
        #         x0 and x1 arrays so that x does not lose an element
        data = self.x[0:-1,0:-1]
        if hasattr(self, 'scale'):
            data = np.ma.log(data)
        
        # Create the image
        im = axes.pcolorfast(x0, x1, data, cmap='nipy_spectral')
        axes.images.append(im)
        
        # Set plot attributes
        self.__plot_apply_xattrs(axes, self.x0)
        self.__plot_apply_yattrs(axes, x1)
        
        # TODO: Add a colorbar
        if colorbar:
            cb = plt.colorbar(im)
            try:
                cb.set_label(self.title)
            except AttributeError:
                pass
        
        # Display the plot
        if show:
            plt.ion()
            plt.show()
    
    
    def iscached(self):
        # Determine if the instance has been cached
        return self in self.__class__._cache
    
    
    def plot(self, axes=[], legend=True, show=True):
        if not axes:
            axes = plt.axes()
        
        # Plot the data
        axes.plot(self.x0.x, self.x)
        
        # Set plot attributes
        self.__plot_apply_xattrs(axes, self.x0)
        self.__plot_apply_yattrs(axes, self)
        
        if legend:
            try:
                axes.legend(self.label)
            except AttributeError:
                pass
        
        # Display the plot
        if show:
            plt.ion()
            plt.show()
    
    
    def remove(self):
        self.__class__._cache.remove(self)
    
    
    @staticmethod
    def __plot_apply_xattrs(ax, x):
        try:
            ax.XLim = x.lim
        except AttributeError:
            pass
        
        try:
            ax.set_title(x.plot_title)
        except AttributeError:
            pass
        
        try:
            ax.set_xscale(x.scale)
        except AttributeError:
            pass
        
        try:
            ax.set_xlabel(x.title.replace('\n', ' '))
        except AttributeError:
            pass
        
    
    @staticmethod
    def __plot_apply_yattrs(ax, y):
        try:
            ax.YLim = y.lim
        except AttributeError:
            pass
        
        try:
            ax.set_title(y.plot_title)
        except AttributeError:
            pass
        
        try:
            ax.set_yscale(y.scale)
        except AttributeError:
            pass
        
        try:
            ax.set_ylabel(y.title)
        except AttributeError:
            pass

    
    @classmethod
    def get(cls, arrlist=[]):
        # Return all variables
        if not arrlist:
            return cls._cache
        
        # Output array
        arrs = []
        
        # Get an array from the cache
        for item in arrlist:
            if isinstance(item, int):
                arrs.append(cls._cache[item])
            
            elif isinstance(item, str):
                names = cls.names()
                arrs.append(cls._cache[names.index(item)])
            
            elif item is MrArray:
                arrs.apend(item)
            
            else:
                raise TypeError('arrlist must contain integers, strings, or MrArray objects.')
        
        # Return a single object if only one input was given
        if not isinstance(item, list):
            arrs = arrs[0]
        
        return arrs

    
    @classmethod
    def names(cls):
        names = [arr.name for arr in cls._cache]
        return names
        
        # Print the index and name of each item in the cache.
        # Use a space to pad between index and name; index is
        # right-justified and padded on the left while name
        # would be left-justified and padded on the right. How
        # then to pad between them?
        if len(cls._cache) == 0:
            print('The cache is empty.')
        else:
            print('{:4}{:3}{}'.format('Index', '', 'Name'))
            for idx, item in enumerate(cls._cache):
                print('{0:>4d}{1:<4}{2}'.format(idx, '', item.name))


def from_cdf(files, variable, cache=False, clobber=False, name=''):
    """
    Read variable data from a CDF file.

    Parameters
    ==========
    filenames : str, list
        Name of the CDF file(s) to be read.
    variable : str
        Name of the variable to be read.

    Returns
    =======
    vars : mrarry
        A mrarray object.
    """
    global cdf_vars
    global file_vars
    
    if isinstance(files, str):
        files = [files]

    # Read variables from files
    cdf_vars = {}
    for file in files:
        file_vars = {}
        with pycdf.CDF(file) as f:
            var = __from_cdf_read_var(f, variable)
    
    # Cache
    if cache:
        var.cache(clobber=clobber)
    
    return var


def __from_cdf_read_var(cdf, varname):
    """
    Read data and collect metadata from a CDF variable.

    Parameters
    ==========
    cdf: : object
        A spacepy.pycdf.CDF object of the CDF file being read.
    varname : str
        The name of the CDF variable to be read
    """
    global cdf_vars
    global file_vars

    # Data has already been read from this CDF file
    if varname in file_vars:
        var = file_vars[varname]

    else:
        # TODO
        #   - Create a time object that inherits from numpy.datetime64 that can
        #   - be cached the same way as MrArray
#        if cdf_var.type() in (pycdf.const.CDF_EPOCH.value, pycdf.const.CDF_EPOCH16.value, 
#                              pycdf.const.CDF_TIME_TT2000.value):
#            var = np.array(data, dtype='datetime64')
#        else:
#            var = MrArray(data)
            
        # Read the data
        var = MrArray(cdf[varname][...])
        var.name = varname
        var.rec_vary = cdf[varname].rv
    
        # Append to existing data
        if varname in cdf_vars and cdf[varname].rv:
            d0 = cdf_vars[varname]
            np.append(var.y, d0, 0)
    
        # Mark as read
        #  - Prevent infinite loop. Must save the variable in the registry
        #  so that variable attributes do not try to read the same variable
        #  again.
        cdf_vars[varname] = var
        file_vars[varname] = var
    
        # Read the metadata
        var = __from_cdf_read_var_attrs(cdf, var)
        var = __from_cdf_attrs2gfxkeywords(cdf, var)

    return var


def __from_cdf_read_var_attrs(cdf, var):
    """
    Read metadata from a CDF variable.
    
    Parameters
    ==========
    cdf: : object
        A spacepy.pycdf.CDF object of the CDF file being read.
    var : object
        The MrArray object that will take on CDF variable attributes
        as object attributes.
    
    Returns
    ==========
    var : object
        The input tsdata object with new attributes.
    """
    
    # CDF variable and properties
    cdf_var = cdf[var.name]
    var.cdf_name = cdf_var.name()
    var.cdf_type = cdf_var.type()
    
    # Variable attributes
    for vattr in cdf_var.attrs:
    
        # Follow pointers
        attrvalue = cdf_var.attrs[vattr]
        if isinstance(attrvalue, str) and attrvalue in cdf:
            varname = attrvalue
            attrvalue = __from_cdf_read_var(cdf, varname)
        
        # Set the attribute value
        if vattr == 'DELTA_PLUS_VAR':
            var.delta_plus = attrvalue
        elif vattr == 'DELTA_MINUS_VAR':
            var.delta_minus = attrvalue
        elif vattr == 'DEPEND_0':
            var.x0 = attrvalue
        elif vattr == 'DEPEND_1':
            var.x1 = attrvalue
        elif vattr == 'DEPEND_2':
            var.x2 = attrvalue
        elif vattr == 'DEPEND_3':
            var.x3 = attrvalue
#        elif vattr == 'LABL_PTR_1':
#            var.label = attrvalue
        elif vattr == 'LABL_PTR_2':
            var.label2 = attrvalue
        elif vattr == 'LABL_PTR_3':
            var.label3 = attrvalue
        else:
            setattr(var, vattr, attrvalue)
    
    return var


def __from_cdf_attrs2gfxkeywords(cdf, var):
    """
    Set plotting attributes for the variable based on CDF metadata.
    
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
        The input MrArray object with new attributes.
    """
    
    # Extract the attributes for ease of access
    cdf_attrs = cdf[var.name].attrs
    
    # Plot title
    if 'FIELDNAM' in cdf_attrs:
        var.plot_title = cdf_attrs['FIELDNAM']
    elif 'CATDESC' in cdf_attrs:
        var.plot_title = cdf_attrs['CATDESC']
    
    # Axis label
    title = ''
    if 'LABLAXIS' in cdf_attrs:
        title = cdf_attrs['LABLAXIS']
    elif 'FIELDNAM' in cdf_attrs:
        title = cdf_attrs['FIELDNAM']
    if 'UNITS' in cdf_attrs:
        title = title + '\n(' + cdf_attrs['UNITS'] + ')'
    var.title = title
    
    # Legend label
    if 'LABL_PTR_1' in cdf_attrs:
        var.label = cdf[cdf_attrs['LABL_PTR_1']][...]
        
    # Axis scaling
    if 'SCALETYP' in cdf_attrs:
        var.scale = cdf_attrs['SCALETYP']
    
    return var


def plot(variables=[], layout=[]):
    # Get the object references
    vars = MrArray.get(variables)
    
    # Plot layout
    if not layout:
        layout = [len(vars), 1]
    
    # Setup plot
    fig, axes = plt.subplots(nrows=layout[0], ncols=layout[1])
    
    # Plot each variable
    for idx, var in enumerate(vars):
        if hasattr(var, 'x1'):
            var.image(axes=axes[idx], show=False)
        elif hasattr(var, 'x0'):
            var.plot(axes=axes[idx], show=False)
    
    # Display the figure
    plt.show()


def main1():
    x = [1,2,3,4]
    PST = pytz.timezone('America/Los_Angeles')
    dt = [datetime(2013,1,1,12,30,0,tzinfo=PST),
          datetime(2013,1,1,13,30,0,tzinfo=PST),
          datetime(2013,1,1,14,30,0,tzinfo=PST),
          datetime(2013,1,1,15,30,0,tzinfo=PST)]
    ts = MrArray(x,dt)
    ts2 = ts + 1
    print( ts2, ts2.x, ts2.t, ts2.__class__, ts2.__class__.__bases__)
    print( ts)
    print( ts[0], ts[1], ts[2], ts[3])
    print( ts[datetime(2013,1,1,15,30,0,tzinfo=PST)])
    print( ts[datetime(2013,1,1,14,45,0,tzinfo=PST)])
    print( ts.__class__)
    print( ts.__class__.__bases__)


def main_mms_fpi():
    from pymms.pymms import MrMMS_SDC_API
    
    # Get the data file
    sdc = MrMMS_SDC_API('mms1', 'fpi', 'brst', 'l2', 
                        optdesc='des-moms', 
                        start_date='2015-12-06T00:23:04', 
                        end_date='2015-12-06T00:25:34')
    file = sdc.Download()
    
    # Variable name
    n_vname = 'mms1_des_numberdensity_brst'
    v_vname = 'mms1_des_bulkv_gse_brst'
    espec_vname = 'mms1_des_energyspectr_omni_brst'
    
    # Read data
    n = from_cdf(file, n_vname, cache=True)
    V = from_cdf(file, v_vname, cache=True)
    ESPec = from_cdf(file, espec_vname, cache=True)
    
    # Plot data
    plot()
    
    return file


if __name__ == '__main__':
    main_mms_fpi()

