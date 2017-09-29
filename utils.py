import xarray as xr
import numpy as np
import cv2
from scipy.misc import imresize
import scipy.interpolate

def fillmiss(x):
    if x.ndim != 2:
        raise ValueError("X have only 2 dimensions.")
    mask = ~np.isnan(x)
    xx, yy = np.meshgrid(np.arange(x.shape[1]), np.arange(x.shape[0]))
    xym = np.vstack( (np.ravel(xx[mask]), np.ravel(yy[mask])) ).T
    data0 = np.ravel(x[mask])
    interp0 = scipy.interpolate.NearestNDInterpolator(xym, data0)
    result0 = interp0(np.ravel(xx), np.ravel(yy)).reshape(xx.shape)
    return result0

def interp_dim(x, scale):
    x0, xlast = x[0], x[-1]
    step = (x[1]-x[0])/scale
    y = np.arange(x0, xlast+step*scale, step)
    #[x0 + s*scale for x in range()]
    return y

def interp_tensor(X, scale, fill=True):
    nlt = int(X.shape[1]*scale)
    nln = int(X.shape[2]*scale)
    newshape = (X.shape[0], nlt, nln)
    scaled_tensor = np.empty(newshape)
    for j, im in enumerate(X):
        # fill im with nearest neighbor
        if fill:
            #im = fillmiss(im)
            im[np.isnan(im)] = 0

        scaled_tensor[j] = cv2.resize(im, (newshape[2], newshape[1]),
                                     interpolation=cv2.INTER_CUBIC)
    return scaled_tensor

def interp_da(da, scale):
    '''
    Assume da is of dimensions ('time','lat', 'lon')
    '''
    tensor = da.values

    # lets store our interpolated data
    scaled_tensor = interp_tensor(tensor, scale, fill=True)

    # interpolate lat and lons
    latnew = interp_dim(da[da.dims[1]].values, scale)
    lonnew = interp_dim(da[da.dims[2]].values, scale)
    if latnew.shape[0] != scaled_tensor.shape[1]:
        raise ValueError("New shape is shitty")
    # intialize a new dataarray
    return xr.DataArray(scaled_tensor, coords=[da[da.dims[0]].values, latnew, lonnew],
                 dims=da.dims)

def interp_da2d(da, scale, fillna=False):
    '''
    Assume da is of dimensions ('time','lat', 'lon')
    '''
    # lets store our interpolated data
    newshape = (int(da.shape[0]*scale),int(da.shape[1]*scale))
    im = da.values
    scaled_tensor = np.empty(newshape)
    # fill im with nearest neighbor
    if fillna:
        filled = fillmiss(im)
    else:
        filled = im
    scaled_tensor = cv2.resize(filled, dsize=(0,0), fx=scale, fy=scale,
                              interpolation=cv2.INTER_CUBIC)

    # interpolate lat and lons
    latnew = interp_dim(da[da.dims[0]].values, scale)
    lonnew = interp_dim(da[da.dims[1]].values, scale)
    # intialize a new dataarray
    return xr.DataArray(scaled_tensor, coords=[latnew, lonnew],
                 dims=da.dims)

if __name__=="__main__":
    import matplotlib.pyplot as plt

    fhigh = '/raid/prism/ppt_0.125x0.125/prism_ppt_interp_1981.nc'
    var='ppt'

    dshigh = xr.open_dataset(fhigh)
    dshigh = dshigh.isel(time=[0,1])
    #dshigh['ppt'] = dshigh.ppt.fillna(0)
    dalow = interp_da(dshigh.ppt, 1./8)
    danew = interp_da(dalow, 8.)

    plt.figure(figsize=(8,20))
    plt.subplot(3,1,1)
    danew.isel(time=0).plot()
    plt.subplot(3,1,2)
    dshigh.isel(time=0).ppt.plot()
    plt.subplot(3,1,3)
    dalow.isel(time=0).plot()
    plt.show()
