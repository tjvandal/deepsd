import os
import shutil
import sys
import csv
import tempfile
import gdal
import gdalconst
import zipfile as zf
import numpy as np
import pandas as pd
#from unitconversion import *
from StringIO import StringIO
import xarray as xr
import ConfigParser
import utils

import tensorflow as tf
from tfwriter import convert_to_tf


def recursive_mkdir(path):
    split_dir = path.split("/")
    for k in range(len(split_dir)):
        d = "/".join(split_dir[:(k+1)])
        if (d != '') and (not os.path.exists(d)):
            os.mkdir(d)

class PrismBil:
    def __init__(self, zip_file_pointer):
        self.zf = zip_file_pointer
        self.bilname = [n for n in self.zf.namelist() if n[-4:] == '.bil'][0]
        self.hdrname = [n for n in self.zf.namelist() if n[-4:] == '.hdr'][0]
        date = self.bilname.split("_")[-2]
        self.date = pd.to_datetime(date, format='%Y%m%d')
        print self.date

    def save_temp(self):
        tmp_dir = tempfile.mkdtemp()
        print "making temp dir", tmp_dir
        for f in [self.bilname, self.hdrname]:
            fn = os.path.join(tmp_dir, f)
            with open(fn, 'wb') as file:
                file.write(self.zf.read(f))
        self.bilfile = os.path.join(tmp_dir, self.bilname)
        self.hdrfile = os.path.join(tmp_dir, self.hdrname)
        return tmp_dir

    def bil_to_xray(self):
        try:
            tmp_dir = self.save_temp()
            img = gdal.Open(self.bilfile, gdalconst.GA_ReadOnly)
            band = img.GetRasterBand(1)
            self.nodatavalue = band.GetNoDataValue()
            self.ncol = img.RasterXSize
            self.nrow = img.RasterYSize
            geotransform = img.GetGeoTransform()
            self.originX = geotransform[0]
            self.originY = geotransform[3]
            self.pixelWidth = geotransform[1]
            self.pixelHeight = geotransform[5]
            self.data = band.ReadAsArray()
            self.data = np.ma.masked_where(self.data==self.nodatavalue, self.data)
            lats = np.linspace(self.originY, self.originY + self.pixelHeight *(self.nrow - 1),
                             self.nrow)
            lons = np.linspace(self.originX, self.originX + self.pixelWidth * (self.ncol -1),
                             self.ncol)
            #print lats[:-1] - lats[1:]
            #sys.exit()
            dr = xr.DataArray(self.data[np.newaxis, :, :],
                              coords=dict(time=[self.date], lat=lats, lon=lons), 
                              dims=['time', 'lat', 'lon']) 
        finally:
            shutil.rmtree(tmp_dir)
        return dr

    def plot(self):
        import matplotlib.pyplot as plt
        plt.imshow(self.data[:200, :200])
        plt.colorbar()
        plt.show()

    def close(self):
        self.zf.close()

def downloadPrismFtpData(parm, output_dir=os.getcwd(), timestep='monthly', years=None, server='prism.oregonstate.edu'):
    """
    Downloads ESRI BIL (.hdr) files from the PRISM FTP site.
    'parm' is the parameter of interest: 'ppt', precipitation; 'tmax', temperature, max' 'tmin', temperature, min /
                                         'tmean', temperature, mean
    'timestep' is either 'monthly' or 'daily'. This string is used to direct the function to the right set of remote folders.
    'years' is a list of the years for which data is desired.
    """
    from ftplib import FTP
    import socket

    recursive_mkdir(output_dir)
    data = []

    def handleDownload(block):
        data.append(block)

    # Play some defense
    assert parm in ['ppt', 'tmax', 'tmean', 'tmin'], "'parm' must be one of: ['ppt', 'tmax', 'tmean', 'tmin']"
    assert timestep in ['daily', 'monthly'], "'timestep' must be one of: ['daily', 'monthly']"
    assert years is not None, 'Please enter a year for which data will be fetched.'
    if isinstance(years, int):
        years = list(years)
    try:
        ftp = FTP(server, timeout=5)
        ftp.login()
    except socket.timeout:
        print("Cannot connect to FTP server, socket.timeout")
        return
    # Wrap everything in a try clause so we close the FTP connection gracefully
    try:
        for year in years:
            save_nc_file = os.path.join(output_dir, "prism_%s_4km2_%04i.nc" % (parm, year))
            if os.path.exists(save_nc_file):
                continue
            data = []
            xray_data = []
            if timestep == 'daily':
                dir = timestep
            dir_string = '{}/{}/{}'.format(dir, parm, year)
            remote_files = sorted(ftp.nlst(dir_string))
            for f_string in remote_files:
                print f_string
                f = f_string.rsplit(' ')[-1]
                if not '_bil' in f:
                    continue

                f_path = '{}'.format(f)
                ftp.retrbinary('RETR ' + f_path, handleDownload)
                c = StringIO("".join(data))
                with zf.ZipFile(c) as z:
                    p = PrismBil(z)
                    xray_data.append(p.bil_to_xray())
            ds = xr.Dataset({parm: xr.concat(xray_data, dim='time')})
            ds.to_netcdf(save_nc_file, format='NETCDF3_CLASSIC')

    except Exception as e:
        print e
    finally:
        ftp.close()

    return

class PrismBase(object):
    def __init__(self, data_dir, year, elevation_file=None, var='ppt'):
        self.data_dir = data_dir
        self.var=var
        self.year = year
        self.elevation_file = elevation_file
        self.read_data()

    def _get_year_file(self):
        print "data dir", self.data_dir
        fnames = [f for f in os.listdir(self.data_dir) if str(self.year) in f]

        if len(fnames) == 1:
            return os.path.join(self.data_dir, fnames[0])
        elif len(fnames) == 0:
            raise IndexError("File for year:%i not found" % self.year)
        elif len(fnames) > 1:
            raise IndexError("Multiples files for year:%i found" % self.year)

    def _read_highres(self):
        highres_file = self._get_year_file()
        print 'highres file', highres_file
        self.highres = xr.open_dataset(highres_file)

    def _read_elevation(self):
        elev = xr.open_dataset(self.elevation_file)
        self.elev = elev.rename({"Band1": "elev"})

    def read_data(self):
        self._read_highres()
        if self.elevation_file is not None:
            self._read_elevation()
            self.elev = self.elev.elev.sel(lat=self.highres.lat, lon=self.highres.lon, method='nearest')
            self.elev['lat'] = self.highres.lat
            self.elev['lon'] = self.highres.lon

class PrismSuperRes(PrismBase):
    def __init__(self, data_dir, year, elevation_file, var='ppt', model='srcnn'):
        super(PrismSuperRes, self).__init__(data_dir, year, elevation_file, var=var)
        self.model = model

    def resolve_data(self, scale1=1., scale2=0.5):
        '''
        Interpolate the data in accordance to the scaling factors
        A scaling factor of 0.5 cuts the resolution in half.
        '''
        # crop data to ensure integer upscaling factors
        factor = 1 / (scale1 * scale2)
        if int(factor) != factor:
            print "Factor =", factor
            raise ValueError("scale1 and scale2 must be 1/int()")
        factor = int(factor)
        t, h, w = self.highres[self.var].shape
        Y = self.highres[self.var]
        Y = Y.isel(lat=range(0,h - h % factor), lon=range(0,w - w % factor))

        # get an approximate mask
        tmp = Y.isel(time=0)+1
        mask = tmp/tmp

        # this fills missing values
        Y_interp = utils.interp_da(Y, scale1)
        elev = self.elev.isel(lat=range(0,h - h % factor),
                              lon=range(0,w - w % factor))
        elev = utils.interp_da2d(elev, scale1)
        mask = mask.sel(lat=Y_interp.lat, lon=Y_interp.lon, method='nearest')

        #if self.model == 'resnet':
        X = utils.interp_da(Y_interp, scale2)
        #elif self.model == 'srcnn':
        #    X = utils.interp_da(utils.interp_da(Y_interp, scale2), 1./scale2)
        #else:
        #    raise ValueError('The model parameter should be set to either srcnn or resnet.')
        return mask, X, Y_interp, elev

    def make_patches(self, save_file=None, size=50, stride=30, scale1=1., scale2=0.5):
        assert (size * scale2) == int(size*scale2)
        assert (stride * scale2) == int(stride * scale2)
        mask, da1, da2, elev = self.resolve_data(scale1, scale2)
        obs_lats = da2.lat.values
        obs_lons = da2.lon.values
        X = da1.values
        Y = da2.values

        # keep elevation flexible by returning it seperately
        elev = elev.values[:Y.shape[1],:Y.shape[2],np.newaxis]
        #tmp = np.empty(shape=(X.shape[0], X.shape[1], X.shape[2], 1))
        #tmp[:] = elev
        X = np.expand_dims(X, 3)
        #X = np.concatenate([X, tmp], axis=3)

        labels, inputs, elevs = [], [], []
        lats, lons, times = [], [], []
        timevals = da1.time.values
        for j, t in enumerate(timevals):
            for y in np.arange(0, Y.shape[1], stride):
                for x in np.arange(0, Y.shape[2], stride):
                    if ((y+size) > Y.shape[1]) or ((x+size) > Y.shape[2]):
                        continue

                    x_lr = int(x*scale2)
                    y_lr = int(y*scale2)
                    s_lr = int(size*scale2)
                    x_sub = X[j, np.newaxis, y_lr:y_lr+s_lr, x_lr:x_lr+s_lr]

                    # are we over the ocean? 
                    land_ratio = mask.notnull().values[y:y+size, x:x+size].mean()
                    if land_ratio < 0.50:
                        continue

                    y_sub = Y[j, np.newaxis, y:y+size, x:x+size, np.newaxis]
                    elev_sub = elev[np.newaxis,y:y+size,x:x+size,:]

                    inputs += [x_sub]
                    labels += [y_sub]
                    elevs += [elev_sub]
                    lats += [obs_lats[np.newaxis, y:y+size]]
                    lons += [obs_lons[np.newaxis, x:x+size]]
                    times += [t]

        order = range(len(inputs))
        np.random.shuffle(order)
        self.inputs = np.concatenate(inputs, axis=0)[order]
        self.labels = np.concatenate(labels, axis=0)[order]
        elevs= np.concatenate(elevs, axis=0)[order]
        self.lats = np.vstack(lats)[order]
        self.lons = np.vstack(lons)[order]
        print "Number of subimages", len(self.inputs)
        if save_file is not None:
            convert_to_tf(self.inputs, elevs, self.labels, self.lats, self.lons, np.array(times)[order], save_file)

    def make_test(self, scale1=1., scale2=0.5):
        mask, da1, da2, elev = self.resolve_data(scale1, scale2)
        Y = (da2.values * mask.values)[:,:,:,np.newaxis]
        X = da1.values
        elev_arr = np.empty((Y.shape[0], Y.shape[1], Y.shape[2], 1))
        elev_arr[:] = elev.values[:,:,np.newaxis]
        X = np.expand_dims(X, 3)

        times = da2.time.values
        lats = [da2.lat.values for i in range(Y.shape[0])]
        lons = [da2.lon.values for i in range(Y.shape[0])]
        return X, elev_arr, Y, lats, lons, times

    def make_tf_test(self, save_file, scale1=1., scale2=0.5):
        X, elev, Y, lats, lons, times = self.make_test(scale1, scale2)
        order = range(Y.shape[0])
        np.random.shuffle(order)
        convert_to_tf(X[order], elev[order], Y[order], lats, lons,
                     times[order], save_file)

# Save Basic SRCNN Data
def main_prism_tf(config, model='srcnn'):
    var = config.get('DataOptions', 'variable')
    minyear = int(config.get('DataOptions', 'min_year'))
    maxyear = int(config.get('DataOptions', 'max_year'))
    patch_size = int(config.get('SRCNN', 'training_input_size'))

    highest_resolution = 4
    hr_resolution_km = config.getint('DeepSD', 'high_resolution')
    lr_resolution_km = config.getint('DeepSD', 'low_resolution')
    upscale_factor = config.getint('DeepSD', 'upscale_factor')

    start = hr_resolution_km / highest_resolution
    N = int((lr_resolution_km / hr_resolution_km)**(1./upscale_factor))

    scale2 = 1./upscale_factor  # scale2 is relative to scale1
    for scale1 in [start * scale2**i for i in range(N)]:
        save_dir = os.path.join(config.get('Paths', 'scratch'),
                        '%s_%03i_%03i' % (var, hr_resolution_km/scale1,
                                          hr_resolution_km/(scale1*scale2)))
        recursive_mkdir(save_dir)
        for y in range(minyear, maxyear+1):
            print "Year: %i" % y
            #d = SRCNNData(data_dir, y, elevation_file)
            d = PrismSuperRes(os.path.join(config.get('Paths', 'prism'), 'ppt','raw'), y,
                              config.get('Paths', 'elevation'), model=model.lower())
            if y <= int(config.get('DataOptions', 'max_train_year')):
                print "Making patches or year:", y
                tf_file = os.path.join(save_dir, 'train_%i.tfrecords' % y)
                print tf_file
                if  not os.path.exists(tf_file):
                    print "trying to make patches"
                    d.make_patches(tf_file, size=patch_size, stride=20, scale1=scale1, scale2=scale2)
            else:
                print "Building test set for year:", y
                tf_file = os.path.join(save_dir, 'test_%i.tfrecords' % y)
                print tf_file
                if not os.path.exists(tf_file):
                    d.make_tf_test(tf_file, scale1, scale2)


if __name__ == "__main__":
    flags = tf.flags
    flags.DEFINE_string('config_file', 'config.ini', 'Configuration file with [SRCNN] section.')

    # parse flags
    FLAGS = flags.FLAGS
    FLAGS._parse_flags()

    config = ConfigParser.ConfigParser()
    config.read(FLAGS.config_file)
    data_dir = config.get('Paths', 'prism')
    min_year = int(config.get('DataOptions', 'min_year'))
    max_year = int(config.get('DataOptions', 'max_year'))

    if not os.path.exists(data_dir):
        recursive_mkdir(data_dir)

    for var in ['ppt',]:
        downloadPrismFtpData(var, os.path.join(data_dir, var, 'raw'), 'daily',
                             range(min_year, max_year+1))

    main_prism_tf(config)

