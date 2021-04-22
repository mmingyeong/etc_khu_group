"""This is the interpolation module of MSE ETC.
This module includes calculation of Atmospheric transmission using data set by Boxkernal convolution

Modification Log
2020.02.25 - First created by Taeeun Kim
2020.03.24 - Updated by Tae-Geun Ji
2020.03.29 - Updated by Tae-Geun Ji
2021.04.21 - updated by Mingyoeng Yang
"""

from parameters import *
from scipy import interpolate
from astropy.io import fits
import numpy as np
import time

# change 20210421 by MY
class Throughput:

    def __init__(self):

        print('...... Reading skytable for Low Resolution')

        blue_low_path = 'SKY/MSE_AM1_BLUE_2550.dat'
        green_low_path = 'SKY/MSE_AM1_GREEN_3650.dat'
        red_low_path = 'SKY/MSE_AM1_RED_3600.dat'
        nir_low_path = 'SKY/MSE_AM1_NIR_3600.dat'

        self.data_blue_low = np.genfromtxt(blue_low_path, names=('wavelength', 'data1', 'data2', 'data3'))
        self.data_green_low = np.genfromtxt(green_low_path, names=('wavelength', 'data1', 'data2', 'data3'))
        self.data_red_low = np.genfromtxt(red_low_path, names=('wavelength', 'data1', 'data2', 'data3'))
        self.data_nir_low = np.genfromtxt(nir_low_path, names=('wavelength', 'data1', 'data2', 'data3'))

        # =================================== add by MY ==============================================

        # atmospheric transmission data with no convolution
        blue_box_path = 'SKY/MSE_AM1_box_blue.fits'
        green_box_path = 'SKY/MSE_AM1_box_green.fits'
        red_box_path = 'SKY/MSE_AM1_box_red.fits'
        nir_box_path = 'SKY/MSE_AM1_box_nir.fits'

        # read fits files
        self.file_blue = fits.open(blue_box_path)
        self.file_green = fits.open(green_box_path)
        self.file_red = fits.open(red_box_path)
        self.file_nir = fits.open(nir_box_path)

        self.data_blue = self.file_blue[1].data         #blue 350-560
        self.data_green = self.file_green[1].data       #green 540-740
        self.data_red = self.file_red[1].data           #red 715-985
        self.data_nir = self.file_nir[1].data           #nir 960-1800

        # close files
        self.file_blue.close()
        self.file_green.close()
        self.file_red.close()
        self.file_nir.close()

        # set data array (wavelength, transmission)
        self.wave_blue = self.data_blue.field(0)
        self.wave_green = self.data_green.field(0)
        self.wave_red = self.data_red.field(0)
        self.wave_nir = self.data_nir.field(0)

        # self.atmo_blue = []
        self.atmo_blue_pwv1 = self.data_blue.field(1)
        self.atmo_blue_pwv2 = self.data_blue.field(2)
        self.atmo_blue_pwv7 = self.data_blue.field(3)

        # self.atmo_green = []
        self.atmo_green_pwv1 = self.data_green.field(1)
        self.atmo_green_pwv2 = self.data_green.field(2)
        self.atmo_green_pwv7 = self.data_green.field(3)

        # self.atmo_red = []
        self.atmo_red_pwv1 = self.data_red.field(1)
        self.atmo_red_pwv2 = self.data_red.field(2)
        self.atmo_red_pwv7 = self.data_red.field(3)

        # self.atmo_nir = []
        self.atmo_nir_pwv1 = self.data_nir.field(1)
        self.atmo_nir_pwv2 = self.data_nir.field(2)
        self.atmo_nir_pwv7 = self.data_nir.field(3)

        # ==========================================================================================

        self.tau_wave = []
        self.tel_m1_zecoat_arr = []
        self.tel_wfc_adc_arr = []
        self.sip_fits_arr = []
        self.sip_arr = []

        self.data_pwv = [1.0, 2.5, 7.5]
        self.data_atmo = []
        self.tau_atmo = 0
        self.tau_opt = 0
        self.tau_ie = 0

    def set_data(self, res_mode):
        if res_mode == "LR":
            self.wave_blue = self.data_blue_low['wavelength']
            self.wave_green = self.data_green_low['wavelength']
            self.wave_red = self.data_red_low['wavelength']
            self.wave_nir = self.data_nir_low['wavelength']

            nlen = len(self.data_blue_low)
            self.atmo_blue = [[0] * 3 for i in range(nlen)]

            for i in range(0, nlen):
                self.atmo_blue[i] = [self.data_blue_low['data1'][i],
                                     self.data_blue_low['data2'][i],
                                     self.data_blue_low['data3'][i]]

            self.wave_green = self.data_green_low['wavelength']

            nlen = len(self.data_green_low)
            self.atmo_green = [[0] * 3 for i in range(nlen)]

            for i in range(0, nlen):
                self.atmo_green[i] = [self.data_green_low['data1'][i],
                                      self.data_green_low['data2'][i],
                                      self.data_green_low['data3'][i]]

            nlen = len(self.data_red_low)
            self.atmo_red = [[0] * 3 for i in range(nlen)]

            for i in range(0, nlen):
                self.atmo_red[i] = [self.data_red_low['data1'][i],
                                    self.data_red_low['data2'][i],
                                    self.data_red_low['data3'][i]]

            nlen = len(self.data_nir_low)
            self.atmo_nir = [[0] * 3 for i in range(nlen)]

            for i in range(0, nlen):
                self.atmo_nir[i] = [self.data_nir_low['data1'][i],
                                    self.data_nir_low['data2'][i],
                                    self.data_nir_low['data3'][i]]

            data = np.loadtxt("Throughput_LR.dat")

            self.tau_wave = data[:, 0]
            self.tel_m1_zecoat_arr = data[:, 1]
            self.tel_wfc_adc_arr = data[:, 2]
            self.sip_fits_arr = data[:, 3]
            self.sip_arr = data[:, 4]
            self.data_tau_ie = data[:, 5]


    def tau_atmo_blue(self, pwv):

        if pwv == 1.0:
            func = interpolate.interp1d(self.wave_blue, self.atmo_blue_pwv1, kind='linear', bounds_error=False,)
            self.tau_atmo = func(self.wave_blue)

        if pwv == 2.5:
            func = interpolate.interp1d(self.wave_blue, self.atmo_blue_pwv2, kind='linear', bounds_error=False,)
            self.tau_atmo = func(self.wave_blue)

        if pwv == 7.5:
            func = interpolate.interp1d(self.wave_blue, self.atmo_blue_pwv7, kind='linear', bounds_error=False,)
            self.tau_atmo = func(self.wave_blue)

        return self.tau_atmo

    def tau_atmo_green(self, pwv):

        if pwv == 1.0:
            func = interpolate.interp1d(self.wave_green, self.atmo_green_pwv1, kind='linear', bounds_error=False,)
            self.tau_atmo = func(self.wave_green)

        if pwv == 2.5:
            func = interpolate.interp1d(self.wave_green, self.atmo_green_pwv2, kind='linear', bounds_error=False,)
            self.tau_atmo = func(self.wave_green)

        if pwv == 7.5:
            func = interpolate.interp1d(self.wave_green, self.atmo_green_pwv7, kind='linear', bounds_error=False,)
            self.tau_atmo = func(self.wave_green)

        return self.tau_atmo

    def tau_atmo_red(self, pwv):

        if pwv == 1.0:
            func = interpolate.interp1d(self.wave_red, self.atmo_red_pwv1, kind='linear', bounds_error=False,)
            self.tau_atmo = func(self.wave_red)

        if pwv == 2.5:
            func = interpolate.interp1d(self.wave_red, self.atmo_red_pwv2, kind='linear', bounds_error=False,)
            self.tau_atmo = func(self.wave_red)

        if pwv == 7.5:
            func = interpolate.interp1d(self.wave_red, self.atmo_red_pwv7, kind='linear', bounds_error=False,)
            self.tau_atmo = func(self.wave_red)

        return self.tau_atmo

    def tau_atmo_nir(self, pwv):

        if pwv == 1.0:
            func = interpolate.interp1d(self.wave_nir, self.atmo_nir_pwv1, kind='linear', bounds_error=False,)
            self.tau_atmo = func(self.wave_nir)

        if pwv == 2.5:
            func = interpolate.interp1d(self.wave_nir, self.atmo_nir_pwv2, kind='linear', bounds_error=False,)
            self.tau_atmo = func(self.wave_nir)

        if pwv == 7.5:
            func = interpolate.interp1d(self.wave_nir, self.atmo_nir_pwv7, kind='linear', bounds_error=False,)
            self.tau_atmo = func(self.wave_nir)

        return self.tau_atmo


    #calculate atmosphric throughput with parameters(wavelength, transmission, pwv)
    def Cal_TAU_atmo(self, wave, transmission1, transmission2, transmission7, pwv):

        N_data = len(wave)
        y = np.zeros(N_data)

        if pwv >= 1 and pwv <= 2.5:
            for i in np.arange(0, N_data):
                y[i] = transmission1[i] + (pwv - 1) * (transmission2[i] - transmission1[i])
                #if pwv >= 2 and pwv <= 4:
                #for i in np.arange(0, N_data):
                #y[i] = transmission2[i] + (pwv - 2) * (transmission4[i] - transmission2[i]) / (4 - 2)
        elif pwv > 2.5 and pwv <= 7.5:
            for i in np.arange(0, N_data):
                y[i] = transmission2[i] + (pwv - 2.5) * (transmission7[i] - transmission2[i]) / (7.5 - 2.5)
        else:
            return self.print_error

        return y


    #determine atmospheric throughput
    def Get_TAU_atmo(self, input_pwv, input_wavelength):


        if 350.0 <= input_wavelength < 540.0:

            transmission1 = self.tau_atmo_blue(self.data_pwv[0])
            transmission2 = self.tau_atmo_blue(self.data_pwv[1])
            transmission7 = self.tau_atmo_blue(self.data_pwv[2])

            throughput = self.Cal_TAU_atmo(self.wave_blue, transmission1, transmission2, transmission7, input_pwv)
            func = interpolate.interp1d(self.wave_blue, throughput, kind='linear', bounds_error=False,)


        if 540.0 <= input_wavelength < 715.0:

            transmission1 = self.tau_atmo_green(self.data_pwv[0])
            transmission2 = self.tau_atmo_green(self.data_pwv[1])
            transmission7 = self.tau_atmo_green(self.data_pwv[2])

            throughput = self.Cal_TAU_atmo(self.wave_green, transmission1, transmission2, transmission7, input_pwv)
            func = interpolate.interp1d(self.wave_green, throughput, kind='linear', bounds_error=False,)


        if 715.0 <= input_wavelength < 960:

            transmission1 = self.tau_atmo_red(self.data_pwv[0])
            transmission2 = self.tau_atmo_red(self.data_pwv[1])
            transmission7 = self.tau_atmo_red(self.data_pwv[2])

            throughput = self.Cal_TAU_atmo(self.wave_red, transmission1, transmission2, transmission7, input_pwv)
            func = interpolate.interp1d(self.wave_red, throughput, kind='linear', bounds_error=False,)


        if 960 <= input_wavelength:

            transmission1 = self.tau_atmo_nir(self.data_pwv[0])
            transmission2 = self.tau_atmo_nir(self.data_pwv[1])
            transmission7 = self.tau_atmo_nir(self.data_pwv[2])

            throughput = self.Cal_TAU_atmo(self.wave_nir, transmission1, transmission2, transmission7, input_pwv)
            func = interpolate.interp1d(self.wave_nir, throughput, kind='linear', bounds_error=False,)

        self.tau_atmo = func(input_wavelength)

        return self.tau_atmo

    def tau_opt_res(self, wave):
        func_tel_m1_zecoat = interpolate.interp1d(self.tau_wave, self.tel_m1_zecoat_arr, kind='cubic')
        func_tel_wfc_adc = interpolate.interp1d(self.tau_wave, self.tel_wfc_adc_arr, kind='cubic')
        func_sip_fits = interpolate.interp1d(self.tau_wave, self.sip_fits_arr, kind='cubic')
        func_sip = interpolate.interp1d(self.tau_wave, self.sip_arr, kind='cubic')

        self.tau_opt = ENCL_LR * TEL_MSTR_LR * func_tel_m1_zecoat(wave) * TEL_PFHS_LR * func_tel_wfc_adc(wave) \
                         * SIP_POSS_LR * func_sip_fits(wave) * func_sip(wave)
        return self.tau_opt

    def tau_ie_res(self, wave):
        func_tau_ie = interpolate.interp1d(self.tau_wave, self.data_tau_ie, kind='cubic')
        self.tau_ie = func_tau_ie(wave)

        return self.tau_ie
