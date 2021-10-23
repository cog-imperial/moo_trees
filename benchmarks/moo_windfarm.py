# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 23:39:23 2021

@author: calvintsay
"""
import numpy as np


def windfarm(x=np.array([0]), y=np.array([0]), ispresent=np.array([True])):
    # evaluate energy production and efficiency from windfarm placement (up to 16 turbines)
    #
    # inputs:
    #   x           - list of x-coordinates of turbines [0, 3900m]
    #   y           - list of y-coordinates of turbines [0, 3900m]
    #   ispresent   - list of whether each turbine is present [boolean]
    # outputs:
    #   E_wf     - average energy production
    #   nu_wf    - energy efficiency

    if not type(x) == np.ndarray: x = np.array(x)
    if not type(y) == np.ndarray: y = np.array(y)
    if not type(ispresent) == np.ndarray: ispresent = np.array(ispresent)

    x = x[ispresent]
    y = y[ispresent]
    N_turbines = len(x)
    N_max = 16
    locs = np.stack((x, y)).T

    if N_turbines == 0:
        return 500, 500

    # wind data      dir    speed  freq
    #                [deg]  [m/s]  [-]
    data = np.array([[0, 9.77, 0.063],
                     [30, 8.34, 0.059],
                     [60, 7.93, 0.055],
                     [90, 10.18, 0.078],
                     [120, 8.14, 0.083],
                     [150, 8.24, 0.065],
                     [180, 9.05, 0.114],
                     [210, 11.59, 0.146],
                     [240, 12.11, 0.121],
                     [270, 11.90, 0.085],
                     [300, 10.38, 0.064],
                     [330, 8.14, 0.067]])

    E_wf_num = 0;
    E_wf_dem = 0
    for wind in data:
        for i in range(N_turbines):
            v = wind[1] * (1 - get_total_deficit(locs, locs[i], wind[1], wind[0]))
            power, _ = turbine_curve(v)
            E_wf_num += power * wind[2]
        power, _ = turbine_curve(wind[1])
        E_wf_dem += power * wind[2] * N_max

    P_ideal = E_wf_dem / N_max
    E_wf = E_wf_num / E_wf_dem
    nu_wf = E_wf_num / N_turbines / P_ideal


    return -E_wf, -nu_wf


def get_total_deficit(locs1, loc2, v, theta):
    # get deficits caused by all turbines at locs1 entries to turbine at loc2
    # square root of sum of squared individiual deficits
    deficit = np.zeros(len(locs1))
    for i in range(len(locs1)):
        deficit[i] = get_deficit(locs1[i], loc2, v, theta)
    return np.sqrt(np.sum(deficit ** 2))


def get_deficit(loc1, loc2, v, theta):
    # get velocity deficit from wake effects caused by turbine at loc1 to turbine at loc2
    # Katic-Jensen wake model
    #
    # inputs:
    #   loc1 (x,y)  - location of turbine 1
    #   loc2 (x,y)  - location of turbine 2
    #   v [m/s]     - wind speed
    #   theta [deg] - wind angle
    # outputs:
    #   deficit     - reduction in effective wind velocity

    theta = np.deg2rad(theta)

    # turbine geometry
    D = 164  # rotor diameter [m]
    H = 107  # hug height [m]

    # rotate coordinates to calculate distances
    loc1r = np.array([loc1[0] * np.cos(theta) + loc1[1] * np.sin(theta),
                      -loc1[0] * np.sin(theta) + loc1[1] * np.cos(theta)])
    loc2r = np.array([loc2[0] * np.cos(theta) + loc2[1] * np.sin(theta),
                      -loc2[0] * np.sin(theta) + loc2[1] * np.cos(theta)])

    d_kj = loc2r[0] - loc1r[0]  # axial distance in wind direction
    c_kj = np.abs(loc2r[1] - loc1r[1])  # radial distance in wind direction
    alpha = 0.5 / np.log(H / 0.0005)
    R_j = D / 2
    R_kw = R_j + alpha * d_kj

    if d_kj <= 0 or c_kj >= R_j + R_kw:
        deficit = 0
    else:
        _, thrust = turbine_curve(v)
        U_kj = (1 - np.sqrt(1 - thrust)) / (1 + alpha * d_kj / R_j) ** 2
        if c_kj <= R_kw - R_j:
            deficit = U_kj
        else:
            A_j = np.pi / 4 * D ** 2
            A_kj = 1 / 2 * R_kw ** 2 * (2 * np.arccos(
                (R_kw ** 2 + c_kj ** 2 - R_j ** 2) / (2 * R_kw * c_kj)) - np.sin(
                2 * np.arccos((R_kw ** 2 + c_kj ** 2 - R_j ** 2) / (2 * R_kw * c_kj)))) + \
                   1 / 2 * R_j ** 2 * (2 * np.arccos(
                (R_j ** 2 + c_kj ** 2 - R_kw ** 2) / (2 * R_j * c_kj)) - np.sin(
                2 * np.arccos((R_j ** 2 + c_kj ** 2 - R_kw ** 2) / (2 * R_j * c_kj))))
            deficit = U_kj * A_kj / A_j
    return deficit


def turbine_curve(v):
    # interpolate power production [kW], thrust value [-], as a function of wind speed [m/s]
    data = np.array([[4, 100, 0.700000000],
                     [5, 570, 0.722386304],
                     [6, 1103, 0.773588333],
                     [7, 1835, 0.773285946],
                     [8, 2858, 0.767899317],
                     [9, 4089, 0.732727569],
                     [10, 5571, 0.688896343],
                     [11, 7105, 0.623028669],
                     [12, 7873, 0.500046699],
                     [13, 7986, 0.373661747],
                     [14, 8008, 0.293230676],
                     [15, 8008, 0.238407400],
                     [16, 8008, 0.196441644],
                     [17, 8008, 0.163774674],
                     [18, 8008, 0.137967245],
                     [19, 8008, 0.117309371],
                     [20, 8008, 0.100578122],
                     [21, 8008, 0.086883163],
                     [22, 8008, 0.075565832],
                     [23, 8008, 0.066131748],
                     [24, 8008, 0.058204932],
                     [25, 8008, 0.051495998]])
    power = np.interp(v, data[:, 0], data[:, 1])
    thrust = np.interp(v, data[:, 0], data[:, 2])
    return power, thrust