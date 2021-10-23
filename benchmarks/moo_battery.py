# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 14:28:17 2021
PyBaMM functions for loading parameters from sets and calculatuing specific power and
energy.
@author: tom
"""

import sys
import pybamm
from importlib import import_module
import numpy as np
import matplotlib.pyplot as plt
import signal
from contextlib import contextmanager

# load model equations - SPMe is good compromise between speed and accuracy
# Likely to get more solving issues with DFN and not enough accuracy with SPM
model = pybamm.lithium_ion.SPMe()
# Set restricted voltage limit window - used later optionally
V_low = 3.20
V_high = 4.0


def calc_volume_and_area(param):
    r'''
    PyBaMM only models single layers of cells however the thermal models can use
    the cell volume which may be made up of many layers - See Ai2020 which has a 30x
    difference in thickness between single layer and cell.
    This may be something we need to look at more closely
    '''
    thickness = [
        'Negative current collector thickness [m]',
        'Negative electrode thickness [m]',
        'Positive current collector thickness [m]',
        'Positive electrode thickness [m]',
        'Separator thickness [m]'
    ]
    t = 0.0
    for var in thickness:
        t += param[var]
    h = param['Electrode height [m]']
    w = param['Electrode width [m]']
    a_calc = h * w
    v_calc = a_calc * t
    return v_calc, a_calc


def scale_volume_parameters(param):
    r'''
    Calculate volume based on single layer cell and adjust the Nominal capacity for
    good measure
    '''
    stated_cell_volume = param['Cell volume [m3]']
    calculated_vol, _ = calc_volume_and_area(param)
    volume_scaling = calculated_vol / stated_cell_volume
    update_params = [
        'Cell volume [m3]',
        'Nominal cell capacity [A.h]',
    ]
    for upd in update_params:
        param[upd] = param[upd] * volume_scaling
    param['Number of electrodes connected in parallel to make a cell'] = 1


def normalize_areas(param, value=25e-2):
    r'''
    Change the unimportant dimensions so have length of fixed value, resize the volume
    and capacity and typical current.
    '''
    volume, area = calc_volume_and_area(param)
    param['Electrode height [m]'] = value
    param['Electrode width [m]'] = value
    new_volume, new_area = calc_volume_and_area(param)
    param['Cell volume [m3]'] = volume
    area_scaling = new_area / area
    update_params = [
        'Cell volume [m3]',
        'Nominal cell capacity [A.h]',
        'Typical current [A]'
    ]
    for upd in update_params:
        param[upd] = param[upd] * area_scaling


def get_discharged_capacity(solution):
    r'''
    For a solution - assuming this is a single step get the time interval and
    multiply by the current if fixed or integrate if variable. Should always be fixed
    But you might change protocol later
    '''
    time = solution['Time [s]'].entries
    current = solution['Current [A]'].entries
    tot_time = time[-1] - time[0]
    if np.all(current) == current[0]:
        capacity = current[0] * tot_time
    else:
        dt = time[1:] - time[:-1]
        capacity = np.sum(current[:1] * dt)
    return capacity / 3600


def get_discharged_energy(solution):
    r'''
    Return the final discharged energy value from the solution
    N.B assumes solution is a single cycle

    Parameters
    ----------
    solution : TYPE
        DESCRIPTION.

    Returns
    -------
    float

    '''
    energy = solution['Discharge energy [W.h]'].entries
    return energy[-1] - energy[0]


def get_mean_power(solution):
    r'''
    Return the mean power for a given solution
    '''
    power = solution['Terminal power [W]'].entries
    return np.mean(power)


def get_voltage_limits(param):
    r'''
    Accespts a parameter set and gives back voltage limits for simulation
    These are the manufacturer safe limits but usually they also specify a max current
    For higher currents need to operate cell in a more restricted window
    '''
    V_min = param['Lower voltage cut-off [V]']
    V_max = param['Upper voltage cut-off [V]']

    return V_min, V_max


def run_cycles(param, set_name, c_rates, voltage_limits="max"):
    r'''
    Calculate specific energy and power for a parameter set over a list of c_rates
    within a voltage_limit

    param : pybamm.ParameterValues

    c_rate : list or array e.g. [1, 2, 3]

    voltage_limits : string or list
        default = "max" use parameter set limits
        "restricted" use V_low and V_high (set at top of script)
        or supply own e.g. [3.2, 4.0]
    '''
    # print('Solving model for parameter set: ', set_name)
    V_min, V_max = get_voltage_limits(param)
    volume, area = calc_volume_and_area(param)
    cycle_e = []
    cycle_p = []
    for i, c_rate in enumerate(c_rates):
        # print('Running c_rate', c_rate)
        sol = run_capacity_check(param, c_rate, voltage_limits)
        if sol is None or sol == "failed":
            # The C_rate is infeasible with this parameter set
            break
        # print('Solution termination', sol.termination)
        cycle_e.append(get_discharged_energy(sol))
        cycle_p.append(get_mean_power(sol))

    cycle_e = np.asarray(cycle_e)
    cycle_p = np.asarray(cycle_p)
    summary = np.vstack((cycle_p, cycle_e)).T
    # Calculate specific energy and power
    summary /= (1e6 * volume)
    return summary


def run_capacity_check(param, c_rate=1.0, voltage_limits='max'):
    r'''
    Calculate solution for a parameter set for one C rate within a voltage_limit

    param : pybamm.ParameterValues

    c_rate : float
        default = 1.0

    voltage_limits : string or list
        default = "max" use parameter set limits
        "restricted" use V_low and V_high (set at top of script)
        or supply own e.g. [3.2, 4.0]
    '''
    V_min, V_max = get_voltage_limits(param)
    if voltage_limits == 'max':
        instr = [
            f"Charge at C/100 until {V_max}V (5 minute period)",
            f"Discharge at {c_rate}C until {V_min}V (10 second period)",
        ]
    elif voltage_limits == 'restricted':
        instr = [
            f"Charge at C/100 until {V_high}V (5 minute period)",
            f"Discharge at {c_rate}C until {V_low}V (10 second period)",
        ]
    elif len(voltage_limits) == 2:
        vlow, vhigh = voltage_limits
        instr = [
            f"Charge at C/100 until {vhigh}V (5 minute period)",
            f"Discharge at {c_rate}C until {vlow}V (10 second period)",
        ]
    else:
        instr = [
            f"Charge at C/100 until {V_max}V (5 minute period)",
            f"Discharge at {c_rate}C until {V_min}V (10 second period)",
        ]

    experiment = pybamm.Experiment(instr)

    # define timeout function, to automatically terminate function call
    # if pybamm gets stuck and doesn't terminate by itself
    class TimeoutException(Exception):
        pass

    @contextmanager
    def time_limit(seconds):
        def signal_handler(signum, frame):
            signal.alarm(2)
            raise TimeoutException("Timed out!")

        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            # signal.alarm(seconds)
            signal.alarm(0)

    try:
        print(f"* * * simulation is running...")
        simulation = pybamm.Simulation(model=model, experiment=experiment,
                                       parameter_values=param)

        with time_limit(20):
            simulation.solve(calc_esoh=False)

    except TimeoutException as e:
        print("    -> simulation timed out!")
        return "failed"
    except:
        return "failed"

    sol = simulation.solution
    try:
        cycle = sol.cycles[1]
    except:
        cycle = None

    return cycle


def standardize_sizing_and_SOC(parameters):
    r'''
    Standardize the cross-sectional area, nominal capacity, typical current and set
    # initial SOC to 0.5

    This could be done once and parameters pickled for use later multiple times
    as the process is a little long

    Parameters
    ----------
    parameters : dict
        The whole range of parameters.

    Returns
    -------
    None.

    '''
    # Another parameter which is used for scaling is the nominal capacity, this is
    # Specified by the user and will not update by changing the other parameters so
    # Need to be a bit careful here
    # Make sure the nominal capacity matches the 1C rate capacity
    # And adjust the Typical current [A] parameter also
    set_names = list(parameters.keys())
    for set_name in set_names:
        scale_volume_parameters(parameters[set_name])
        normalize_areas(parameters[set_name], value=5e-2)

    nom_cap = 'Nominal cell capacity [A.h]'
    # print('*' * 30)
    # print('Standardizing cell sizing and intial SOC')
    # print('Cell Capacity - Nominal and Calculated')
    cap_sols = []
    cap_sols_corrected = []
    for set_name in set_names:
        # print(set_name)
        nc = parameters[set_name][nom_cap]  # A.h
        volume, area = calc_volume_and_area(parameters[set_name])
        nc_scaled = np.around(nc / (1e6 * volume), 3)
        # print('---', 'nominal', ':', nc_scaled, '[mAh/cm3]')
        cap_sol = run_capacity_check(parameters[set_name])

        # return False if simulation failed
        if cap_sol == "failed" or not cap_sol:
            return False

        cc = get_discharged_capacity(cap_sol)
        cc_scaled = np.around(cc / (1e6 * volume), 3)
        # print('---', 'calculated', cc_scaled, '[mAh/cm3]')
        # print('Updating Nominal Capacity and Typical Current')
        parameters[set_name][nom_cap] = cc
        parameters[set_name]['Typical current [A]'] = cc
        cap_sols.append(cap_sol)
        # Run simulation again to check that the corrected capacity results in
        # A discharge of 1 hour as we are running at 1C
        cap_sol = run_capacity_check(parameters[set_name])

        # return False if simulation failed
        if cap_sol == "failed" or not cap_sol:
            return False

        cap_sols_corrected.append(cap_sol)

    return True

    fig, axes = plt.subplots(1, 2, figsize=[10, 5], sharey=True)
    for i in range(len(cap_sols)):
        for j in range(2):
            if j == 0:
                sol = cap_sols[i]
            else:
                sol = cap_sols_corrected[i]
            V = sol['Terminal voltage [V]'].entries
            t = sol['Time [h]'].entries
            t -= t[0]
            axes[j].plot(t, V, label=set_names[i])
    plt.legend()

    axes[0].set_ylabel('Terminal voltage [V]')
    axes[0].set_xlabel('Time [h]')
    axes[1].set_xlabel('Time [h]')
    axes[0].title.set_text('Uncorrected Nominal Capacity')
    axes[1].title.set_text('Corrected Nominal Capacity')

    plt.tight_layout()

    # Start parameter sets at 0.5 SOC using the average concentrations from the 1C run
    fig, axes = plt.subplots(1, 2, figsize=[10, 5], sharey=True)
    for i, sol in enumerate(cap_sols_corrected):
        param = parameters[set_names[i]]
        neg_conc = sol['Average negative particle concentration [mol.m-3]'].entries
        pos_conc = sol['Average positive particle concentration [mol.m-3]'].entries
        param['Initial concentration in negative electrode [mol.m-3]'] = np.mean(neg_conc)
        param['Initial concentration in positive electrode [mol.m-3]'] = np.mean(pos_conc)
        t = sol['Time [h]'].entries
        t -= t[0]
        axes[0].plot(t, neg_conc, label=set_names[i])
        axes[1].plot(t, pos_conc, label=set_names[i])
    plt.legend()
    axes[0].set_ylabel('Average particle concentration [mol.m-3]')
    axes[0].set_xlabel('Time [h]')
    axes[1].set_xlabel('Time [h]')
    axes[0].title.set_text('Negative')
    axes[1].title.set_text('Positive')
    plt.tight_layout()

def run_battery_simulation(X):
    param_dict = {
        0: 'Ai2020',
        1: 'Chen2020',
        2: 'Ecker2015',
        3: 'Marquis2019',
        4: 'Yang2017'
    }

    # check which set is proposed and take c_rate
    set_name = param_dict[X[0]]
    c_rate = X[1]

    # load all the parameters
    mod = import_module("pybamm.parameters.parameter_sets")
    met = getattr(mod, set_name)
    set_details = {}
    set_details[set_name] = met

    parameters = {}
    # print('Parameter Set:', set_name)
    parameters[set_name] = pybamm.ParameterValues(chemistry=set_details[set_name])

    # update parameters according to X
    parameters[set_name].update(
        {
            'Negative electrode porosity': X[2],
            'Negative electrode active material volume fraction': X[3],
            'Negative particle radius [m]': X[4],

            'Positive electrode porosity': X[5],
            'Positive electrode active material volume fraction': X[6],
            'Positive particle radius [m]': X[7],
         }
    )

    # update size parameters
    neg_vol_params = [
        # 'Negative current collector thickness [m]',
        'Negative electrode thickness [m]']
    for p in neg_vol_params:
        parameters[set_name].update(
            {p: parameters[set_name][p]*X[8]})

    pos_vol_params = [
        # 'Positive current collector thickness [m]',
        'Positive electrode thickness [m]']
    for p in pos_vol_params:
        parameters[set_name].update(
            {p: parameters[set_name][p]*X[9]})

    # standardize sizing, i.e. returns False if the simulation fails
    if not standardize_sizing_and_SOC(parameters):
        return 0.0, 0.0

    # compute energy and power of battery
    cycle_summary = {}
    param = parameters[set_name]
    summary = run_cycles(param, set_name, [c_rate], voltage_limits='max')

    cycle_summary[set_name] = summary

    # return the results of the simulation, i.e. returns (0.0, 0.0) if failed
    if not cycle_summary[set_name][:, 0] or \
        not cycle_summary[set_name][:, 1]:
        return 0.0, 0.0
    else:
        y1 = -cycle_summary[set_name][:, 0][0]
        y2 = -cycle_summary[set_name][:, 1][0]
        return y1, y2