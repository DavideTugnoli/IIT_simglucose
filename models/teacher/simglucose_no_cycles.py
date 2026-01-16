## TEACHER MODEL (WITHOUT CYCLES)
from datetime import datetime, timedelta
import torch
from torch import nn
import numpy as np
from utils.t1dpatient_no_cycles import T1DPatient
from utils.env import T1DSimEnv
from utils.controller import BBController
from utils.pump import InsulinPump
from simglucose.sensor.cgm import CGMSensor
from simglucose.simulation.scenario import CustomScenario


class Simglucose(nn.Module):
    """
    """
    def __init__(
            self,
            pred_horizon,
            timeseries_iit = False
        ):
        super().__init__()
        # Oracle simulator
        self.model = self.simulator
        self.loss = nn.MSELoss(reduction='mean')
        # specify start_time as the beginning of today
        self.start_time = datetime(2024,2,14,8,0,0,0)
        self.pred_horizon = pred_horizon
        self.timeseries_iit = timeseries_iit
        self.time_env_sample = 3

    def simulator(self, 
                  pat_name, 
                  meal_size,
                  insulin_dosage,
                  pred_horizon,
                  # for interchange.
                  interchanged_variables=None,
                  variable_names=None,
                  interchanged_activations=None):
        # Create a simulation environment
        pat_name = pat_name.replace("_hyper", "").replace("_hypo", "")
        patient = T1DPatient.withName(name=pat_name)
        sensor = CGMSensor.withName('Dexcom', seed=1)
        pump = InsulinPump.withName('Insulet')
        # custom scenario is a list of tuples (time in hours, meal_size)
        scen = [(0.5, meal_size*self.time_env_sample)]
        scenario = CustomScenario(start_time=self.start_time, scenario=scen)
        env = T1DSimEnv(patient, sensor, pump, scenario)

        # Create a controller
        controller = BBController(insulin_dosage=insulin_dosage)

        controller.reset()
        obs, reward, done, info = env.reset()

        sim_time = timedelta(minutes=30 + self.pred_horizon)

        while env.time < scenario.start_time + sim_time:
            action = controller.policy(obs, reward, done, **info)
            obs, reward, done, info = env.step(action=action,
                                               pred_horizon=pred_horizon,
                                               interchanged_variables=interchanged_variables,
                                               variable_names=variable_names,
                                               interchanged_activations=interchanged_activations,
                                               timeseries_iit = self.timeseries_iit)
        return np.array(patient.state_hist), obs.CGM


    def forward(
        self,
        input_ids,
        labels=None,
        look_up = None,
        # for interchange.
        interchanged_variables=None, 
        variable_names=None,
        interchanged_activations=None
    ):
        """
        Inputs:
            input_ids: pre meal parameters
            labels: post meal parameters
            look_up: patient name
            interchanged_variables: alignment,
            variable_names: mapping
            interchanged_activations: values to interchange
        """

        teacher_ouputs = {}
        teacher_ouputs["hidden_states"]=[]
        # we perform the interchange intervention
        meal_size = float(input_ids[-11])
        insulin_dosage = float(input_ids[-12])
        x, output = self.simulator(look_up,
                           meal_size,
                           insulin_dosage,
                           self.pred_horizon,
                           variable_names=variable_names,
                           interchanged_variables=interchanged_variables,
                           interchanged_activations=interchanged_activations)

        if not self.timeseries_iit:
            # last state (13,)
            teacher_ouputs["hidden_states"] = x[-1]
        else:
            # x is np.array(state_hist) where state_hist[t] = state at minute t
            times = list(range(30, 30 + self.pred_horizon, 3))  # K points
            x_sel = x[times, :]  # (K, 13)
            teacher_ouputs["hidden_states"] = x_sel.reshape(-1)  # time-major: (t0 vars, t1 vars, ...)
        
        teacher_ouputs["outputs"]=torch.tensor(output*0.01, dtype=torch.float32)

        return teacher_ouputs
