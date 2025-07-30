# Public packages
import os, sys, time
import multiprocessing as mp
# Model packages
from openalea.rhizosoil.model_no_roots import RhizoSoil
# Utility packages
from openalea.fspm.utility.writer import Logger
from openalea.fspm.utility.plot import analyze_data
from openalea.fspm.utility.scenario import MakeScenarios as ms


def single_run(scenario, outputs_dirpath="outputs", simulation_length=2500, echo=True, log_settings={}, analyze=False):
    rhizosoil = RhizoSoil(time_step=3600, scene_xrange=0.15, scene_yrange=0.15, **scenario)

    logger = Logger(model_instance=rhizosoil, components=rhizosoil.components,
                    outputs_dirpath=outputs_dirpath, 
                    time_step_in_hours=1, logging_period_in_hours=24,
                    recording_shoot=False,
                    echo=echo, **log_settings)
    
    try:
        for _ in range(simulation_length):
            # Placed here also to capture mtg initialization
            logger()
            # logger.run_and_monitor_model_step()
            rhizosoil.run()

    except (ZeroDivisionError, KeyboardInterrupt):
        logger.exceptions.append(sys.exc_info())

    finally:
        logger.stop()
        if analyze:
            analyze_data(scenarios=[os.path.basename(outputs_dirpath)], outputs_dirpath=outputs_dirpath, target_properties=None, **log_settings)


def simulate_scenarios(scenarios, simulation_length=2500, echo=True, log_settings={}):
    processes = []
    max_processes = mp.cpu_count() - 1
    for scenario_name, scenario in scenarios.items():
        
        # Wait until there is a free slot
        while True:
            # Remove any finished processes and join them to release resources
            alive_processes = []
            for proc in processes:
                if proc.is_alive():
                    alive_processes.append(proc)
                else:
                    proc.join()
            processes = alive_processes

            if len(processes) < max_processes:
                break
            time.sleep(1)

        print(f"[INFO] Launching scenario {scenario_name}...")
        p = mp.Process(target=single_run, kwargs=dict(scenario=scenario, 
                                                      outputs_dirpath=os.path.join("outputs", str(scenario_name)),
                                                      simulation_length=simulation_length,
                                                      echo=echo,
                                                      log_settings=log_settings))
        p.start()
        processes.append(p)


if __name__ == '__main__':
    scenarios = ms.from_table(file_path="inputs/Scenarios_24_11_10.xlsx", which=["RhizoSoil_1"])
    simulate_scenarios(scenarios, simulation_length=2500, log_settings=Logger.heavy_log)
    