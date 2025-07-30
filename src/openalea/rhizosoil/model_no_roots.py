# Soil model
from openalea.rhizosoil.soil_model_no_roots import RhizoSoil

# Utilities
from openalea.metafspm.composite_wrapper import CompositeModel
from openalea.metafspm.component_factory import Choregrapher


class RhizosphericSoil(CompositeModel):
    """
    Root-BRIDGES model

    Use guideline :
    1. store in a variable Model(g, time_step) to initialize the model, g being an openalea.MTG() object and time_step a time interval in seconds.

    2. print Model.documentation for more information about editable model parameters (optional).

    3. Use Model.scenario(**dict) to pass a set of scenario-specific parameters to the model (optional).

    4. Use Model.run() in a for loop to perform the computations of a time step on the passed MTG File
    """

    def __init__(self, time_step: int, scene_xrange: float, scene_yrange: float, **scenario):
        """
        DESCRIPTION
        ----------
        __init__ method of the model. Initializes the thematic modules and link them.

        :param g: the openalea.MTG() instance that will be worked on. It must be representative of a root architecture.
        :param time_step: the resolution time_step of the model in seconds.
        """
        # DECLARE GLOBAL SIMULATION TIME STEP, FOR THE CHOREGRAPHER TO KNOW IF IT HAS TO SUBDIVIDE TIME-STEPS
        Choregrapher().add_simulation_time_step(time_step)
        self.time = 0
        soil_parameters = scenario["parameters"]["soil_model"]["soil"]
        self.input_tables = scenario["input_tables"]

        # INIT INDIVIDUAL MODULES
        self.soil = RhizoSoil(time_step_in_seconds=time_step,
                                scene_xrange=scene_xrange, scene_yrange=scene_yrange, **soil_parameters)

        self.soil_voxels = self.soil.voxels

        # Manually assigning data structure for logger retreive
        self.declare_data(soil=self.soil_voxels)
        self.components = [self.soil]


    def run(self):
        self.apply_input_tables(tables=self.input_tables, to=self.components, when=self.time)
        # Update environment boundary conditions
        self.soil()

        self.time += 1

