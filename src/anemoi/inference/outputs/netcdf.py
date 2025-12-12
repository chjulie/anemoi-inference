# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
import os
import threading
from pathlib import Path

import numpy as np
from datetime import datetime

from anemoi.inference.context import Context
from anemoi.inference.types import ProcessorConfig
from anemoi.inference.types import State

from ..decorators import ensure_path
from ..decorators import main_argument
from ..output import Output
from . import output_registry

LOG = logging.getLogger(__name__)


# In case HDF5 was not compiled with thread safety on
LOCK = threading.RLock()


@output_registry.register("netcdf")
@main_argument("path")
@ensure_path("path")
class NetCDFOutput(Output):
    """NetCDF output class."""

    def __init__(
        self,
        context: Context,
        path: Path,
        variables: list[str] | None = None,
        post_processors: list[ProcessorConfig] | None = None,
        output_frequency: int | None = None,
        write_initial_state: bool | None = None,
        float_size: str = "f4",
        missing_value: float | None = np.nan,
    ) -> None:
        """Initialise the NetCDF output object.

        Parameters
        ----------
        context : dict
            The context dictionary.
        path : Path
            The path to save the NetCDF file to.
            If the parent directory does not exist, it will be created.
        post_processors : Optional[List[ProcessorConfig]], default None
            Post-processors to apply to the input
        output_frequency : int, optional
            The frequency of output, by default None.
        write_initial_state : bool, optional
            Whether to write the initial state, by default None.
        float_size : str, optional
            The size of the float, by default "f4".
        missing_value : float, optional
            The missing value, by default np.nan.
        """

        super().__init__(
            context,
            variables=variables,
            post_processors=post_processors,
            output_frequency=output_frequency,
            write_initial_state=write_initial_state,
        )

        from netCDF4 import Dataset

        self.path = path
        self.ncfile: Dataset | None = None
        self.float_size = float_size
        self.missing_value = missing_value
        self.initial_state_date = None
        self.current_initial_date_index = 0  # Track the current initial_date index
        self._active_initial_date: datetime | None = None
        if self.write_initial_state:
            self.extra_time = 1
        else:
            self.extra_time = 0

    def __repr__(self) -> str:
        """Return a string representation of the NetCDFOutput object."""
        return f"NetCDFOutput({self.path})"

    def open(self, state: State) -> None:
        """Open the NetCDF file and initialize dimensions and variables.

        Parameters
        ----------
        state : State
            The state dictionary.
        """
        from netCDF4 import Dataset

        with LOCK:
            if self.ncfile is not None:
                return

        # If the file exists, we may get a 'Permission denied' error
        if os.path.exists(self.path):
            os.remove(self.path)

        with LOCK:
            self.ncfile = Dataset(self.path, "w", format="NETCDF4")

        state = self.post_process(state)

        compression = {}  # dict(zlib=False, complevel=0)

        values = len(state["latitudes"])

        time = 0
        # self.reference_date = state["date"]
        if (time_step := getattr(self.context, "time_step", None)) and (
            lead_time := getattr(self.context, "lead_time", None)
        ):
            time = lead_time // time_step
            time += self.extra_time
        
        LOG.info(f"⏰ TIME: {time}")
        LOG.info(f"⏰ EXTRA TIME: {self.extra_time}")

        if reference_date := getattr(self.context, "date", None):
            self.reference_date = reference_date
        LOG.info(f"🚧🚧🚧🚧🚧🚧 open(): SELF.REFERENCE_DATE[0]: {self.reference_date}")

        with LOCK:
            # dimensions
            self.values_dim = self.ncfile.createDimension("values", values)
            self.lead_time_dim = self.ncfile.createDimension("lead_time", time)
            self.initial_date_dim = self.ncfile.createDimension("initial_date", None)

            # lead_time var
            self.lead_time_var = self.ncfile.createVariable("lead_time", "i4", ("lead_time",), **compression)
            self.lead_time_var.units = ""
            self.lead_time_var.long_name = "lead_time"

            # initial_date var
            # Store epoch seconds as int64 to avoid nanosecond overflow in decoders
            self.initial_date_var = self.ncfile.createVariable("initial_date", "i4", ("initial_date",), **compression)
            self.initial_date_var.units = f"seconds since {self.reference_date[0]}"
            self.initial_date_var.long_name = "initial_date"
            self.initial_date_var.calendar = "gregorian"
            LOG.info(f"🚧🚧🚧🚧🚧🚧 open(): INITIAL DATE UNITS: seconds since {self.reference_date[0]}")

        # Pre-fill the lead_time values (in hours)
        # lead_times = [(i * time_step).total_seconds() / 3600 for i in range(time)]
        lead_times = [6 * i for i in range(time - (1 - self.extra_time))]
        self.lead_time_var[:] = lead_times

        LOG.info(f"⏰ LEAD TIMES: {lead_times}")

        with LOCK:
            latitudes = state["latitudes"]

            self.latitude_var = self.ncfile.createVariable("latitude", self.float_size, ("values",), **compression)
            self.latitude_var.units = "degrees_north"
            self.latitude_var.long_name = "latitude"

            longitudes = state["longitudes"]

            self.longitude_var = self.ncfile.createVariable("longitude", self.float_size, ("values",), **compression)
            self.longitude_var.units = "degrees_east"
            self.longitude_var.long_name = "longitude"

            self.latitude_var[:] = latitudes
            self.longitude_var[:] = longitudes

        self.vars = {}

        self.n = 0      # counter for lead_time steps

    def ensure_variables(self, state: State) -> None:
        """Ensure that all variables are created in the NetCDF file.

        Parameters
        ----------
        state : State
            The state dictionary.
        """

        values = len(state["latitudes"])

        compression = {}  # dict(zlib=False, complevel=0)

        for name in state["fields"].keys():
            if self.skip_variable(name):
                continue

            if name in self.vars:
                continue

            chunksizes = (1, 1, values)

            while np.prod(chunksizes) > 1000000:
                chunksizes = tuple(int(np.ceil(x / 2)) for x in chunksizes)

            with LOCK:
                missing_value = self.missing_value

                self.vars[name] = self.ncfile.createVariable(
                    name,
                    self.float_size,
                    ("initial_date", "lead_time", "values"),
                    chunksizes=chunksizes,
                    fill_value=missing_value,
                    **compression,
                )

                self.vars[name].fill_value = missing_value
                self.vars[name].missing_value = missing_value

    def write_state(self, state: State, initial_date: datetime) -> None:
        """Write the state.

        Parameters
        ----------
        state : State
            The state to write.
        """
        step = state["step"]
        if self.output_frequency is not None:
            if (step % self.output_frequency).total_seconds() != 0:
                return
            
        LOG.info(f"🚧🚧🚧🚧🚧🚧 write_state(): INITIAL DATE: {initial_date}")
        return self.write_step(self.post_process(state), initial_date)

    def write_step(self, state: State, initial_date: datetime) -> None:
        """Write the state.

        Parameters
        ----------
        state : State
            The state dictionary.
        """

        self.ensure_variables(state)

        # step = state["date"] - self.reference_date
        # self.time_var[self.n] = step.total_seconds()

        # Write the initial_date only when it changes; keep lead_time counter for the same date
        if self._active_initial_date != initial_date:
            self._active_initial_date = initial_date
            with LOCK:
                step = initial_date - self.reference_date[0]
                LOG.info(f"🚧🚧🚧🚧🚧🚧 write_step(): INITIAL DATE: {initial_date}")
                LOG.info(f"🚧🚧🚧🚧🚧🚧 write_step(): SELF.REFERENCE_DATE[0]: {self.reference_date[0]}")
                LOG.info(f"🚧🚧🚧🚧🚧🚧 write_step(): STEP: {step}")
                self.initial_date_var[self.current_initial_date_index] = step.total_seconds()
            self.n = 0

        for name, value in state["fields"].items():
            if self.skip_variable(name):
                continue

            with LOCK:
                LOG.debug(f"🚧🚧🚧🚧🚧🚧 XXXXXX {name}, {self.n}, {value.shape}")
                self.vars[name][self.current_initial_date_index, self.n, :] = value

        self.n += 1
        if self.n >= len(self.lead_time_var):
            self.n = 0
            self.current_initial_date_index += 1
            self._active_initial_date = None

    def close(self) -> None:
        """Close the NetCDF file."""
        if self.ncfile is not None:
            with LOCK:
                self.ncfile.close()
            self.ncfile = None
