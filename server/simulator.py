"""
SpiceRL Simulator — Wrapper around spicelib for netlist editing,
simulation execution, and metric extraction.

Supports both LTspice (local development) and NGSpice (Docker deployment).

AC analysis uses a separate replica netlist with small-signal injection
sources already wired in, matching how power electronics engineers
manually do loop-gain measurements in LTspice.
"""

import logging
import os
import re
import shutil
import tempfile
import threading
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def get_simulator_class():
    """Return the appropriate spicelib simulator class based on SPICE_BACKEND env var."""
    backend = os.getenv("SPICE_BACKEND", "ngspice").lower()

    if backend == "ltspice":
        from spicelib.simulators.ltspice_simulator import LTspice
        return LTspice
    elif backend == "qspice":
        from spicelib.simulators.qspice_simulator import Qspice
        return Qspice
    else:  # default: ngspice
        from spicelib.simulators.ngspice_simulator import NGspiceSimulator
        return NGspiceSimulator


class SpiceSimulator:
    """Wraps spicelib for the SpiceRL environment.

    Handles:
    1. Loading a template .net file
    2. Applying agent-chosen parameter values
    3. Running transient simulation
    4. (Optionally) running AC simulation on a replica netlist
    5. Extracting performance metrics from .log and .raw files
    """

    def __init__(self, output_folder: str = "./sim_output"):
        self.output_folder = output_folder
        os.makedirs(output_folder, exist_ok=True)
        self._sim_class = None  # lazy-loaded

    @property
    def simulator(self):
        if self._sim_class is None:
            self._sim_class = get_simulator_class()
        return self._sim_class

    def run_simulation(
        self,
        template_path: str,
        params: Dict[str, float],
        run_name: str = "sim",
        ac_template_path: str = "",
    ) -> Dict[str, Any]:
        """Run a complete simulation cycle and return extracted metrics.

        Args:
            template_path: Path to the transient .net template
            params: Dict of parameter name -> value to set
            run_name: Unique name for this simulation run
            ac_template_path: Path to AC analysis template (empty = skip AC)

        Returns:
            Dict of extracted metrics
        """
        metrics: Dict[str, Any] = {
            "sim_converged": True,
            "sim_error": None,
        }

        # --- Run transient simulation ---
        try:
            tran_metrics = self._run_single_sim(template_path, params, f"{run_name}_tran")
            metrics.update(tran_metrics)
        except Exception as e:
            logger.error(f"Transient simulation failed: {e}")
            metrics["sim_error"] = str(e)
            metrics["sim_converged"] = False
            return metrics

        # --- Run AC simulation (separate replica netlist) ---
        if ac_template_path and os.path.exists(ac_template_path):
            try:
                ac_metrics = self._run_single_sim(
                    ac_template_path, params, f"{run_name}_ac"
                )
                # Only take AC-specific keys
                for key in ["phase_margin_deg", "gain_margin_dB", "crossover_freq_kHz"]:
                    if key in ac_metrics:
                        metrics[key] = ac_metrics[key]
            except Exception as e:
                logger.warning(f"AC simulation failed (non-fatal): {e}")
                # AC failure is non-fatal, transient results still count

        return metrics

    def _run_single_sim(
        self,
        template_path: str,
        params: Dict[str, float],
        run_name: str,
        timeout_secs: int = 300,
    ) -> Dict[str, Any]:
        """Run a single simulation (transient or AC) and return metrics."""
        import time
        from spicelib import SpiceEditor, SimRunner

        # Copy model/include files to output folder so relative paths resolve
        template_dir = os.path.dirname(os.path.abspath(template_path))
        models_src = os.path.join(template_dir, "models")
        models_dst = os.path.join(self.output_folder, "models")
        if os.path.isdir(models_src) and not os.path.isdir(models_dst):
            shutil.copytree(models_src, models_dst)
        elif os.path.isdir(models_src):
            for fname in os.listdir(models_src):
                src_f = os.path.join(models_src, fname)
                dst_f = os.path.join(models_dst, fname)
                if os.path.isfile(src_f):
                    shutil.copy2(src_f, dst_f)

        # Load template and apply parameters
        net = SpiceEditor(template_path)

        for param_name, value in params.items():
            try:
                net.set_parameters(**{param_name: value})
            except Exception as e:
                logger.warning(f"Could not set param {param_name}={value}: {e}")

        # Create a SimRunner and run
        runner = SimRunner(
            output_folder=self.output_folder,
            simulator=self.simulator,
            parallel_sims=1,
        )

        run_filename = f"{run_name}.net"
        print(f"  [SIM] Running {run_name}...", flush=True)
        t0 = time.time()

        # Print a live spinner with elapsed time so long LTspice calls
        # don't appear frozen in the terminal.
        stop_progress = threading.Event()

        def _progress_worker() -> None:
            last_bucket = -1
            while not stop_progress.wait(1.0):
                elapsed_now = time.time() - t0
                bucket = int(elapsed_now // 5)
                if bucket == last_bucket:
                    continue
                last_bucket = bucket

                if timeout_secs:
                    pct_of_timeout = min((elapsed_now / timeout_secs) * 100.0, 99.9)
                    msg = (
                        f"  [SIM] {run_name} still running: "
                        f"{elapsed_now:6.1f}s elapsed ({pct_of_timeout:4.1f}% timeout)"
                    )
                else:
                    msg = f"  [SIM] {run_name} still running: {elapsed_now:6.1f}s elapsed"
                print(msg, flush=True)

        progress_thread = threading.Thread(target=_progress_worker, daemon=True)
        progress_thread.start()

        try:
            try:
                raw_file, log_file = runner.run_now(
                    net, run_filename=run_filename, timeout=timeout_secs
                )
            except TypeError:
                # Older spicelib versions may not support timeout kwarg
                raw_file, log_file = runner.run_now(net, run_filename=run_filename)
        finally:
            stop_progress.set()
            progress_thread.join(timeout=2.0)
            print(flush=True)

        elapsed = time.time() - t0
        print(f"  [SIM] {run_name} completed in {elapsed:.1f}s", flush=True)

        # Check for .fail file indicating simulation failure
        fail_path = os.path.join(
            self.output_folder, run_filename.replace(".net", ".fail")
        )
        if os.path.exists(fail_path):
            with open(fail_path, "r", errors="replace") as f:
                fail_msg = f.read().strip()
            print(f"  [SIM] FAILED: {fail_msg}", flush=True)
            return {"sim_error": fail_msg, "sim_converged": False}

        # Extract metrics from log file (.MEAS results)
        metrics = self._parse_log_metrics(log_file)

        # If needed, also extract from raw file
        if raw_file and os.path.exists(str(raw_file)):
            raw_metrics = self._parse_raw_metrics(str(raw_file))
            metrics.update(raw_metrics)

        n_metrics = len([k for k in metrics if k not in ("sim_error", "sim_converged")])
        print(f"  [SIM] Extracted {n_metrics} metrics from log", flush=True)

        return metrics

    def _parse_log_metrics(self, log_file) -> Dict[str, Any]:
        """Extract .MEAS directive results from the simulation log file.

        .MEAS results appear in the log as lines like:
            vout_avg: AVG(v(vout))=4.98756 FROM 0.0005 TO 0.001
            vout_ripple: PP(v(vout))=0.0234 FROM 0.0005 TO 0.001
        """
        metrics: Dict[str, Any] = {}

        if not log_file or not os.path.exists(str(log_file)):
            return metrics

        log_reader_cls = None
        try:
            # spicelib >= 1.4.x
            from spicelib.log.ltsteps import LTSpiceLogReader
            log_reader_cls = LTSpiceLogReader
        except Exception:
            try:
                # Older spicelib path
                from spicelib.log.ltspice_log_reader import LTSpiceLogReader
                log_reader_cls = LTSpiceLogReader
            except Exception:
                log_reader_cls = None

        if log_reader_cls is not None:
            try:
                log_reader = log_reader_cls(str(log_file))
                # spicelib API differs across versions:
                # - Some parse measures in __init__.
                # - Some expose read_measures()/get_measures().
                if hasattr(log_reader, "read_measures"):
                    log_reader.read_measures()
                elif hasattr(log_reader, "get_measures"):
                    log_reader.get_measures()

                for meas_name, values in log_reader.dataset.items():
                    name = meas_name.lower()
                    if values and len(values) > 0:
                        metrics[name] = float(values[0])
            except Exception as e:
                logger.warning(f"LTSpiceLogReader parsing failed, falling back to regex: {e}")
                metrics = self._parse_log_regex(str(log_file))
        else:
            # Fallback: parse log file manually when reader class is unavailable.
            metrics = self._parse_log_regex(str(log_file))

        return metrics

    def _parse_log_regex(self, log_path: str) -> Dict[str, Any]:
        """Fallback regex parser for .MEAS results in log files."""
        metrics = {}
        # Pattern matches: "measurement_name: ... = value ..."
        pattern = re.compile(
            r"^\s*(\w+):\s+.*?=\s*([\d.eE+\-]+)", re.MULTILINE
        )

        try:
            with open(log_path, "r", errors="replace") as f:
                content = f.read()
            for match in pattern.finditer(content):
                name = match.group(1).lower()
                value = float(match.group(2))
                metrics[name] = value
        except Exception as e:
            logger.error(f"Regex log parsing failed: {e}")

        return metrics

    def _parse_raw_metrics(self, raw_path: str) -> Dict[str, Any]:
        """Extract additional metrics from .raw waveform files.

        Only used when .MEAS results are insufficient (e.g., for efficiency
        calculation that requires integrating power waveforms).
        """
        metrics = {}

        try:
            from spicelib import RawRead
            raw = RawRead(raw_path)
            trace_names = raw.get_trace_names()

            # Try to get time axis and output voltage for basic checks
            time = raw.get_axis(0)
            if time is not None and len(time) > 0:
                # Check if V(vout) exists
                vout_trace_name = None
                for name in trace_names:
                    if "vout" in name.lower():
                        vout_trace_name = name
                        break

                if vout_trace_name:
                    vout = raw.get_trace(vout_trace_name).get_wave(0)
                    vout = np.array(vout, dtype=float)
                    time = np.array(time, dtype=float)

                    # Use the last 20% of the sim for steady-state analysis
                    ss_start = int(len(time) * 0.8)
                    vout_ss = vout[ss_start:]

                    # Only overwrite if not already from .MEAS
                    if "vout_avg" not in metrics:
                        metrics["vout_avg"] = float(np.mean(vout_ss))
                    if "vout_ripple" not in metrics:
                        metrics["vout_ripple"] = float(np.ptp(vout_ss))

        except Exception as e:
            msg = str(e).lower()
            if "does not have an axis" in msg:
                logger.info("Raw file has no axis; skipping raw metric extraction.")
            else:
                logger.warning(f"Raw file parsing failed (non-fatal): {e}")

        return metrics

    def validate_params(
        self,
        params: Dict[str, float],
        bounds: Dict[str, tuple],
    ) -> Tuple[Dict[str, float], list]:
        """Validate and clamp parameter values to physical bounds.

        Returns:
            (clamped_params, warnings_list)
        """
        clamped = {}
        warnings = []

        for name, value in params.items():
            if name in bounds:
                lo, hi = bounds[name]
                if value < lo:
                    warnings.append(f"{name}={value} clamped to min {lo}")
                    value = lo
                elif value > hi:
                    warnings.append(f"{name}={value} clamped to max {hi}")
                    value = hi
            clamped[name] = value

        return clamped, warnings
