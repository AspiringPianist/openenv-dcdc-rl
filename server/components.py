"""
Real Component Catalog for SpiceRL.

Parses Coilcraft LTSpice library files to build an in-memory catalog
of real Inductance (L) and associated DC Resistance (DCR).
Used in 'medium' and 'hard' tasks to snap agent actions to real physical part parameters.
"""

import glob
import re
import os
import bisect
from typing import Tuple

_REAL_INDUCTORS = []  # List of (L_nH, DCR_mOhm, Price_USD)
_REAL_CAPACITORS = [] # List of (C_nF, ESR_mOhm, Price_USD)

def _estimate_inductor_price(l_val: float, dcr_mohm: float) -> float:
    """Estimate a 1k-reel USD price based on inductance and DCR (thicker copper/larger core costs more)."""
    # Base ~ $0.05, scales with volume_factor (L / DCR)
    volume_factor = l_val / max(dcr_mohm, 0.1)
    price = 0.05 + (0.015 * volume_factor)
    return round(min(max(price, 0.02), 5.0), 3)

def _estimate_capacitor_price(c_nf: float, esr_mohm: float) -> float:
    """Estimate a 1k-reel USD price based on capacitance (more layers/larger package costs more)."""
    # Base MLCC ~ $0.01
    price = 0.01 + 0.02 * (c_nf / 1000.0) + (0.05 * (5.0 / max(esr_mohm, 0.5)))
    return round(min(max(price, 0.005), 3.0), 3)

def _load_coilcraft_catalog():
    if _REAL_INDUCTORS:
        return
        
    lib_pattern = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "templates", "models", "CoilcraftLTAdvLib", "**", "*.lib"
    )
    
    pattern = re.compile(r'\.subckt\s+.*?PARAMS:.*?Ind=([\d\.]+)([unm]H).*?\+ R1=.*?\+ R2=([\d\.]+)', re.DOTALL | re.IGNORECASE)
    
    for filepath in glob.glob(lib_pattern, recursive=True):
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                for match in pattern.finditer(content):
                    val_str, unit, r2_str = match.groups()
                    val = float(val_str)
                    
                    if unit.lower() == 'uh':
                        val *= 1000.0
                    elif unit.lower() == 'mh':
                        val *= 1000000.0
                        
                    dcr_mohm = float(r2_str) * 1000.0
                    price = _estimate_inductor_price(val, dcr_mohm)
                    _REAL_INDUCTORS.append((val, dcr_mohm, price))
        except Exception:
            pass
            
    _REAL_INDUCTORS.sort(key=lambda x: x[0])


def _load_murata_catalog():
    if _REAL_CAPACITORS:
        return
        
    lib_pattern = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "templates", "models", "murata-lib-ltspice-s-mlcc-*", "**", "*.lib"
    )
    
    # Murata format: Property : C = X.XXe-XX[uF]
    # Followed by a subcircuit with R03 or R3 ...
    for filepath in glob.glob(lib_pattern, recursive=True):
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
                
            blocks = text.split('.SUBCKT')[1:]
            for block in blocks:
                # Capture Capacitance from comments
                c_match = re.search(r'\*\s*Property\s*:\s*C\s*=\s*([\d\.eE\+\-]+)\[(.*?)\]', block)
                if not c_match:
                    continue
                
                c_val, unit = c_match.groups()
                c_val = float(c_val)
                unit = unit.lower()
                if unit == 'uf': c_nF = c_val * 1000.0
                elif unit == 'nf': c_nF = c_val
                elif unit == 'pf': c_nF = c_val / 1000.0
                elif unit == 'f': c_nF = c_val * 1e9
                else: continue
                
                # Match series resistance (typically R03 inside the C-L-R branching network)
                r_match = re.search(r'R03\s+\S+\s+\S+\s+([\d\.eE\+\-]+)', block)
                if r_match and c_nF > 0:
                    esr_mohm = float(r_match.group(1)) * 1000.0
                    
                    # C filter: Only care about > 0.1nF up to 10uF since this is a buck filter app
                    if c_nF >= 0.1 and c_nF <= 100000.0:
                        price = _estimate_capacitor_price(c_nF, esr_mohm)
                        _REAL_CAPACITORS.append((c_nF, esr_mohm, price))
        except Exception:
            pass
            
    _REAL_CAPACITORS.sort(key=lambda x: x[0])


def get_closest_inductor(requested_l_nH: float) -> Tuple[float, float, float]:
    """Snap a requested L to the closest real part and return (L_nH, DCR_mOhm, Price_USD)."""
    _load_coilcraft_catalog()
    
    if not _REAL_INDUCTORS:
        return (requested_l_nH, 50.0, 0.15)  # Fallback if catalog not found
        
    keys = [x[0] for x in _REAL_INDUCTORS]
    idx = bisect.bisect_left(keys, requested_l_nH)
    
    if idx == 0: return _REAL_INDUCTORS[0]
    if idx == len(_REAL_INDUCTORS): return _REAL_INDUCTORS[-1]
        
    before, after = _REAL_INDUCTORS[idx - 1], _REAL_INDUCTORS[idx]
    if after[0] - requested_l_nH < requested_l_nH - before[0]:
        return after
    return before


def get_closest_capacitor(requested_c_nF: float) -> Tuple[float, float, float]:
    """Snap a requested C to the closest real Murata MLCC part and return (C_nF, ESR_mOhm, Price_USD)."""
    _load_murata_catalog()
    
    if not _REAL_CAPACITORS:
        return (requested_c_nF, 15.0, 0.05)  # Fallback
        
    keys = [x[0] for x in _REAL_CAPACITORS]
    idx = bisect.bisect_left(keys, requested_c_nF)
    
    if idx == 0: return _REAL_CAPACITORS[0]
    if idx == len(_REAL_CAPACITORS): return _REAL_CAPACITORS[-1]
        
    before, after = _REAL_CAPACITORS[idx - 1], _REAL_CAPACITORS[idx]
    if after[0] - requested_c_nF < requested_c_nF - before[0]:
        return after
    return before

