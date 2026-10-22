
import re
import sys
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar, brentq
import scipy.optimize as opt
from uo import generate_sankey_diagram, pure_exergy, equipment_cost, _lmtd

# All cycle thermodynamics below use IDAES property packages instead of
# CoolProp -- Span-Wagner (swco2) for CO2, IAPWS-95 for water, and a
# Peng-Robinson (Generic/Cubic) mixture package for the flue gas -- ported
# from "Costing 5.ipynb"'s sco2 class. IDAES is used for BOTH the cycle
# physics (this section) and the turbine/pump purchased-equipment costing
# (further below).
from pyomo.environ import ConcreteModel, SolverFactory, value, units, Var, Constraint
from idaes.core import FlowsheetBlock, UnitModelCostingBlock
import idaes.logger as idaeslog
from idaes.models.properties.swco2 import SWCO2ParameterBlock, htpx
from idaes.models.properties import iapws95
from idaes.models.properties.iapws95 import htpx as htpxwater
from idaes.models.unit_models.pressure_changer import (
    PressureChanger,
    ThermodynamicAssumption,
)
from idaes.core import VaporPhase, Component
from idaes.core.base.phases import PhaseType as PT
from idaes.models.properties.modular_properties.base.generic_property import GenericParameterBlock
from idaes.models.properties.modular_properties.state_definitions import FTPx
from idaes.models.properties.modular_properties.eos.ceos import Cubic, CubicType
from idaes.models.properties.modular_properties.pure import NIST
from idaes.models_extra.power_generation.costing.power_plant_capcost import (
    QGESSCosting,
    QGESSCostingData,
)

MW_CO2 = 44.0095e-3   # kg/mol
MW_H2O = 18.0153e-3   # kg/mol


# ---------------------------------------------------------------------------
# IDAES property-package helpers (ported from Costing 5.ipynb)
# ---------------------------------------------------------------------------

def _co2_state_from_hp(h, p):
    """Build a throwaway swco2 state fixed at (enth_mol, pressure); solve; return the state block."""
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.properties = SWCO2ParameterBlock()
    m.fs.compr_case_1 = PressureChanger(
        dynamic=False,
        property_package=m.fs.properties,
        compressor=True,
        thermodynamic_assumption=ThermodynamicAssumption.isentropic,
    )
    m.fs.state = m.fs.properties.build_state_block([0], has_phase_equilibrium=False)
    m.fs.state[0].flow_mol.fix(1.0)
    # swco2's default enth_mol bounds are set for a narrow "typical" range
    # and are routinely narrower than the values this file fixes here --
    # both the brentq search bracket in _co2_state_from_sp() below (which
    # deliberately explores far outside any physical range while bisecting)
    # and, for some cycle conditions, the real converged enthalpies
    # themselves. Since this Var is immediately fixed (not solved for),
    # its bounds don't affect the solve -- clearing them just avoids
    # Pyomo's W1002 "outside the bounds" warning at every such call.
    m.fs.state[0].enth_mol.setlb(None)
    m.fs.state[0].enth_mol.setub(None)
    m.fs.state[0].enth_mol.fix(h)
    m.fs.state[0].pressure.fix(p)

    # trigger construction of every derived property we'll want to read
    # (dens_mass added for pump_co2()'s incompressible-liquid pump-work
    # calculation below -- same "_mass" bulk-property naming convention as
    # enth_mass/entr_mass; if your installed IDAES version names overall
    # mixture density differently, e.g. dens_mass_phase["Liq"], update
    # pump_co2() to match).
    _ = m.fs.state[0].temperature
    _ = m.fs.state[0].entr_mol
    _ = m.fs.state[0].enth_mass
    _ = m.fs.state[0].entr_mass
    _ = m.fs.state[0].dens_mass

    solver = SolverFactory('ipopt')
    solver.solve(m, tee=False)
    return m.fs.state[0]


def _co2_state_from_sp(s_target, p, h_guess_low=None, h_guess_high=None):
    """
    Find the enth_mol that produces entr_mol == s_target at fixed pressure p,
    via root-finding (entr_mol is a plain Expression on swco2, not a fixable
    Var, so we can't just flip which variable is 'known' the way we could
    with a real Var -- we have to search for it instead).
    """
    def entropy_residual(h_trial):
        state = _co2_state_from_hp(h_trial, p)
        return value(state.entr_mol) - s_target

    # Bracket the root. CO2 molar enthalpy near these cycle conditions
    # typically falls somewhere in this range -- widen if you get a
    # "f(a) and f(b) must have different signs" error, which means the
    # true root sits outside this bracket for your specific conditions.
    if h_guess_low is None:
        h_guess_low = -500000.0     # J/mol, low bound
    if h_guess_high is None:
        h_guess_high = 40000000.0    # J/mol, high bound

    h_solution = brentq(entropy_residual, h_guess_low, h_guess_high, xtol=1e-3)

    return _co2_state_from_hp(h_solution, p)


def get_idaes_state_co2(p, t):
    """Resolve mass-basis enthalpy and entropy of CO2 from a known T, P (Span-Wagner)."""
    h_molar = value(htpx(T=t * units.K, P=p * units.Pa))
    state = _co2_state_from_hp(h_molar, p)
    return value(state.enth_mass), value(state.entr_mass)


def _water_state_from_hp(h, p):
    """Build a throwaway iapws95 state fixed at (enth_mol, pressure); solve; return the state block."""
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.properties = iapws95.Iapws95ParameterBlock()
    m.fs.compr_case_1 = PressureChanger(
        dynamic=False,
        property_package=m.fs.properties,
        compressor=True,
        thermodynamic_assumption=ThermodynamicAssumption.isentropic,
    )

    m.fs.state = m.fs.properties.build_state_block([0], has_phase_equilibrium=False)
    m.fs.state[0].flow_mol.fix(1.0)
    # See the matching comment in _co2_state_from_hp() above -- clearing
    # these bounds before fixing avoids Pyomo's W1002 "outside the bounds"
    # warning; it has no effect on the solve since this Var is fixed, not
    # solved for.
    m.fs.state[0].enth_mol.setlb(None)
    m.fs.state[0].enth_mol.setub(None)
    m.fs.state[0].enth_mol.fix(h)
    m.fs.state[0].pressure.fix(p)

    # Trigger construction of every derived property we'll want to read
    # (dens_mass added so callers can size equipment -- e.g. a cooling
    # tower -- on the water side's volumetric flow rate, the same way
    # _co2_state_from_hp() above exposes it for pump_co2()'s incompressible-
    # liquid pump-work calc).
    _ = m.fs.state[0].temperature
    _ = m.fs.state[0].entr_mol
    _ = m.fs.state[0].enth_mass
    _ = m.fs.state[0].entr_mass
    _ = m.fs.state[0].dens_mass

    solver = SolverFactory('ipopt')
    solver.solve(m, tee=False)
    return m.fs.state[0]


def get_idaes_state_water(p, t):
    """Resolve mass-basis enthalpy and entropy of Water from a known T, P (IAPWS-95)."""
    h_molar = value(htpxwater(T=t * units.K, P=p * units.Pa))
    state = _water_state_from_hp(h_molar, p)
    return value(state.enth_mass), value(state.entr_mass)


def tmexergy_co2(h, s, m_dot):
    """Mass-basis specific exergy for CO2, reference state 25 C / 1 atm."""
    t0 = 273.15 + 25
    p0 = 101325
    h0, s0 = get_idaes_state_co2(p0, t0)
    ex = m_dot * ((h - h0) - (t0 * (s - s0)))
    return ex / 1000


def tmexergy_water(h, s, m_dot):
    """Mass-basis specific exergy for Water, reference state 25 C / 1 atm."""
    t0 = 273.15 + 25
    p0 = 101325
    h0, s0 = get_idaes_state_water(p0, t0)
    ex = m_dot * ((h - h0) - (t0 * (s - s0)))
    return ex / 1000


# --- Flue gas: Peng-Robinson (Generic/Cubic) mixture property package -----

flue_gas_configuration = {
    "components": {
        "N2": {
            "type": Component,
            "enth_mol_ig_comp": NIST,
            "entr_mol_ig_comp": NIST,
            "parameter_data": {
                "mw": (28.0134e-3, units.kg / units.mol),
                "pressure_crit": (33.94e5, units.Pa),
                "temperature_crit": (126.2, units.K),
                "omega": 0.037,
                "cp_mol_ig_comp_coeff": {
                    "A": 28.98641, "B": 1.853978, "C": -9.647459,
                    "D": 16.63537, "E": 0.000117, "F": -8.671914,
                    "G": 226.4168, "H": 0.0,
                },
            },
        },
        "O2": {
            "type": Component,
            "enth_mol_ig_comp": NIST,
            "entr_mol_ig_comp": NIST,
            "parameter_data": {
                "mw": (31.9988e-3, units.kg / units.mol),
                "pressure_crit": (50.43e5, units.Pa),
                "temperature_crit": (154.6, units.K),
                "omega": 0.021,
                "cp_mol_ig_comp_coeff": {
                    "A": 31.32234, "B": -20.23531, "C": 57.86644,
                    "D": -36.50624, "E": -0.007374, "F": -8.903471,
                    "G": 246.7945, "H": 0.0,
                },
            },
        },
        "H2O": {
            "type": Component,
            "enth_mol_ig_comp": NIST,
            "entr_mol_ig_comp": NIST,
            "valid_phase_types": PT.vaporPhase,
            "parameter_data": {
                "mw": (18.0153e-3, units.kg / units.mol),
                "pressure_crit": (220.64e5, units.Pa),
                "temperature_crit": (647.1, units.K),
                "omega": 0.344,
                "cp_mol_ig_comp_coeff": {
                    "A": 30.09200, "B": 6.832514, "C": 6.793435,
                    "D": -2.534480, "E": 0.082139, "F": -250.8810,
                    "G": 223.3967, "H": -241.8264,
                },
            },
        },
        "CO2": {
            "type": Component,
            "enth_mol_ig_comp": NIST,
            "entr_mol_ig_comp": NIST,
            "parameter_data": {
                "mw": (44.0095e-3, units.kg / units.mol),
                "pressure_crit": (73.82e5, units.Pa),
                "temperature_crit": (304.23, units.K),
                "omega": 0.224,
                "cp_mol_ig_comp_coeff": {
                    "A": 24.99735, "B": 55.18696, "C": -33.69137,
                    "D": 7.948387, "E": -0.136638, "F": -403.6075,
                    "G": 228.2431, "H": -393.5224,
                },
            },
        },
    },
    "phases": {
        "Vap": {
            "type": VaporPhase,
            "equation_of_state": Cubic,
            "equation_of_state_options": {"type": CubicType.PR},
        },
    },
    "base_units": {
        "time": units.s, "length": units.m, "mass": units.kg,
        "amount": units.mol, "temperature": units.K,
    },
    "state_definition": FTPx,
    "state_bounds": {
        "flow_mol": (0, 100, 1000, units.mol / units.s),
        "temperature": (250, 900, 2000, units.K),
        "pressure": (5e4, 101325, 1e7, units.Pa),
    },
    "pressure_ref": (101325, units.Pa),
    "temperature_ref": (298.15, units.K),
    "parameter_data": {
        "PR_kappa": {
            ("N2", "O2"): 0.0,   ("O2", "N2"): 0.0,
            ("N2", "H2O"): 0.0,  ("H2O", "N2"): 0.0,
            ("N2", "CO2"): -0.017, ("CO2", "N2"): -0.017,
            ("O2", "H2O"): 0.0,  ("H2O", "O2"): 0.0,
            ("O2", "CO2"): 0.0,  ("CO2", "O2"): 0.0,
            ("H2O", "CO2"): 0.0, ("CO2", "H2O"): 0.0,
            ("N2", "N2"): 0.0,
            ("O2", "O2"): 0.0,
            ("H2O", "H2O"): 0.0,
            ("CO2", "CO2"): 0.0,
        }
    },
}

_MW_LOOKUP = {"Nitrogen": 28.0134e-3, "Oxygen": 31.9988e-3,
              "Water": 18.0153e-3, "CO2": 44.0095e-3}
_NAME_MAP = {"Nitrogen": "N2", "Oxygen": "O2", "Water": "H2O", "CO2": "CO2"}


def _parse_flue_fluid_string(flue_fluid_str):
    """
    Parse a CoolProp HEOS mixture string, e.g.:
      'HEOS::Nitrogen[0.713...]&Oxygen[0.110...]&Water[0.137...]&CO2[0.039...]'
    Returns {PR_component_name: mole_fraction}, e.g. {'N2': 0.713, ...}.
    The hot_fluid constructor argument is still given in this format (it is
    only ever used here, to get the mole fractions -- it is never passed to
    CoolProp).
    """
    matches = re.findall(r'([A-Za-z0-9]+)\[([\d.eE+-]+)\]', flue_fluid_str)
    return {_NAME_MAP.get(name, name): float(frac) for name, frac in matches}


def _flue_mw(mole_fracs):
    """Mixture molar mass (kg/mol), keyed by PR component names (N2, O2, H2O, CO2)."""
    inv_map = {v: k for k, v in _NAME_MAP.items()}
    return sum(mole_fracs[c] * _MW_LOOKUP[inv_map[c]] for c in mole_fracs)


def _flue_state_from_tpx(T, P, mole_fracs):
    """
    Build a Peng-Robinson (Generic/Cubic) state block for the flue gas
    mixture, fixed at (T, P, mole fractions). UNLIKE the swco2/iapws95
    helpers, this DOES need solver.solve() -- PR's compressibility factor
    (Z) is a real Var determined by a cubic-equation constraint (multiple
    mathematical roots exist; the solver picks the physical one), not a
    closed-form expression like enth_mol was for the Helmholtz packages.
    """
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.properties = GenericParameterBlock(**flue_gas_configuration)
    m.fs.state = m.fs.properties.build_state_block([0], defined_state=True)

    m.fs.state[0].flow_mol.fix(1.0)
    m.fs.state[0].temperature.fix(T)
    m.fs.state[0].pressure.fix(P)
    for comp, frac in mole_fracs.items():
        m.fs.state[0].mole_frac_comp[comp].fix(frac)

    _ = m.fs.state[0].enth_mol
    _ = m.fs.state[0].entr_mol

    SolverFactory('ipopt').solve(m, tee=False)
    return m.fs.state[0]


def tmexergy_flue(h, s, mole_fracs, m_dot):
    """Mass-basis specific exergy for the flue gas mixture, ref 25 C / 1 atm."""
    t0 = 273.15 + 25
    p0 = 101325
    state0 = _flue_state_from_tpx(t0, p0, mole_fracs)
    mw = _flue_mw(mole_fracs)
    h0 = value(state0.enth_mol) / mw
    s0 = value(state0.entr_mol) / mw
    ex = m_dot * ((h - h0) - (t0 * (s - s0)))
    return ex / 1000


# ---------------------------------------------------------------------------
# Cycle unit-operation functions (IDAES property packages, no CoolProp)
# ---------------------------------------------------------------------------

def turbine_co2(p_in, h_in, p_out, efficiency, turbine_fluid, m_dot):
    """
    Turbine calculation using IDAES swco2 (Span-Wagner) instead of CoolProp.
    Inputs/outputs match the original CoolProp-based turbine()'s units and
    mass basis exactly (mass-basis h_in/h_out, kW power).
    """
    h_in_molar = h_in * MW_CO2

    state_in = _co2_state_from_hp(h_in_molar, p_in)
    s_in_molar = value(state_in.entr_mol)
    s_in = value(state_in.entr_mass)

    # Isentropic outlet: same entropy, outlet pressure -> solve for h_isentropic
    state_isen = _co2_state_from_sp(s_in_molar, p_out)
    h_isentropic_molar = value(state_isen.enth_mol)
    h_isentropic = h_isentropic_molar / MW_CO2

    isentropic_specific_turbine_power = h_in - h_isentropic
    actual_specific_turbine_power = isentropic_specific_turbine_power * efficiency
    h_out = h_in - actual_specific_turbine_power

    h_out_molar = h_out * MW_CO2
    state_out = _co2_state_from_hp(h_out_molar, p_out)
    t_out = value(state_out.temperature)
    s_out = value(state_out.entr_mass)

    actual_turbine_power = actual_specific_turbine_power * m_dot / 1000

    ex_in = tmexergy_co2(h_in, s_in, m_dot)
    ex_out = tmexergy_co2(h_out, s_out, m_dot)
    ex_des = ex_in - ex_out - actual_turbine_power
    ex_eff = actual_turbine_power / (ex_in - ex_out)

    exergy = {
        'in': ex_in,
        'out': ex_out,
        'work': actual_turbine_power,
        'destroyed': ex_des,
        'efficiency': ex_eff,
    }
    return (t_out, h_out, actual_turbine_power, exergy)


def pump_co2(p_in, h_in, p_out, eff, pump_fluid, m_dot, factors=None):
    """
    Pump calculation using IDAES swco2 (Span-Wagner) instead of CoolProp,
    per the incompressible-liquid pump formula (ideal specific work =
    dp / density, actual = that divided by efficiency) rather than an
    isentropic-PressureChanger model. Mass-basis h_in/h_out and kW power,
    matching the rest of this file's convention.

    d_in (the density used for the ideal work calc) is read directly off
    the already-built inlet state block -- the original CoolProp version
    computed it at (S, P) rather than (H, P), but since S there was itself
    the actual inlet entropy at that same P, both notations pin down the
    identical single-phase state, so no separate entropy-based solve is
    needed here.

    Cost is sized on actual pump shaft power (kW) via uo.py's generic
    power-law pump correlation (equipment_cost('pump', ...)) -- NOT the
    IDAES sCO2 costing module used for the turbine below, since that
    module has no dedicated "Pump" category (see _cost_co2_turbomachine()'s
    docstring).
    """
    h_in_molar = h_in * MW_CO2
    state_in = _co2_state_from_hp(h_in_molar, p_in)
    s_in = value(state_in.entr_mass)
    t_in = value(state_in.temperature)
    d_in = value(state_in.dens_mass)

    del_h_ideal = (p_out - p_in) / d_in
    h_out = h_in + (del_h_ideal / eff)
    actual_specific_pump_power = h_out - h_in

    h_out_molar = h_out * MW_CO2
    state_out = _co2_state_from_hp(h_out_molar, p_out)
    t_out = value(state_out.temperature)
    s_out = value(state_out.entr_mass)

    actual_pump_power = actual_specific_pump_power * m_dot / 1000

    ex_in = tmexergy_co2(h_in, s_in, m_dot)
    ex_out = tmexergy_co2(h_out, s_out, m_dot)
    ex_des = actual_pump_power - (ex_out - ex_in)
    ex_eff = (ex_out - ex_in) / actual_pump_power

    exergy = {
        'in': ex_in,
        'out': ex_out,
        'work': actual_pump_power,
        'destroyed': ex_des,
        'efficiency': ex_eff,
    }
    cost = {
        'capacity_kw': actual_pump_power,
        'cost': equipment_cost('pump', actual_pump_power, factors)
    }
    return (t_out, h_out, actual_pump_power, exergy, cost)


def hx2_co2_water(p_hot_in, h_hot_in, h_hot_out, flowrate_hot, hot_drop,
                   p_cold_in, h_cold_in, cold_drop, approach):
    """
    Condenser calculation using IDAES swco2 (Span-Wagner) for the hot (CO2)
    side and iapws95 (IAPWS-95) for the cold (water) side, replacing
    CoolProp. Matches the original hx2()'s logic and units exactly -- hot
    outlet enthalpy is already known/given, cold outlet is set by an
    approach temperature, and the required cold flowrate is solved
    backward from the resulting duty.
    """
    p_hot_out = p_hot_in - hot_drop
    p_cold_out = p_cold_in - cold_drop

    state_hot_in = _co2_state_from_hp(h_hot_in * MW_CO2, p_hot_in)
    state_hot_out = _co2_state_from_hp(h_hot_out * MW_CO2, p_hot_out)

    q_kw = flowrate_hot * (h_hot_in - h_hot_out) / 1000

    state_cold_in = _water_state_from_hp(h_cold_in * MW_H2O, p_cold_in)
    t_cold_in = value(state_cold_in.temperature)

    t_cold_out = t_cold_in + approach
    h_cold_out_molar = value(htpxwater(T=t_cold_out * units.K, P=p_cold_out * units.Pa))
    h_cold_out = h_cold_out_molar / MW_H2O

    flowrate_cold = q_kw * 1000 / (h_cold_out - h_cold_in)

    state_cold_out = _water_state_from_hp(h_cold_out * MW_H2O, p_cold_out)

    s_hot_in = value(state_hot_in.entr_mass)
    s_hot_out = value(state_hot_out.entr_mass)
    s_cold_in = value(state_cold_in.entr_mass)
    s_cold_out = value(state_cold_out.entr_mass)

    ex_hot_in = tmexergy_co2(h_hot_in, s_hot_in, flowrate_hot)
    ex_hot_out = tmexergy_co2(h_hot_out, s_hot_out, flowrate_hot)
    ex_cold_in = tmexergy_water(h_cold_in, s_cold_in, flowrate_cold)
    ex_cold_out = tmexergy_water(h_cold_out, s_cold_out, flowrate_cold)

    ex_hot_change = ex_hot_in - ex_hot_out
    ex_cold_change = ex_cold_out - ex_cold_in
    ex_des = ex_hot_change - ex_cold_change
    ex_eff = ex_cold_change / ex_hot_change

    exergy = {
        'hot_in': ex_hot_in,
        'hot_out': ex_hot_out,
        'cold_in': ex_cold_in,
        'cold_out': ex_cold_out,
        'hot_change': ex_hot_change,
        'cold_change': ex_cold_change,
        'destroyed': ex_des,
        'efficiency': ex_eff,
    }
    return (t_cold_out, p_cold_out, h_cold_out, q_kw, flowrate_cold, exergy)


def hx3_flue_co2(hot_fluid, t_hot_in, p_hot_in, flowrate_hot, hot_drop,
                  cold_fluid, p_cold_in, h_cold_in, h_cold_out, flowrate_cold,
                  cold_drop, approach2):
    """
    Flue-gas/CO2 heat exchanger: Peng-Robinson (hot, flue gas mixture) +
    swco2 Span-Wagner (cold, CO2), replacing CoolProp -- ported directly
    from Costing 6's cell 3 (same function, same calling convention), used
    here in place of the earlier heater_co2() helper.

    Unlike heater_co2(), the cold (CO2) outlet enthalpy is a GIVEN input
    here, not solved for -- it's only used for the cold-side exergy
    calculation. calc() below passes this iteration's assumed turbine-inlet
    enthalpy (self.h1) as h_cold_out, then -- exactly like heater_co2() did
    internally -- computes what the CO2 outlet enthalpy would actually be
    from the resulting duty (q_kw) via a simple energy balance, to check
    against that assumption in its convergence loop. hot_fluid is the
    CoolProp HEOS mixture string, parsed for mole fractions; cold_fluid is
    kept only for call-site compatibility (unused, always CO2 here).
    """
    p_hot_out = p_hot_in - hot_drop
    p_cold_out = p_cold_in - cold_drop

    mole_fracs = _parse_flue_fluid_string(hot_fluid)
    mw_flue = _flue_mw(mole_fracs)

    # --- Hot side (flue gas, PR) ---
    state_hot_in = _flue_state_from_tpx(t_hot_in, p_hot_in, mole_fracs)
    h_hot_in = value(state_hot_in.enth_mol) / mw_flue   # mass basis
    s_hot_in = value(state_hot_in.entr_mol) / mw_flue

    # --- Cold side (CO2, Span-Wagner), from GIVEN mass enthalpies ---
    state_cold_in = _co2_state_from_hp(h_cold_in * MW_CO2, p_cold_in)
    t_cold_in = value(state_cold_in.temperature)
    s_cold_in = value(state_cold_in.entr_mass)

    state_cold_out = _co2_state_from_hp(h_cold_out * MW_CO2, p_cold_out)
    s_cold_out = value(state_cold_out.entr_mass)

    # --- Hot outlet: approach-temperature logic (same as heater_co2()) ---
    t_hot_out = t_cold_in + approach2
    state_hot_out = _flue_state_from_tpx(t_hot_out, p_hot_out, mole_fracs)
    h_hot_out = value(state_hot_out.enth_mol) / mw_flue
    s_hot_out = value(state_hot_out.entr_mol) / mw_flue

    q_kw = flowrate_hot * (h_hot_in - h_hot_out) / 1000

    # --- Exergy balance ---
    ex_hot_in = tmexergy_flue(h_hot_in, s_hot_in, mole_fracs, flowrate_hot)
    ex_hot_out = tmexergy_flue(h_hot_out, s_hot_out, mole_fracs, flowrate_hot)
    ex_cold_in = tmexergy_co2(h_cold_in, s_cold_in, flowrate_cold)
    ex_cold_out = tmexergy_co2(h_cold_out, s_cold_out, flowrate_cold)

    ex_hot_change = ex_hot_in - ex_hot_out
    ex_cold_change = ex_cold_out - ex_cold_in
    ex_des = ex_hot_change - ex_cold_change
    ex_eff = (ex_cold_change / ex_hot_change) if ex_hot_change != 0 else 0

    exergy = {
        'hot_in': ex_hot_in, 'hot_out': ex_hot_out,
        'cold_in': ex_cold_in, 'cold_out': ex_cold_out,
        'hot_change': ex_hot_change, 'cold_change': ex_cold_change,
        'destroyed': ex_des, 'efficiency': ex_eff,
    }
    return (t_hot_out, p_hot_out, h_hot_out, p_cold_out, q_kw, exergy)


# ---------------------------------------------------------------------------
# Costing helpers
# ---------------------------------------------------------------------------

def _safe_cost(cost_dict):
    if not cost_dict:
        return 0.0
    value_ = cost_dict.get('cost')
    return value_ if value_ is not None else 0.0


def _safe_num(x):
    return x if x is not None else 0.0


def _escalate(cost_value, factor):
    """
    Applies a cost-index escalation factor to a single cost value, passing
    None through unchanged -- so an un-costable component (invalid
    capacity/LMTD) or a failed turbomachinery costing call doesn't turn
    into a spurious number. Same helper as in super.py.
    """
    return None if cost_value is None else cost_value * factor


def _hx_cost_given_u(q_kw, temps, u, f, factors=None):
    if q_kw is None or q_kw <= 0:
        return None, None
    if u is None or u <= 0:
        return None, None
    q_watts = q_kw * 1000.0
    dtlm = _lmtd(temps['t_hot_in'], temps['t_hot_out'], temps['t_cold_in'], temps['t_cold_out'])
    if dtlm is None:
        return None, None
    area = q_watts / (u * dtlm * f)
    cost_value = equipment_cost('heat_exchanger', area, factors)
    return area, cost_value


def _cost_co2_turbomachine(p_in, h_in, p_out, m_dot, efficiency, compressor,
                            equipment, hot_temp_c=None, n_equip=1):
    """
    Cost a CO2 turbine or pump using IDAES's dedicated Supercritical CO2
    Costing Module -- part of the NETL Power Plant Costing Library
    (idaes.models_extra.power_generation.costing.power_plant_capcost,
    QGESSCostingData.get_sCO2_unit_cost) -- instead of uo.py's generic
    power-law cost correlations for turbine().

    Used for the TURBINE only in this file -- the pump is costed inside
    pump_co2() via uo.py's generic equipment_cost('pump', ...) correlation
    instead, since get_sCO2_unit_cost (below) has no dedicated "Pump"
    category (it is a Brayton-cycle/turbomachinery-oriented module: its
    closest analog, a generic gas compressor, is a poor cost proxy for a
    liquid pump). The `compressor` argument is kept (rather than hardcoded
    to False) only because get_sCO2_unit_cost's underlying PressureChanger
    model needs to know the flow direction of work; this function is not
    otherwise limited to turbines.

    It provides named equipment categories -- confirmed against the actual
    installed source of get_sCO2_unit_cost: "Axial turbine", "Radial
    turbine", "Coal-fired heater", "Natural gas-fired heater", "Recuperator"
    (all SINGULAR -- the plural forms from the documentation prose raise a
    KeyError, since the real dictionary keys don't match the docs) -- scaled
    on shaft power (MW_sh), with a temperature correction factor applied
    above 550 C for the equipment types above:

        equipment_cost = reference_cost * scaled_param**alpha * temp_factor

    Mechanically: build a real, standalone swco2 PressureChanger unit model,
    fully fix its physical operating point to reproduce the already-solved
    turbine_co2() result for this state, attach the sCO2 costing block, and
    solve the combined (physics + costing) model.

    Returns total_plant_cost in raw USD (the underlying Var is in $MM;
    get_sCO2_unit_cost's own default CE_index_year is "2018"). Per the
    module's own documentation, "There are currently no estimates for the
    total plant cost components, so bare erected cost will be the same as
    total plant cost for now" -- i.e. this is equipment + bare-erection cost
    only, not a full installed/EPC cost.

    PLEASE VERIFY before trusting the $ this returns against your installed
    IDAES version. If you get a KeyError on the equipment string, run this
    to get the real keys directly:

        from idaes.models_extra.power_generation.costing.power_plant_capcost import load_sCO2_costing_dictionary
        print(list(load_sCO2_costing_dictionary().keys()))

      * `equipment` must exactly match one of the module's category strings
        (case-sensitive). trans.py's default is "Radial turbine" -- override
        turbine_cost_equipment on tco2's constructor if your machine differs
        ("Axial turbine" scales the same way).
      * `scaled_param` is built below as an explicit Var in MW, tied to the
        unit's work_mechanical via a unit-converting constraint --
        get_sCO2_unit_cost calls scaled_param.get_units() on whatever is
        passed in and uses that verbatim, so it must be a real Var/Param
        (not a bare expression) and must genuinely be in MW (not IDAES's
        native Watts).
      * `hot_temp_c` (the equipment's hottest operating temperature, in
        Celsius) only changes the result once it exceeds 550 C -- trans.py
        passes the turbine inlet temperature, its hottest point.
    """
    flow_mol = m_dot * 1000.0 / 44.0095  # mol/s

    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.properties = SWCO2ParameterBlock()
    m.fs.unit = PressureChanger(
        dynamic=False,
        property_package=m.fs.properties,
        compressor=compressor,
        thermodynamic_assumption=ThermodynamicAssumption.isentropic,
    )

    m.fs.unit.inlet.flow_mol[0].fix(flow_mol)
    m.fs.unit.inlet.enth_mol[0].fix(h_in * MW_CO2)
    m.fs.unit.inlet.pressure[0].fix(p_in)
    m.fs.unit.outlet.pressure[0].fix(p_out)
    m.fs.unit.efficiency_isentropic[0].fix(efficiency)

    m.fs.costing = QGESSCosting()

    m.fs.unit.shaft_power = Var(initialize=1.0, units=units.MW)
    if compressor:
        m.fs.unit.shaft_power_eq = Constraint(
            expr=m.fs.unit.shaft_power == units.convert(m.fs.unit.work_mechanical[0], to_units=units.MW)
        )
    else:
        # IDAES PressureChanger sign convention: work_mechanical is negative
        # when work is extracted from the fluid (turbines/expanders).
        m.fs.unit.shaft_power_eq = Constraint(
            expr=m.fs.unit.shaft_power == -units.convert(m.fs.unit.work_mechanical[0], to_units=units.MW)
        )

    costing_args = {
        "equipment": equipment,
        "scaled_param": m.fs.unit.shaft_power,
        "n_equip": n_equip,
    }
    if hot_temp_c is not None:
        m.fs.unit.hot_temp = Var(initialize=hot_temp_c, units=units.C)
        m.fs.unit.hot_temp.fix(hot_temp_c)
        costing_args["temp_C"] = m.fs.unit.hot_temp

    m.fs.unit.costing = UnitModelCostingBlock(
        flowsheet_costing_block=m.fs.costing,
        costing_method=QGESSCostingData.get_sCO2_unit_cost,
        costing_method_arguments=costing_args,
    )

    m.fs.unit.initialize(outlvl=idaeslog.WARNING)
    QGESSCostingData.costing_initialization(m.fs.costing)
    SolverFactory('ipopt').solve(m, tee=False)
    # total_plant_cost is in $MM (2018-dollar basis by default); convert to raw USD
    return value(m.fs.unit.costing.total_plant_cost) * 1e6


# In[2]:


class tco2:
    fluid = 'CO2'
    def __init__(self, t_hot_in, p_hot_in, m_hot, hot_fluid, turbine_isen_eff, pump_eff, heater_effectiveness, pressure_loss,
                 turbine_cost_equipment="Radial turbine",
                 n_turbine_equip=1):
        self.t_hot_in = t_hot_in
        self.p_hot_in = p_hot_in
        self.m_hot = m_hot
        self.hot_fluid = hot_fluid
        self.t1 = None
        self.p1 = None
        self.h1 = None
        self.s1 = None
        self.turbine_isen_eff = turbine_isen_eff
        self.pump_eff = pump_eff
        self.heater_effectiveness = heater_effectiveness
        self.cond_drop_hot = pressure_loss[0]
        self.cond_drop_cold = pressure_loss[1]
        self.boiler_drop_hot = pressure_loss[2]
        self.boiler_drop_cold = pressure_loss[3]
        self.m = None
        self.Heater = None
        self.net_p = None
        self.state1 = None
        self.hx_heater = None
        self.turbine_power = None
        self.pump_exergy = None
        self.heater_exergy = None
        self.t_hot_out = None
        self.p_hot_out = None
        self.h_hot_out = None
        # IDAES sCO2 Costing Module (NETL/QGESS Power Plant Costing Library)
        # equipment category for _cost_co2_turbomachine() above (turbine
        # only -- the pump is costed via uo.py's generic equipment_cost()
        # inside pump_co2() instead). See that function's docstring for the
        # equipment-category options and what to verify against your
        # installed IDAES version.
        self.turbine_cost_equipment = turbine_cost_equipment
        self.n_turbine_equip = n_turbine_equip
        self.turbine_cost = None
        self.pump_cost = None
        self.condenser_cost = None
        self.heater_cost = None
        # State points captured at the end of my_expression() (below), so
        # that after the final (optimal mass-flowrate) call these hold the
        # converged values needed for turbomachinery/HX costing.
        self.p2 = None
        self.h2 = None
        self.t2 = None
        self.p3 = None
        self.h3 = None
        self.t3 = None
        self.p4 = None
        self.h4 = None
        self.t4 = None
        self.t01 = None
        self.pump_power = None
        self.hx_condenser = None
        self.t_cond_cold_in = None
        self.p_cond_cold_in = None
        self.h_cond_cold_in = None
        self.t_cond_cold_out = None
        self.flowrate_water = None
        self.cooling_tower_cost = None
        # CO2 working-fluid (inventory) cost: required CO2 charge is taken
        # as 0.3 kg per kW of evaporator (flue-gas heater) heat input,
        # priced at 18 INR/kg.
        self.co2_price_inr_per_kg = 18.0
        self.co2_charge_kg_per_kw = 0.3
        self.co2_mass_kg = None
        self.co2_fluid_cost = None



    def calc(self):
        pinch = 10
        fluid = 'CO2'
        f_heater = 0.75
        f_condenser = 0.8
        self.t1 = self.t_hot_in - pinch
        self.p1 = 20_000_000  #Pa (200 bar) Turbine Inlet Pressure

        mole_fracs = _parse_flue_fluid_string(self.hot_fluid)

        factors_pump = {
            'material_factor': 2.4,
            'temperature_factor': 1.6,
            'pressure_factor': 1.9,
            'piping_factor': 0.3
        }
        hx_cost_factors_heater = {
            'material_factor': 2.4,
            'temperature_factor': 1.6,
            'pressure_factor': 1.9,
            'piping_factor': 0.46
        }
        hx_cost_factors_condenser = {
            'material_factor': 2.4,
            'temperature_factor': 1,
            'pressure_factor': 1.9,
            'piping_factor': 0.46
        }
        cooling_tower_cost_factors = {
            'material_factor': 1,
            'temperature_factor': 1,
            'pressure_factor': 1,
            'piping_factor': 0.7
        }

        # Cost-index escalation factors (user-supplied), applied to bring
        # each cost block up to a March-2026 basis -- same two factors used
        # in super.py:
        #   - TURBOMACHINERY_ESCALATION: escalates the turbine cost from
        #     IDAES's sCO2 Costing Module's 2018 basis (see
        #     _cost_co2_turbomachine()'s docstring above) to March 2026.
        #   - REMAINING_COST_ESCALATION: escalates everything else (pump,
        #     condenser, heater, cooling tower -- all priced via uo.py's
        #     2024-basis COST_TABLE correlations) from 2024 to March 2026.
        TURBOMACHINERY_ESCALATION = 846.43 / 603.1
        REMAINING_COST_ESCALATION = 103.625 / 102.6166667

        def my_expression(mass_flowrate):

            

            # Turbine Inputs

            p2 = 7_000_000 # Pa  Turbine Outlet Pressure (70 bar)

            #Condenser

            t3 = 28 + 273.15
            t_cold_in = 20 + 273.15 # K   (20℃)
            p_cold_in = 250_000 # Pa  (2.5 bar)
            h_cold_in = value(htpxwater(T=t_cold_in * units.K, P=p_cold_in * units.Pa)) / MW_H2O

            # Turbine
            self.h1 = value(htpx(T=self.t1 * units.K, P=self.p1 * units.Pa)) / MW_CO2

            t2, h2, self.turbine_power, turbine_exergy = turbine_co2(self.p1, self.h1, p2, self.turbine_isen_eff, fluid, mass_flowrate)

            p3 = p2 - self.cond_drop_hot
            h3 = value(htpx(T=t3 * units.K, P=p3 * units.Pa)) / MW_CO2

            t_water_out, p_water_out, h_water_out, hx_condenser, flowrate_water, condenser_exergy = hx2_co2_water(p2, h2, h3, mass_flowrate, self.cond_drop_hot, p_cold_in, h_cold_in, self.cond_drop_cold, 5)

            # pump
            p4 = self.p1 + self.cond_drop_hot + self.boiler_drop_cold # Pa  Pump Outlet Pressure
            t4, h4, pump_power, self.pump_exergy, pump_cost_dict = pump_co2(p3, h3, p4, self.pump_eff, fluid, mass_flowrate, factors_pump)

            # Heater
            self.t_hot_out, self.p_hot_out, self.h_hot_out, p01, self.hx_heater, self.heater_exergy = hx3_flue_co2(self.hot_fluid, self.t_hot_in, self.p_hot_in, self.m_hot, self.boiler_drop_hot, fluid, p4, h4, self.h1, mass_flowrate, self.boiler_drop_cold, pinch)
            # hx3_flue_co2() takes the CO2 outlet enthalpy as a GIVEN target
            # (self.h1, this iteration's guess) rather than solving for it --
            # so the actual achievable outlet state is computed here from the
            # resulting duty via energy balance, for the convergence check
            # just below (same math heater_co2() used to do internally).
            #h01 = h4 + (self.hx_heater * 1000.0) / mass_flowrate
            #t01 = value(_co2_state_from_hp(h01 * MW_CO2, p01).temperature)

            #To make self.t1 and t01 equal
            
            self.p2 = p2
            self.h2 = h2
            self.t2 = t2
            self.p3 = p3
            self.h3 = h3
            self.t3 = t3
            self.p4 = p4
            self.h4 = h4
            self.t4 = t4
            self.pump_power = pump_power
            self.pump_cost = pump_cost_dict
            self.hx_condenser = hx_condenser
            self.t_cond_cold_in = t_cold_in
            self.p_cond_cold_in = p_cold_in
            self.h_cond_cold_in = h_cold_in
            self.t_cond_cold_out = t_water_out
            self.flowrate_water = flowrate_water

            net_power = self.turbine_power - pump_power

            self.exergy = dict(heater=self.heater_exergy, turbine=turbine_exergy, pump=self.pump_exergy, condenser=condenser_exergy)  # for the Sankey diagram in app.py
            return net_power



        #optimize end

        # Maximum flow rate calculation Start
        mw_flue = _flue_mw(mole_fracs)
        state_flue_in = _flue_state_from_tpx(self.t_hot_in, self.p_hot_in, mole_fracs)
        h_hot_in = value(state_flue_in.enth_mol) / mw_flue

        # IDAES's Peng-Robinson flue-gas package here (ported from
        # Costing 5.ipynb) models the flue gas as vapor-only, so it has no
        # condensation/dew-point calculation to reproduce CoolProp's
        # "Q|gas" call this bound used previously. A fixed floor
        # temperature substitutes for it -- a practical minimum stack
        # temperature (~127 C) used to avoid over-cooling the flue gas in
        # this mass-flowrate search bound, not a hard cycle constraint.
        t_hot_cond = 273.15+65  # K (~65 C)
        state_flue_cond = _flue_state_from_tpx(t_hot_cond, self.p_hot_in - self.boiler_drop_hot, mole_fracs)
        h_hot_cond = value(state_flue_cond.enth_mol) / mw_flue
        q_hot_max = self.m_hot * (h_hot_in - h_hot_cond)
        net_power = my_expression(1)

        best_x = q_hot_max/(self.h1 - self.h4)

        #Final
        mass_flowrate = best_x
        self.t1 = self.t_hot_in - pinch
        self.p1 = 20_000_000  #Pa (200 bar) Turbine Inlet Pressure
        net_power = my_expression(mass_flowrate)
        self.t1 = value(_co2_state_from_hp(self.h1 * MW_CO2, self.p1).temperature)


        #Cell 6

        net_efficiency = 100 * net_power/self.hx_heater

        exergy_input = self.heater_exergy['hot_change'] 
        exergy_output = self.turbine_power - self.pump_exergy['work']
        exergy_efficiency = 100 * exergy_output/exergy_input
        self.m = mass_flowrate
        #print('t_flue_in = ', self.t_hot_in - 273.15, 't1 = ', self.t1-273.15, 'p1 = ', self.p1/100_000, 'h1 = ', self.h1, 'hx heater = ', self.hx_heater, 'm = ', mass_flowrate, 'net_power = ', net_power, 'self.turbine_power = ', self.turbine_power, 'Pump power = ', self.pump_exergy['work'])

        # ------------------------------------------------------------------
        # Costing
        # ------------------------------------------------------------------
        

        # Turbine: costed with IDAES's dedicated Supercritical CO2 Costing
        # Module (NETL/QGESS Power Plant Costing Library,
        # QGESSCostingData.get_sCO2_unit_cost) instead of uo.py's built-in
        # power-law cost correlation -- see _cost_co2_turbomachine()'s
        # docstring above for the equipment-category caveats. Wrapped in
        # try/except so an equipment-category mismatch reports a clear
        # warning instead of crashing the whole cycle solve.
        # Escalated from IDAES's sCO2 Costing Module's 2018 basis to March
        # 2026 via TURBOMACHINERY_ESCALATION (defined above), before the
        # existing USD->INR (*95) conversion applied later in total_cost.
        USD_TO_INR = 95
        try:
            turbine_cost_usd_2018 = _cost_co2_turbomachine(
                self.p1, self.h1, self.p2, self.m, self.turbine_isen_eff, compressor=False,
                equipment=self.turbine_cost_equipment,
                hot_temp_c=self.t1 - 273.15,   # turbine inlet: hottest point of an expander
                n_equip=self.n_turbine_equip)
        except Exception as err:
            print('Turbine costing failed -- check turbine_cost_equipment against your installed IDAES sCO2 costing module:', err)
            turbine_cost_usd_2018 = None
        turbine_cost_usd_2026 = _escalate(turbine_cost_usd_2018, TURBOMACHINERY_ESCALATION)
        # Same dict shape as pump/condenser/heater/cooling-tower costs: sizing
        # basis + operating point + 'cost' (INR, March-2026 basis).
        self.turbine_cost = {
            'capacity_kw': self.turbine_power,
            'equipment': self.turbine_cost_equipment,
            'n_equip': self.n_turbine_equip,
            'inlet_temp_c': self.t1 - 273.15,
            'inlet_pressure_bar': self.p1 / 1e5,
            'outlet_pressure_bar': self.p2 / 1e5,
            'isentropic_eff': self.turbine_isen_eff,
            'cost_usd_2018': turbine_cost_usd_2018,
            'escalation_factor': TURBOMACHINERY_ESCALATION,
            'cost_usd_2026': turbine_cost_usd_2026,
            'usd_to_inr': USD_TO_INR,
            'cost': _escalate(turbine_cost_usd_2026, USD_TO_INR),
        }

        # Pump: self.pump_cost was already set inside pump_co2() (via
        # uo.py's generic equipment_cost('pump', ...) correlation, sized on
        # actual pump shaft power) at the end of the converged
        # my_expression() call above. Escalated here from uo.py's 2024
        # basis to March 2026 via REMAINING_COST_ESCALATION.
        if self.pump_cost is not None:
            self.pump_cost['cost_2024'] = self.pump_cost.get('cost')
            self.pump_cost['escalation_factor'] = REMAINING_COST_ESCALATION
            self.pump_cost['cost'] = _escalate(self.pump_cost.get('cost'), REMAINING_COST_ESCALATION)
            self.pump_cost.update({
                'inlet_pressure_bar': self.p3 / 1e5,
                'outlet_pressure_bar': self.p4 / 1e5,
                'efficiency': self.pump_eff,
            })

        # Condenser and heater: costed the same duty/LMTD/assumed-U way as
        # the desorber/evaporator/absorber/condenser in libr.py -- NOT
        # uo.py's built-in phase-estimated-U costing for hx2()/hx_pinch_orc().
        condenser_temp = {
            't_hot_in': self.t2,
            't_hot_out': self.t3,
            't_cold_in': self.t_cond_cold_in,
            't_cold_out': self.t_cond_cold_out
        }
        condenser_u = 500  # W/m^2-K, assumed for a CO2-to-water condenser

        heater_temp = {
            't_hot_in': self.t_hot_in,
            't_hot_out': self.t_hot_out,
            't_cold_in': self.t4,
            't_cold_out': self.t1
        }
        heater_u = 60  # W/m^2-K, assumed for a flue-gas-heated CO2 heater

        # Each heat-exchanger cost below is escalated from uo.py's 2024
        # basis to March 2026 via REMAINING_COST_ESCALATION (defined above).
        condenser_area, condenser_cost_value = _hx_cost_given_u(self.hx_condenser, condenser_temp, condenser_u, f_condenser, hx_cost_factors_condenser)
        condenser_cost_value = _escalate(condenser_cost_value, REMAINING_COST_ESCALATION)
        heater_area, heater_cost_value = _hx_cost_given_u(self.hx_heater, heater_temp, heater_u, f_heater, hx_cost_factors_heater)
        heater_cost_value = _escalate(heater_cost_value, REMAINING_COST_ESCALATION)

        self.condenser_cost = {
            'duty_kw': self.hx_condenser,
            'area_m2': condenser_area,
            'u_assumed': condenser_u,
            'f_correction': f_condenser,
            'lmtd_k': _lmtd(condenser_temp['t_hot_in'], condenser_temp['t_hot_out'], condenser_temp['t_cold_in'], condenser_temp['t_cold_out']),
            'temps_c': {k: v - 273.15 for k, v in condenser_temp.items()},
            'escalation_factor': REMAINING_COST_ESCALATION,
            'cost': condenser_cost_value,
        }
        self.heater_cost = {
            'duty_kw': self.hx_heater,
            'area_m2': heater_area,
            'u_assumed': heater_u,
            'f_correction': f_heater,
            'lmtd_k': _lmtd(heater_temp['t_hot_in'], heater_temp['t_hot_out'], heater_temp['t_cold_in'], heater_temp['t_cold_out']),
            'temps_c': {k: v - 273.15 for k, v in heater_temp.items()},
            'escalation_factor': REMAINING_COST_ESCALATION,
            'cost': heater_cost_value,
        }

        # Cooling tower: sized on the condenser's cooling-water volumetric
        # flow rate (self.flowrate_water, kg/s -> m3/hr via the water
        # density at the condenser's cold-side inlet), priced with uo.py's
        # generic power-law correlation (equipment_cost('cooling_tower', ...))
        # -- the same correlation added for the sCO2 cycle's cooling tower.
        # No CoolProp here (this file uses IDAES's iapws95 package
        # exclusively), so the density comes from _water_state_from_hp(),
        # which now also exposes dens_mass (see that function above).
        state_cw_in = _water_state_from_hp(self.h_cond_cold_in * MW_H2O, self.p_cond_cold_in)
        cooling_water_density = value(state_cw_in.dens_mass)
        cooling_tower_flow_m3hr = (self.flowrate_water / cooling_water_density) * 3600.0
        cooling_tower_cost_value = equipment_cost('cooling_tower', cooling_tower_flow_m3hr, cooling_tower_cost_factors)
        cooling_tower_cost_value = _escalate(cooling_tower_cost_value, REMAINING_COST_ESCALATION)
        self.cooling_tower_cost = {
            'flow_kgs': self.flowrate_water,
            'flow_m3hr': cooling_tower_flow_m3hr,
            'water_density_kgm3': cooling_water_density,
            'water_in_c': self.t_cond_cold_in - 273.15,
            'water_out_c': self.t_cond_cold_out - 273.15,
            'escalation_factor': REMAINING_COST_ESCALATION,
            'cost': cooling_tower_cost_value,
        }

        # CO2 working-fluid cost (INR): required CO2 charge = 0.3 * evaporator
        # (flue-gas heater) heat input in kW, in kg, priced at 18 INR/kg.
        # self.hx_heater is that duty in kW (from hx3_flue_co2).
        self.co2_mass_kg = self.co2_charge_kg_per_kw * self.hx_heater
        self.co2_fluid_cost = self.co2_mass_kg * self.co2_price_inr_per_kg

        self.co2_cost = {
            'charge_kg_per_kw': self.co2_charge_kg_per_kw,
            'heater_duty_kw': self.hx_heater,
            'mass_kg': self.co2_mass_kg,
            'price_inr_per_kg': self.co2_price_inr_per_kg,
            'cost': self.co2_fluid_cost,
        }

        # Flat per-component totals (kept for backward compatibility) plus a
        # full per-component breakdown under 'details'.
        total_cost = {
            'turbine_cost': _safe_cost(self.turbine_cost),
            'pump_cost': _safe_cost(self.pump_cost),
            'condenser_cost': _safe_cost(self.condenser_cost),
            'heater_cost': _safe_cost(self.heater_cost),
            'cooling_tower_cost': _safe_cost(self.cooling_tower_cost),
            'co2_fluid_cost': _safe_cost(self.co2_cost),
        }
        total_cost['equipment_total'] = (total_cost['turbine_cost'] + total_cost['pump_cost']
                                         + total_cost['condenser_cost'] + total_cost['heater_cost']
                                         + total_cost['cooling_tower_cost'])
        total_cost['total'] = total_cost['equipment_total'] + total_cost['co2_fluid_cost']

        total_cost['details'] = {
            'turbine': self.turbine_cost,
            'pump': self.pump_cost,
            'condenser': self.condenser_cost,
            'heater': self.heater_cost,
            'cooling_tower': self.cooling_tower_cost,
            'co2_fluid': self.co2_cost,
        }
        total_cost['share_pct'] = {
            k: (100 * total_cost[k] / total_cost['total'] if total_cost['total'] else 0.0)
            for k in ('turbine_cost', 'pump_cost', 'condenser_cost', 'heater_cost',
                      'cooling_tower_cost', 'co2_fluid_cost')
        }
        total_cost['basis'] = {
            'currency': 'INR',
            'year': 'March 2026',
            'turbomachinery_escalation': TURBOMACHINERY_ESCALATION,
            'remaining_cost_escalation': REMAINING_COST_ESCALATION,
            'usd_to_inr': USD_TO_INR,
        }
        total_cost['performance'] = {
            'net_power_kw': net_power,
            'turbine_power_kw': self.turbine_power,
            'pump_power_kw': self.pump_power,
            'heater_duty_kw': self.hx_heater,
            'condenser_duty_kw': self.hx_condenser,
            'co2_flow_kgs': self.m,
            'cooling_water_flow_kgs': self.flowrate_water,
            'specific_cost_per_kw': (total_cost['total'] / net_power) if net_power else None,
        }

        #print('--- tCO2 cost breakdown (INR, March 2026) ---')
        #for name, d in total_cost['details'].items():
            #print(f'  {name:14s}: {_safe_cost(d):>15,.0f}  ({total_cost["share_pct"][name + "_cost"]:5.1f} %)  {d}')
        #print(f'  {"TOTAL":14s}: {total_cost["total"]:>15,.0f}')
        #print('  Specific cost (INR/kW net) = ', total_cost['performance']['specific_cost_per_kw'])
        #print('Heater Input = ', self.hx_heater)


        flue = (self.t_hot_out, self.p_hot_out, self.h_hot_out, self.m_hot, self.hot_fluid)
        return net_efficiency, exergy_efficiency, net_power, flue, total_cost



