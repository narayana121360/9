# New Property Package

import CoolProp.CoolProp as CP
import plotly.graph_objects as go
import sys
import re
from pyomo.environ import ConcreteModel, SolverFactory, value, units
from idaes.core import FlowsheetBlock
import idaes.logger as idaeslog
from idaes.models.properties.swco2 import SWCO2ParameterBlock, StateVars, htpx
from idaes.models.properties import iapws95
from idaes.models.properties.iapws95 import htpx as htpxwater
from idaes.models.unit_models.pressure_changer import (
    PressureChanger,
    ThermodynamicAssumption,
)
from idaes.core.util.model_statistics import degrees_of_freedom
from pyomo.environ import Var, Param, Expression, Constraint
from scipy.optimize import root_scalar
from scipy.optimize import brentq
from idaes.core import VaporPhase, Component
from idaes.core.base.phases import PhaseType as PT
from idaes.models.properties.modular_properties.base.generic_property import GenericParameterBlock
from idaes.models.properties.modular_properties.state_definitions import FTPx
from idaes.models.properties.modular_properties.eos.ceos import Cubic, CubicType
from idaes.models.properties.modular_properties.pure import NIST
from idaes.core import UnitModelCostingBlock
from idaes.models_extra.power_generation.costing.power_plant_capcost import (
    QGESSCosting,
    QGESSCostingData,
)
fluid = 'CO2'


def _safe_cost(cost_dict):
    """
    Pulls the numeric 'cost' out of a component cost dict, treating a
    missing dict or a None cost (e.g. invalid capacity/LMTD) as 0 so a
    single un-costable component doesn't blow up the plant total. Same
    helper as in libr.py/steam.py/orc.py.
    """
    if not cost_dict:
        return 0.0
    value = cost_dict.get('cost')
    return value if value is not None else 0.0


def _safe_num(x):
    """Treats a bare None cost (e.g. an IDAES turbomachinery costing call
    that failed) as 0 so it doesn't blow up a running total."""
    return x if x is not None else 0.0


def _escalate(cost_value, factor):
    """
    Applies a cost-index escalation factor to a single cost value, passing
    None through unchanged -- so an un-costable component (invalid
    capacity/LMTD) or a failed turbomachinery costing call doesn't turn
    into a spurious number.
    """
    return None if cost_value is None else cost_value * factor


def _hx_cost_given_u(q_kw, temps, u, f, factors=None):
    """
    Prices a heat exchanger whose overall heat-transfer coefficient (U) is
    supplied directly (rather than estimated from stream phases, as in
    uo.py's hx_cost_from_duty). Area = Q / (U * LMTD).
    `temps` is a dict with keys t_hot_in, t_hot_out, t_cold_in, t_cold_out.
    Returns (area_m2, cost_value); either may be None if the duty or the
    terminal temperatures don't support a valid LMTD.

    This is the exact same helper used in libr.py to cost its desorber,
    evaporator, absorber and condenser -- used here for the 5 heat
    exchangers in this sCO2 cycle (HTR, LTR, bottoming-cycle HX,
    cooler/condenser, flue-gas heater) instead of any IDAES heat-exchanger
    costing method.
    """
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

from uo import generate_sankey_diagram, pure_exergy, tmexergy, equipment_cost, _lmtd

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
    # themselves. Since this Var is immediately fixed (not solved for), its
    # bounds don't affect the solve -- clearing them just avoids Pyomo's
    # W1002 "outside the bounds" warning at every such call.
    m.fs.state[0].enth_mol.setlb(None)
    m.fs.state[0].enth_mol.setub(None)
    m.fs.state[0].enth_mol.fix(h)
    m.fs.state[0].pressure.fix(p)

    # trigger construction of every derived property we'll want to read
    _ = m.fs.state[0].temperature
    _ = m.fs.state[0].entr_mol
    _ = m.fs.state[0].enth_mass
    _ = m.fs.state[0].entr_mass

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

    # Bracket the root. CO2 molar enthalpy near sCO2 cycle conditions
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
    _ = m.fs.state[0].temperature
    _ = m.fs.state[0].entr_mol
    _ = m.fs.state[0].enth_mass
    _ = m.fs.state[0].entr_mass

    solver = SolverFactory('ipopt')
    solver.solve(m, tee=False)
    return m.fs.state[0]

def get_idaes_state_water(p, t):
    """Resolve mass-basis enthalpy and entropy of Water from a known T, P (IAPWS-95)."""
    # Use the native IDAES IAPWS-95 utility function to calculate molar enthalpy from T and P
    h_molar = value(htpxwater(T=t * units.K, P=p * units.Pa))
    state = _water_state_from_hp(h_molar, p)
    return value(state.enth_mass), value(state.entr_mass)


def tmexergy_co2(h, s, m_dot):
    """Mass-basis specific exergy for CO2, reference state 25 C / 1 atm."""
    t0 = 273.15 + 25
    p0 = 101325
    h0, s0 = get_idaes_state_co2(p0, t0)   # fixed: was 'y0' (undefined) in original
    ex = m_dot * ((h - h0) - (t0 * (s - s0)))
    return ex / 1000

def tmexergy_water(h, s, m_dot):
    """Mass-basis specific exergy for Water, reference state 25 C / 1 atm."""
    t0 = 273.15 + 25
    p0 = 101325
    h0, s0 = get_idaes_state_water(p0, t0)
    ex = m_dot * ((h - h0) - (t0 * (s - s0)))
    return ex / 1000

def turbine_co2(p_in, h_in, p_out, efficiency, turbine_fluid, m_dot):
    """
    Turbine calculation using IDAES swco2 (Span-Wagner) instead of CoolProp.
    Inputs/outputs match the original function's units and mass basis exactly.
    """
    # Inlet state (mass-basis h_in, p_in known -> get T, s)
    state_in = _co2_state_from_hp_mass_to_molar_wrapper = None  # placeholder, see note below

    # h_in comes in on a MASS basis (matching CoolProp's default convention) --
    # swco2's htpx/state blocks work on a MOLAR basis internally, so convert.
    MW_CO2 = 44.0095e-3  # kg/mol
    h_in_molar = h_in * MW_CO2

    state_in = _co2_state_from_hp(h_in_molar, p_in)
    s_in_molar = value(state_in.entr_mol)
    t_in = value(state_in.temperature)
    s_in = value(state_in.entr_mass)          # mass-basis, for tmexergy_co2 / return compatibility

    # Isentropic outlet: same entropy, outlet pressure -> solve for h_isentropic
    state_isen = _co2_state_from_sp(s_in_molar, p_out)
    h_isentropic_molar = value(state_isen.enth_mol)
    h_isentropic = h_isentropic_molar / MW_CO2   # back to mass basis

    isentrophic_specific_turbine_power = h_in - h_isentropic
    actual_specific_turbine_power = isentrophic_specific_turbine_power * efficiency
    h_out = h_in - actual_specific_turbine_power

    # Actual outlet state (mass h_out, p_out known -> get T, s)
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
def hx_co2(hot_fluid, p_hot_in, h_hot_in, flowrate_hot, hot_drop,
           cold_fluid, p_cold_in, h_cold_in, flowrate_cold, cold_drop, hx_eff):
    """
    Heat exchanger calculation using IDAES swco2 (Span-Wagner) instead of
    CoolProp. Both hot and cold streams are assumed to be CO2 (hot_fluid/
    cold_fluid arguments are kept only for call-site compatibility with the
    original hx() signature -- they're not used here).
    """
    MW_CO2 = 44.0095e-3  # kg/mol

    p_hot_out = p_hot_in - hot_drop
    p_cold_out = p_cold_in - cold_drop

    # --- Inlet states ---
    state_hot_in = _co2_state_from_hp(h_hot_in * MW_CO2, p_hot_in)
    t_hot_in = value(state_hot_in.temperature)

    state_cold_in = _co2_state_from_hp(h_cold_in * MW_CO2, p_cold_in)
    t_cold_in = value(state_cold_in.temperature)

    # --- Crossed enthalpies (hot stream cooled to cold-inlet T, and vice versa) ---
    h_hot_c_molar = value(htpx(T=t_cold_in * units.K, P=p_hot_out * units.Pa))
    h_hot_c = h_hot_c_molar / MW_CO2

    h_cold_h_molar = value(htpx(T=t_hot_in * units.K, P=p_cold_out * units.Pa))
    h_cold_h = h_cold_h_molar / MW_CO2

    # --- Duty (effectiveness-limited) ---
    q_hot = flowrate_hot * (h_hot_in - h_hot_c)
    q_cold = flowrate_cold * (h_cold_h - h_cold_in)
    q_max = min(q_hot, q_cold)
    q = q_max * hx_eff
    q_kw = q / 1000

    h_hot_out = h_hot_in - (q / flowrate_hot)
    h_cold_out = h_cold_in + (q / flowrate_cold)

    # --- Outlet states ---
    state_hot_out = _co2_state_from_hp(h_hot_out * MW_CO2, p_hot_out)
    t_hot_out = value(state_hot_out.temperature)

    state_cold_out = _co2_state_from_hp(h_cold_out * MW_CO2, p_cold_out)
    t_cold_out = value(state_cold_out.temperature)

    # --- Entropies (mass basis, for exergy) ---
    s_hot_in = value(state_hot_in.entr_mass)
    s_hot_out = value(state_hot_out.entr_mass)
    s_cold_in = value(state_cold_in.entr_mass)
    s_cold_out = value(state_cold_out.entr_mass)

    # --- Exergy balance (using the CO2-specific exergy function, not the original tmexergy) ---
    ex_hot_in = tmexergy_co2(h_hot_in, s_hot_in, flowrate_hot)
    ex_hot_out = tmexergy_co2(h_hot_out, s_hot_out, flowrate_hot)
    ex_cold_in = tmexergy_co2(h_cold_in, s_cold_in, flowrate_cold)
    ex_cold_out = tmexergy_co2(h_cold_out, s_cold_out, flowrate_cold)

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
    return (t_hot_out, p_hot_out, h_hot_out, t_cold_out, p_cold_out, h_cold_out, q_kw, exergy)

def hx2_co2_water(p_hot_in, h_hot_in, h_hot_out, flowrate_hot, hot_drop,
                   p_cold_in, h_cold_in, cold_drop, approach):
    """
    Heat exchanger calculation using IDAES swco2 (Span-Wagner) for the hot
    (CO2) side and iapws95 (IAPWS-95) for the cold (water) side, replacing
    CoolProp. Matches the original hx2() logic and units exactly -- hot
    outlet enthalpy is already known/given, cold outlet is set by an
    approach temperature, and the required cold flowrate is solved
    backward from the resulting duty.
    """
    MW_CO2 = 44.0095e-3   # kg/mol
    MW_H2O = 18.0153e-3   # kg/mol
    p_hot_out = p_hot_in - hot_drop
    p_cold_out = p_cold_in - cold_drop

    # --- Hot side (CO2) states ---
    state_hot_in = _co2_state_from_hp(h_hot_in * MW_CO2, p_hot_in)
    state_hot_out = _co2_state_from_hp(h_hot_out * MW_CO2, p_hot_out)

    q_kw = flowrate_hot * (h_hot_in - h_hot_out) / 1000

    # --- Cold side (Water) states ---
    state_cold_in = _water_state_from_hp(h_cold_in * MW_H2O, p_cold_in)
    t_cold_in = value(state_cold_in.temperature)

    # Same approach-temperature logic as the original: cold outlet = cold inlet + approach
    t_cold_out = t_cold_in + approach
    h_cold_out_molar = value(iapws95.htpx(T=t_cold_out * units.K, P=p_cold_out * units.Pa))
    h_cold_out = h_cold_out_molar / MW_H2O

    # Required cooling water flowrate to absorb the hot-side duty
    flowrate_cold = q_kw * 1000 / (h_cold_out - h_cold_in)

    state_cold_out = _water_state_from_hp(h_cold_out * MW_H2O, p_cold_out)

    # --- Entropies (mass basis, for exergy) ---
    s_hot_in = value(state_hot_in.entr_mass)
    s_hot_out = value(state_hot_out.entr_mass)
    s_cold_in = value(state_cold_in.entr_mass)
    s_cold_out = value(state_cold_out.entr_mass)

    # --- Exergy balance ---
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

# Flue Heater Start
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


def hx3_flue_co2(hot_fluid, t_hot_in, p_hot_in, flowrate_hot, hot_drop,
                  cold_fluid, p_cold_in, h_cold_in, h_cold_out, flowrate_cold,
                  cold_drop, approach2):
    """
    Flue-gas/CO2 heat exchanger: Peng-Robinson (hot, flue gas mixture) +
    swco2 Span-Wagner (cold, CO2), replacing CoolProp. hot_fluid is the
    CoolProp HEOS mixture string, parsed for mole fractions. The '|gas'
    and '|supercritical_gas' phase hints from the original are ignored,
    per your instructions.
    """
    MW_CO2 = 44.0095e-3  # kg/mol

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

    # --- Hot outlet: same approach-temperature logic as the original ---
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
    ex_eff = ex_cold_change / ex_hot_change

    exergy = {
        'hot_in': ex_hot_in, 'hot_out': ex_hot_out,
        'cold_in': ex_cold_in, 'cold_out': ex_cold_out,
        'hot_change': ex_hot_change, 'cold_change': ex_cold_change,
        'destroyed': ex_des, 'efficiency': ex_eff,
    }
    return (t_hot_out, p_hot_out, h_hot_out, p_cold_out, q_kw, exergy)
# Flue Heater End


def compressor_co2(p_in, h_in, p_out, eff, compressor_fluid, m_dot):
    """
    Compressor calculation using IDAES swco2 (Span-Wagner) instead of
    CoolProp. Inputs/outputs match the original function's units and
    mass basis exactly.
    """
    MW_CO2 = 44.0095e-3  # kg/mol

    # h_in is on a MASS basis (matching CoolProp's convention) -- convert
    # to molar for the swco2 state block.
    h_in_molar = h_in * MW_CO2

    # --- Inlet state ---
    state_in = _co2_state_from_hp(h_in_molar, p_in)
    s_in_molar = value(state_in.entr_mol)
    t_in = value(state_in.temperature)
    s_in = value(state_in.entr_mass)   # mass-basis, for tmexergy_co2

    # --- Isentropic outlet: same entropy, outlet pressure -> solve for h_isentropic ---
    state_isen = _co2_state_from_sp(s_in_molar, p_out)
    h_isentropic_molar = value(state_isen.enth_mol)
    h_isentropic = h_isentropic_molar / MW_CO2   # back to mass basis

    isentrophic_specific_compressor_power = h_isentropic - h_in
    actual_specific_compressor_power = isentrophic_specific_compressor_power / eff
    h_out = h_in + actual_specific_compressor_power

    # --- Actual outlet state ---
    h_out_molar = h_out * MW_CO2
    state_out = _co2_state_from_hp(h_out_molar, p_out)
    t_out = value(state_out.temperature)
    s_out = value(state_out.entr_mass)

    actual_compressor_power = actual_specific_compressor_power * m_dot / 1000

    ex_in = tmexergy_co2(h_in, s_in, m_dot)
    ex_out = tmexergy_co2(h_out, s_out, m_dot)
    ex_des = actual_compressor_power - (ex_out - ex_in)
    ex_eff = (ex_out - ex_in) / actual_compressor_power

    exergy = {
        'in': ex_in,
        'out': ex_out,
        'work': actual_compressor_power,
        'destroyed': ex_des,
        'efficiency': ex_eff,
    }
    return (t_out, h_out, actual_compressor_power, exergy)
    
def mixer_co2(massflow1, p1, h1, massflow2, p2, h2, fluid):
    """
    Mixer calculation using IDAES swco2 (Span-Wagner) instead of CoolProp.
    Both inlets and the outlet are CO2. Inputs/outputs match the original
    function's units and mass basis exactly.
    """
    MW_CO2 = 44.0095e-3  # kg/mol

    total_massflow = massflow1 + massflow2

    # --- Inlet states ---
    state1 = _co2_state_from_hp(h1 * MW_CO2, p1)
    s1 = value(state1.entr_mass)

    state2 = _co2_state_from_hp(h2 * MW_CO2, p2)
    s2 = value(state2.entr_mass)

    # --- Mixed outlet: simple average pressure, mass-weighted enthalpy ---
    p = (p1 + p2) / 2
    h = ((massflow1 * h1) + (massflow2 * h2)) / total_massflow

    ex1 = tmexergy_co2(h1, s1, massflow1)
    ex2 = tmexergy_co2(h2, s2, massflow2)

    # --- Outlet state (mixed h, p known -> get T, s) ---
    state_out = _co2_state_from_hp(h * MW_CO2, p)
    s = value(state_out.entr_mass)
    t = value(state_out.temperature)

    ex = tmexergy_co2(h, s, total_massflow)
    ex_des = (ex1 + ex2) - ex

    exergy = {
        'ex1': ex1,
        'ex2': ex2,
        'ex': ex,
        'destroyed': ex_des,
    }
    return (p, h, t, total_massflow, exergy)


def _cost_co2_turbomachine(p_in, h_in, p_out, m_dot, efficiency, compressor,
                            equipment, hot_temp_c=None, n_equip=1):
    """
    Cost a CO2 turbine or compressor using IDAES's dedicated Supercritical
    CO2 Costing Module -- part of the NETL Power Plant Costing Library
    (idaes.models_extra.power_generation.costing.power_plant_capcost,
    QGESSCostingData.get_sCO2_unit_cost) -- rather than the generic
    coal/NGCC-baseline accounts (get_PP_costing) used in an earlier version
    of this function. This module was purpose-built for sCO2 cycles, from:

      - White, Gray, Plunkett, Shelton, Weiland & Shultz, NETL_PUB_21490,
        "Techno-economic Evaluation of Utility-Scale Power Plants Based on
        the Indirect sCO2 Brayton Cycle" (2017)
      - Weiland, Lance & Pidaparti, "sCO2 Power Cycle Component Cost
        Correlations from DOE Data Spanning Multiple Scales and
        Applications," ASME Turbo Expo 2019, GT2019-90493

    It provides named equipment categories -- confirmed against the actual
    installed source of get_sCO2_unit_cost: "Axial turbine", "Radial
    turbine", "Coal-fired heater", "Natural gas-fired heater", "Recuperator"
    (all SINGULAR -- an earlier version of this function used the plural
    forms from the documentation prose and got a KeyError, since the real
    dictionary keys don't match the docs) -- scaled on shaft power (MW_sh),
    with a temperature correction factor applied above 550 C for the
    equipment types above (compressors are not in that list and don't take
    one):

        equipment_cost = reference_cost * scaled_param**alpha * temp_factor

    Mechanically this follows the same pattern as the SSLW pump-costing
    example and the earlier get_PP_costing version of this function: build
    a real, standalone swco2 PressureChanger unit model, fully fix its
    physical operating point so it reproduces the already-solved
    turbine_co2()/compressor_co2() result for this state, attach the sCO2
    costing block, and solve the combined (physics + costing) model.

    Returns total_plant_cost in raw USD (the underlying Var is in $MM;
    get_sCO2_unit_cost's own default CE_index_year is "2018", not the 2017
    publication year of the underlying Weiland et al. paper -- confirmed
    from the installed source, not assumed). Per the module's own
    documentation, "There are currently no estimates for the total plant
    cost components, so bare erected cost will be the same as total plant
    cost for now" -- i.e. this is equipment + bare-erection cost only, not
    a full installed/EPC cost.

    PLEASE VERIFY before trusting the $ this returns -- the turbine/heater
    equipment strings above are confirmed against real installed IDAES
    source (see get_sCO2_unit_cost's temperature-correction gate list), but
    the COMPRESSOR strings ("IG centrifugal compressor" / "Barrel type
    compressor", singular, below) are inferred by matching that same
    singular-naming pattern -- not directly confirmed, since compressors
    don't appear in that gate list. If you still get a KeyError on a
    compressor equipment string, run this against your install to get the
    real keys directly:

        from idaes.models_extra.power_generation.costing.power_plant_capcost import load_sCO2_costing_dictionary
        print(list(load_sCO2_costing_dictionary().keys()))

      * `equipment` must exactly match one of the module's category
        strings (case-sensitive). sco2's defaults are "Radial turbine" for
        the turbine (this cycle is under 10 MW, below the ~50 MWe range
        where axial becomes the more efficient/economical choice per
        Weiland et al.'s own sizing guidance -- note the radial-turbine
        correlation's own fitted range is 8-35 MW_sh, so if turbine_power
        ends up below ~8 MW you're extrapolating slightly below where the
        correlation was validated) and "IG centrifugal compressor"
        (integrally-geared centrifugal -- valid 1.5-200 MW_sh, comfortably
        covering a sub-10-MW machine) for both compressors. If your actual
        machines differ, override turbine_cost_equipment /
        *_compressor_cost_equipment on sco2's constructor ("Axial turbine"
        scales the same way as "Radial turbine"; "Barrel type compressor"
        scales on inlet volumetric flow rather than shaft power and would
        need a different scaled_param than what this function builds).
      * `scaled_param` is built below as an explicit Var in MW, tied to the
        unit's work_mechanical via a unit-converting constraint --
        confirmed necessary from source: get_sCO2_unit_cost calls
        scaled_param.get_units() on whatever is passed in and uses that
        verbatim, so it must be a real Var/Param (not a bare expression)
        and must genuinely be in MW (not IDAES's native Watts), since the
        correlation's fractional exponent gets applied to the raw number.
        The work_mechanical sign convention assumed (positive = work added
        to the fluid, i.e. compressors; negative = work extracted, i.e.
        turbines) is standard for IDAES PressureChanger, but confirm it
        against your version if the costing solve produces a nonsensical
        (e.g. negative-cost) result.
      * `hot_temp_c` (the equipment's hottest operating temperature, in
        Celsius) only changes the result once it exceeds 550 C -- sco2
        passes the turbine inlet temperature for the turbine and the
        compressor outlet temperature for each compressor, as the hottest
        point of each respective machine.
    """
    MW_CO2 = 44.0095e-3
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

    # get_sCO2_unit_cost's correlations are fit against shaft power in MW
    # ("MWsh" per Weiland et al.). It calls scaled_param.get_units() on
    # whatever is passed in and preserves those units verbatim into its own
    # Var -- so this must be (a) a real Var/Param, not a bare -expression
    # (general Pyomo expressions aren't guaranteed to carry .get_units()),
    # and (b) genuinely in MW, not IDAES's native Watts, or the correlation's
    # fractional exponent would be applied to a number ~1e6x too large.
    # Building an explicit converting Var+Constraint sidesteps both issues.
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
    # total_plant_cost is in $MM (2018-dollar basis by default per this
    # get_sCO2_unit_cost signature's CE_index_year="2018"); convert to raw USD
    return value(m.fs.unit.costing.total_plant_cost) * 1e6


class sco2:
    def __init__(self, flue_hx_approach, p1, bypass, htr_effectiveness, ltr_effectiveness, drop_hot, drop_cold, flue, bc, cooler,
                 turbine_cost_equipment="Radial turbine",
                 main_compressor_cost_equipment="IG centrifugal compressor",
                 bypass_compressor_cost_equipment="IG centrifugal compressor",
                 n_turbine_equip=1,
                 n_main_compressor_equip=1,
                 n_bypass_compressor_equip=1):
        self.flue_hx_approach = flue_hx_approach
        self.p1 = p1
        self.bypass = bypass
        self.htr_effectiveness = htr_effectiveness
        self.ltr_effectiveness = ltr_effectiveness
        self.htr_hot_drop = drop_hot[0]
        self.ltr_hot_drop = drop_hot[1]
        self.bt_hot_drop = drop_hot[2]
        self.condenser_hot_drop = drop_hot[3]
        self.heater_hot_drop = drop_hot[4]
        self.fluegas_hx_hot_drop = drop_hot[5]
        self.htr_cold_drop = drop_cold[0]
        self.ltr_cold_drop = drop_cold[1]
        self.bt_cold_drop = drop_cold[2]
        self.condenser_cold_drop = drop_cold[3]
        self.heater_cold_drop = drop_cold[4]
        self.fluegas_hx_cold_drop = drop_cold[5]
        self.t_bc_in = bc[0]
        self.p_bc_in = bc[1]
        self.m_bc_in = bc[2]
        self.bc_fluid = bc[3]
        self.t_flue_in = flue[0]
        self.p_flue_in = flue[1]
        self.m_flue_in = flue[2]
        self.flue_fluid = flue[3]
        self.t_cooler_in = cooler[0]
        self.p_cooler_in = cooler[1]
        self.cooler_fluid = cooler[2]
        self.turbine_power = None
        self.main_comp_power = None
        self.bypass_compressor_power = None
        self.net_power = None
        self.heat_input = None
        self.t1 = self.t_flue_in - self.flue_hx_approach
        self.t12 = None
        self.m = None
        self.t_flue_out = None
        self.p_flue_out = None
        # IDAES sCO2 Costing Module (NETL/QGESS Power Plant Costing Library)
        # equipment categories for _cost_co2_turbomachine() below. See that
        # function's docstring for the equipment-category options and what
        # to verify against your installed IDAES version.
        self.turbine_cost_equipment = turbine_cost_equipment
        self.main_compressor_cost_equipment = main_compressor_cost_equipment
        self.bypass_compressor_cost_equipment = bypass_compressor_cost_equipment
        self.n_turbine_equip = n_turbine_equip
        self.n_main_compressor_equip = n_main_compressor_equip
        self.n_bypass_compressor_equip = n_bypass_compressor_equip
        self.turbine_cost = None
        self.main_compressor_cost = None
        self.bypass_compressor_cost = None
        self.total_turbomachinery_cost = None
        # CO2 working-fluid (inventory) cost: required CO2 charge is taken
        # as 30% of the heater capacity in kW (i.e. 0.3 kg per kW), priced
        # at 18 INR/kg.
        self.co2_price_inr_per_kg = 18.0
        self.co2_charge_kg_per_kw = 0.3
        self.co2_mass_kg = None
        self.co2_fluid_cost = None


    def calc(self):
        # Inputs
        mass_flowrate = 1 # kg/s
        molar_flowrate = mass_flowrate*1000/44.0095

        t7 = 306.6 # K  Main Compressor Inlet Temperature (2.5℃ above critical temperature of 30.98℃) 
        p7 = 8_400_000 # Pa (84 bar) Main Compressor Inlet Pressure
        p2 = p7 + self.htr_hot_drop + self.ltr_hot_drop + self.bt_hot_drop + self.condenser_hot_drop # Pa  Turbine Outlet Temperature
        p8 = self.p1 + self.htr_cold_drop + self.ltr_cold_drop + self.fluegas_hx_cold_drop   # Pa   Main Compressor Outlet Pressure
        p92 = p8 - self.ltr_cold_drop  # 245 bar Bypass Compressor Outlet Pressure
        
        
        turbine_isen_eff = 0.85  # Turbine Efficiency
        main_compressor_eff = 0.82
        bypass_compressor_eff = 0.78


        h_bc_in, s_bc_in = get_idaes_state_co2(self.p_bc_in, self.t_bc_in)

        
        # 2 Turbine
        h1, s1 = get_idaes_state_co2(self.p1, self.t1)

        t2, h2, self.turbine_power, turbine_exergy = turbine_co2(self.p1, h1, p2, turbine_isen_eff, fluid, mass_flowrate)
        #print(self.t1-273.15, self.p1/100_000)
        #print(t2-273.15, p2/100_000, self.turbine_power)
        

        # 3 High Temperature Recuperator (HTR)
        p81 = p8 - self.ltr_cold_drop    #delete if p81 is not requried for anything else, it is calculated in LTR

        t11 = 450   # Initial Assumption
        p11 = (p81+p92)/2
        h11, s11 = get_idaes_state_co2(p11, t11)
        t3, p3, h3, self.t12, p12, h12, hx_htr, htr_exergy = hx_co2(fluid, p2, h2, mass_flowrate, self.htr_hot_drop, fluid, p11, h11, mass_flowrate, self.htr_cold_drop, self.htr_effectiveness)
        #print(t3-273.15, p3, h3, self.t12-273.15, p12, h12, hx_htr)

        # 4 Splitter

        flow_after_bypass = mass_flowrate * (1 - self.bypass)
        bypass_flow = mass_flowrate * self.bypass

        # 5 Low Temperature Recuperator (LTR)

        p08 =p8
        h7, s7 = get_idaes_state_co2(p7, t7)
        t08, h08, main_comp_power_08, compex = compressor_co2(p7, h7, p08, main_compressor_eff, fluid, flow_after_bypass)
        #print('Compressor', 't08 =', t08-273.15, 't7 =', t7-273.15, h08, main_comp_power_08)
        t4, p4, h4, t81, p81, h81, hx_ltr, ltr_exergy = hx_co2(fluid, p3, h3, mass_flowrate, self.ltr_hot_drop, fluid, p08, h08, flow_after_bypass, self.ltr_cold_drop, self.ltr_effectiveness)
        #print('LTR', t4-273.15, p4/100_000, h4, t81-273.15, p81/100_000, h81, hx_ltr)

        # 6 after bypass circuit

        t5 = t4
        p5 = p4
        h5 = h4
        

        # 7 Heat Exchanger for bottoming cycle


        t6, p6, h6, t_bc_out, p_bc_out, h_bc_out, hx_bt, bt_exergy = hx_co2(fluid, p5, h5, flow_after_bypass, self.bt_hot_drop, self.bc_fluid, self.p_bc_in, h_bc_in, self.m_bc_in, self.bt_cold_drop, 0.9)
        #print(t6-273.15, p6/100_000, h6, t_bc_out-273.15, p_bc_out/100_000, h_bc_out, hx_bt)

        # 8 Cooling water heat exchanger (Cooler) 6 to 7



        h_cooler_in, s_cooler_in = get_idaes_state_water(self.p_cooler_in, self.t_cooler_in)
        t_cooler_out, p_cooler_out, h_cooler_out, hx_condenser, flowrate_cooler, cooler_exergy = hx2_co2_water(p6, h6, h7, flow_after_bypass, self.condenser_hot_drop, self.p_cooler_in, h_cooler_in, self.condenser_cold_drop, 20)
        #print('Cooler', t_cooler_out-273.15, p_cooler_out/100_000, h_cooler_out, hx_condenser, flowrate_cooler)
        # 9 Main Compressor 7 to 8
            
        t8, h8, self.main_comp_power, main_comp_exergy = compressor_co2(p7, h7, p8, 0.82, fluid, flow_after_bypass)
        

        # 10 bypass flow 4 to 91


        t91 = t4
        p91 = p4
        h91 = h4

        # 11 Bypass Compressor 91 to 92

        h91, s91 = get_idaes_state_co2(p91, t91)
        t92, h92, self.bypass_compressor_power, bypass_comp_exergy = compressor_co2(p91, h91, p92, 0.78, fluid, bypass_flow)
        #print('Bypass Compressor', t92-273.15, h92, self.bypass_compressor_power)
        #print('p in =', p91, 'p out = ', p92)

        # 12 Mixer 81 & 92 to 10


        p10, h10, t10, massflow_mixer, mixer_exergy = mixer_co2(flow_after_bypass, p81, h81, bypass_flow, p92, h92, fluid)
        #print('Mixer Inlet', (flow_after_bypass, p81/100_000, h81, bypass_flow, p92/100_000, h92, fluid))
        #print('Mixer Outlet', p10/100_000, h10, t10-273.15, massflow_mixer)
        #print(p8, self.ltr_cold_drop)
        #print(self.p1, self.htr_cold_drop, self.ltr_cold_drop, self.fluegas_hx_cold_drop)

        # 13 To match Mixture Outlet Temperatue and Cold inlet stream Temperature of HTR

        while abs(t10-t11)>=0.1:
            t11 =t10
            p11 = p92
            h11, s11 = get_idaes_state_co2(p11, t11)
            t3, p3, h3, self.t12, p12, h12, hx_htr, htr_exergy = hx_co2(fluid, p2, h2, mass_flowrate, self.htr_hot_drop, fluid, p11, h11, mass_flowrate, self.htr_cold_drop, self.htr_effectiveness)
            p08 =p8
            h7, s7 = get_idaes_state_co2(p7, t7)
            t08, h08, main_comp_power_08, compex = compressor_co2(p7, h7, p08, main_compressor_eff, fluid, flow_after_bypass)
            t4, p4, h4, t81, p81, h81, hx_ltr, ltr_exergy = hx_co2(fluid, p3, h3, mass_flowrate, self.ltr_hot_drop, fluid, p08, h08, flow_after_bypass, self.ltr_cold_drop, self.ltr_effectiveness)
            t5 = t4
            p5 = p4
            h5 = h4
            t6, p6, h6, t_bc_out, p_bc_out, h_bc_out, hx_bt, bt_exergy = hx_co2(fluid, p5, h5, flow_after_bypass, self.bt_hot_drop, self.bc_fluid, self.p_bc_in, h_bc_in, self.m_bc_in, self.bt_cold_drop, 0.9)
            h_cooler_in, s_cooler_in = get_idaes_state_water(self.p_cooler_in, self.t_cooler_in)
            t_cooler_out, p_cooler_out, h_cooler_out, hx_condenser, flowrate_cooler, cooler_exergy = hx2_co2_water(p6, h6, h7, flow_after_bypass, self.condenser_hot_drop, self.p_cooler_in, h_cooler_in, self.condenser_cold_drop, 20)    
            t8, h8, self.main_comp_power, main_comp_exergy = compressor_co2(p7, h7, p8, 0.82, fluid, flow_after_bypass)
            t91 = t4
            p91 = p4
            h91 = h4
            h91, s91 = get_idaes_state_co2(p91, t91)
            t92, h92, self.bypass_compressor_power, bypass_comp_exergy = compressor_co2(p91, h91, p92, 0.78, fluid, bypass_flow)
            p10, h10, t10, massflow_mixer, mixer_exergy = mixer_co2(flow_after_bypass, p81, h81, bypass_flow, p92, h92, fluid)
            # Final Calculation
            self.net_power = self.turbine_power - self.main_comp_power - self.bypass_compressor_power


        # 14 Flue Gas Heat Exchanger


        t01 = self.t1
        p01 = self.p1
        h01, s01 = get_idaes_state_co2(p01, t01)
        h12, s12 = get_idaes_state_co2(p12, self.t12)
        specific_heat_input = (h01-h12)/1000
        


        self.t_flue_out, self.p_flue_out, h_flue_out, p01, self.heat_input, flue_hx_exergy = hx3_flue_co2(self.flue_fluid, self.t_flue_in, self.p_flue_in, self.m_flue_in, self.fluegas_hx_hot_drop, fluid, p12, h12, h1, mass_flowrate, 0, 20)
        self.t_flue_out = self.t12+20
        mole_fracs = _parse_flue_fluid_string(self.flue_fluid)
        mw_flue = _flue_mw(mole_fracs)
        
        state_flue_in = _flue_state_from_tpx(self.t_flue_in, self.p_flue_in, mole_fracs)
        h_flue_in = value(state_flue_in.enth_mol) / mw_flue   # mass basis, matching CoolProp's convention
        
        state_flue_out = _flue_state_from_tpx(self.t_flue_out, self.p_flue_out, mole_fracs)
        h_flue_out = value(state_flue_out.enth_mol) / mw_flue   # mass basis
        self.m = self.m_flue_in * (h_flue_in - h_flue_out)/(h1-h12)
        

        
        heater_ex_in = tmexergy_co2(h12, s12, mass_flowrate)
        heater_ex_out = tmexergy_co2(h01, s01, mass_flowrate)
        heater_exergy = heater_ex_out - heater_ex_in





        #UPDATE MASS
        # Inputs
        mass_flowrate = self.m #7.89   #kg/s  CO2 Mass Flowrate

        t7 = 306.6 # K  Main Compressor Inlet Temperature (2.5℃ above critical temperature of 30.98℃) 
        p7 = 8_400_000 # Pa (84 bar) Main Compressor Inlet Pressure
        p2 = p7 + self.htr_hot_drop + self.ltr_hot_drop + self.bt_hot_drop + self.condenser_hot_drop # Pa  Turbine Outlet Temperature
        p8 = self.p1 + self.htr_cold_drop + self.ltr_cold_drop + self.fluegas_hx_cold_drop   # Pa   Main Compressor Outlet Pressure
        p92 = p8 - self.ltr_cold_drop  # 245 bar Bypass Compressor Outlet Pressure
        
        
        turbine_isen_eff = 0.85  # Turbine Efficiency
        main_compressor_eff = 0.82
        bypass_compressor_eff = 0.78


        h_bc_in, s_bc_in = get_idaes_state_co2(self.p_bc_in, self.t_bc_in)

        
        # 2 Turbine
        h1, s1 = get_idaes_state_co2(self.p1, self.t1)

        t2, h2, self.turbine_power, turbine_exergy = turbine_co2(self.p1, h1, p2, turbine_isen_eff, fluid, mass_flowrate)
        #print(self.t1-273.15, self.p1/100_000)
        #print(t2-273.15, p2/100_000, self.turbine_power)
        

        # 3 High Temperature Recuperator (HTR)
        p81 = p8 - self.ltr_cold_drop    #delete if p81 is not requried for anything else, it is calculated in LTR

        t11 = 450   # Initial Assumption
        p11 = (p81+p92)/2
        h11, s11 = get_idaes_state_co2(p11, t11)
        t3, p3, h3, self.t12, p12, h12, hx_htr, htr_exergy = hx_co2(fluid, p2, h2, mass_flowrate, self.htr_hot_drop, fluid, p11, h11, mass_flowrate, self.htr_cold_drop, self.htr_effectiveness)
        #print(t3-273.15, p3, h3, self.t12-273.15, p12, h12, hx_htr)

        # 4 Splitter

        flow_after_bypass = mass_flowrate * (1 - self.bypass)
        bypass_flow = mass_flowrate * self.bypass

        # 5 Low Temperature Recuperator (LTR)

        p08 =p8
        h7, s7 = get_idaes_state_co2(p7, t7)
        t08, h08, main_comp_power_08, compex = compressor_co2(p7, h7, p08, main_compressor_eff, fluid, flow_after_bypass)
        #print('Compressor', 't08 =', t08-273.15, 't7 =', t7-273.15, h08, main_comp_power_08)
        t4, p4, h4, t81, p81, h81, hx_ltr, ltr_exergy = hx_co2(fluid, p3, h3, mass_flowrate, self.ltr_hot_drop, fluid, p08, h08, flow_after_bypass, self.ltr_cold_drop, self.ltr_effectiveness)
        #print('LTR', t4-273.15, p4/100_000, h4, t81-273.15, p81/100_000, h81, hx_ltr)

        # 6 after bypass circuit

        t5 = t4
        p5 = p4
        h5 = h4
        

        # 7 Heat Exchanger for bottoming cycle


        t6, p6, h6, t_bc_out, p_bc_out, h_bc_out, hx_bt, bt_exergy = hx_co2(fluid, p5, h5, flow_after_bypass, self.bt_hot_drop, self.bc_fluid, self.p_bc_in, h_bc_in, self.m_bc_in, self.bt_cold_drop, 0.9)
        #print(t6-273.15, p6/100_000, h6, t_bc_out-273.15, p_bc_out/100_000, h_bc_out, hx_bt)

        # 8 Cooling water heat exchanger (Cooler) 6 to 7



        h_cooler_in, s_cooler_in = get_idaes_state_water(self.p_cooler_in, self.t_cooler_in)
        t_cooler_out, p_cooler_out, h_cooler_out, hx_condenser, flowrate_cooler, cooler_exergy = hx2_co2_water(p6, h6, h7, flow_after_bypass, self.condenser_hot_drop, self.p_cooler_in, h_cooler_in, self.condenser_cold_drop, 20)
        #print('Cooler', t_cooler_out-273.15, p_cooler_out/100_000, h_cooler_out, hx_condenser, flowrate_cooler)
        # 9 Main Compressor 7 to 8
            
        t8, h8, self.main_comp_power, main_comp_exergy = compressor_co2(p7, h7, p8, 0.82, fluid, flow_after_bypass)
        

        # 10 bypass flow 4 to 91


        t91 = t4
        p91 = p4
        h91 = h4

        # 11 Bypass Compressor 91 to 92

        h91, s91 = get_idaes_state_co2(p91, t91)
        t92, h92, self.bypass_compressor_power, bypass_comp_exergy = compressor_co2(p91, h91, p92, 0.78, fluid, bypass_flow)
        #print('Bypass Compressor', t92-273.15, h92, self.bypass_compressor_power)
        #print('p in =', p91, 'p out = ', p92)

        # 12 Mixer 81 & 92 to 10


        p10, h10, t10, massflow_mixer, mixer_exergy = mixer_co2(flow_after_bypass, p81, h81, bypass_flow, p92, h92, fluid)
        #print('Mixer Inlet', (flow_after_bypass, p81/100_000, h81, bypass_flow, p92/100_000, h92, fluid))
        #print('Mixer Outlet', p10/100_000, h10, t10-273.15, massflow_mixer)
        #print(p8, self.ltr_cold_drop)
        #print(self.p1, self.htr_cold_drop, self.ltr_cold_drop, self.fluegas_hx_cold_drop)
        while abs(t10-t11)>=0.1:
            t11 =t10
            p11 = p92
            h11, s11 = get_idaes_state_co2(p11, t11)
            t3, p3, h3, self.t12, p12, h12, hx_htr, htr_exergy = hx_co2(fluid, p2, h2, mass_flowrate, self.htr_hot_drop, fluid, p11, h11, mass_flowrate, self.htr_cold_drop, self.htr_effectiveness)
            p08 =p8
            h7, s7 = get_idaes_state_co2(p7, t7)
            t08, h08, main_comp_power_08, compex = compressor_co2(p7, h7, p08, main_compressor_eff, fluid, flow_after_bypass)
            t4, p4, h4, t81, p81, h81, hx_ltr, ltr_exergy = hx_co2(fluid, p3, h3, mass_flowrate, self.ltr_hot_drop, fluid, p08, h08, flow_after_bypass, self.ltr_cold_drop, self.ltr_effectiveness)
            t5 = t4
            p5 = p4
            h5 = h4
            t6, p6, h6, t_bc_out, p_bc_out, h_bc_out, hx_bt, bt_exergy = hx_co2(fluid, p5, h5, flow_after_bypass, self.bt_hot_drop, self.bc_fluid, self.p_bc_in, h_bc_in, self.m_bc_in, self.bt_cold_drop, 0.9)
            h_cooler_in, s_cooler_in = get_idaes_state_water(self.p_cooler_in, self.t_cooler_in)
            t_cooler_out, p_cooler_out, h_cooler_out, hx_condenser, flowrate_cooler, cooler_exergy = hx2_co2_water(p6, h6, h7, flow_after_bypass, self.condenser_hot_drop, self.p_cooler_in, h_cooler_in, self.condenser_cold_drop, 20)    
            t8, h8, self.main_comp_power, main_comp_exergy = compressor_co2(p7, h7, p8, 0.82, fluid, flow_after_bypass)
            t91 = t4
            p91 = p4
            h91 = h4
            h91, s91 = get_idaes_state_co2(p91, t91)
            t92, h92, self.bypass_compressor_power, bypass_comp_exergy = compressor_co2(p91, h91, p92, 0.78, fluid, bypass_flow)
            p10, h10, t10, massflow_mixer, mixer_exergy = mixer_co2(flow_after_bypass, p81, h81, bypass_flow, p92, h92, fluid)
            # Final Calculation
            self.net_power = self.turbine_power - self.main_comp_power - self.bypass_compressor_power



        # 14 Flue Gas Heat Exchanger


        t01 = self.t1
        p01 = self.p1
        h01, s01 = get_idaes_state_co2(p01, t01)
        h12, s12 = get_idaes_state_co2(p12, self.t12)
        specific_heat_input = (h01-h12)/1000
        


        self.t_flue_out, self.p_flue_out, h_flue_out, p01, self.heat_input, flue_hx_exergy = hx3_flue_co2(self.flue_fluid, self.t_flue_in, self.p_flue_in, self.m_flue_in, self.fluegas_hx_hot_drop, fluid, p12, h12, h1, mass_flowrate, 0, 20)
        self.t_flue_out = self.t12+20
        mole_fracs = _parse_flue_fluid_string(self.flue_fluid)
        mw_flue = _flue_mw(mole_fracs)
        
        state_flue_in = _flue_state_from_tpx(self.t_flue_in, self.p_flue_in, mole_fracs)
        h_flue_in = value(state_flue_in.enth_mol) / mw_flue   # mass basis, matching CoolProp's convention
        
        state_flue_out = _flue_state_from_tpx(self.t_flue_out, self.p_flue_out, mole_fracs)
        h_flue_out = value(state_flue_out.enth_mol) / mw_flue   # mass basis
        

        
        heater_ex_in = tmexergy_co2(h12, s12, mass_flowrate)
        heater_ex_out = tmexergy_co2(h01, s01, mass_flowrate)
        heater_exergy = heater_ex_out - heater_ex_in

        # Purchased-equipment cost factors (material, temperature, pressure,
        # piping), applied via uo.py's equipment_cost() as:
        #   Adjusted Cost = Base_Cost * material_factor * temperature_factor
        #                    * pressure_factor * (1 + piping_factor)
        # Set to no-ops here (1, 1, 1, 0) so the reported cost matches the
        # unfactored base correlation cost.
        hx_cost_factors_htr = {
            'material_factor': 2.9,
            'temperature_factor': 2.1,
            'pressure_factor': 1.9,
            'piping_factor': 0.46
        }
        hx_cost_factors_ltr = {
            'material_factor': 2.9,
            'temperature_factor': 1,
            'pressure_factor': 1.9,
            'piping_factor': 0.46
        }
        hx_cost_factors_heater = {
            'material_factor': 2.9,
            'temperature_factor': 2.1,
            'pressure_factor': 1.9,
            'piping_factor': 0.46
        }
        hx_cost_factors_bt = {
            'material_factor': 2.9,
            'temperature_factor': 1,
            'pressure_factor': 1.9,
            'piping_factor': 0.46
        }
        hx_cost_factors_cooler = {
            'material_factor': 2.9,
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
        # each cost block up to a March-2026 basis:
        #   - TURBOMACHINERY_ESCALATION: escalates the turbine/main
        #     compressor/bypass compressor costs from IDAES's sCO2 Costing
        #     Module's 2018 basis (see _cost_co2_turbomachine()'s docstring)
        #     to March 2026.
        #   - REMAINING_COST_ESCALATION: escalates everything else (the 5
        #     heat exchangers plus the cooling tower, all priced via uo.py's
        #     2024-basis COST_TABLE correlations) from 2024 to March 2026.
        TURBOMACHINERY_ESCALATION = 846.43 / 603.1
        REMAINING_COST_ESCALATION = 103.625 / 102.6166667

        # Purchased-equipment cost for the 5 heat exchangers in this cycle
        # (HTR, LTR, bottoming-cycle HX, cooler/condenser, flue-gas heater),
        # done the same way as the desorber/evaporator/absorber/condenser in
        # libr.py: back-calculate the required area from duty, terminal
        # temperatures and an assumed overall heat-transfer coefficient (U)
        # via Area = Q / (U * LMTD), then price that area with uo.py's
        # heat-exchanger cost correlation -- NOT via any IDAES
        # heat-exchanger costing method. All terminal temperatures below are
        # taken from the final, mass-corrected pass through the cycle above.
        htr_temp = {"t_hot_in": t2, "t_hot_out": t3, "t_cold_in": t11, "t_cold_out": self.t12}
        ltr_temp = {"t_hot_in": t3, "t_hot_out": t4, "t_cold_in": t08, "t_cold_out": t81}
        bt_temp = {"t_hot_in": t5, "t_hot_out": t6, "t_cold_in": self.t_bc_in, "t_cold_out": t_bc_out}
        condenser_temp = {"t_hot_in": t6, "t_hot_out": t7, "t_cold_in": self.t_cooler_in, "t_cold_out": t_cooler_out}
        heater_temp = {"t_hot_in": self.t_flue_in, "t_hot_out": self.t_flue_out, "t_cold_in": self.t12, "t_cold_out": self.t1}

        # Assumed overall heat-transfer coefficients (W/m2-K) -- screening-
        # level placeholders, same spirit as libr.py's hardcoded hx_*_u
        # values; tune these if better plant data is available. HTR/LTR/BT
        # are sCO2-side (printed-circuit-type) exchangers, much higher U
        # than a typical shell-and-tube unit; the cooler is sCO2-to-water;
        # the heater is flue-gas-side dominated, same figure as uo.py's
        # gas-gas U_TABLE entry.
        htr_u = 500
        ltr_u = 500
        bt_u = 500
        condenser_u = 500
        heater_u = 60

        htr_f = 1
        ltr_f = 1
        bt_f = 0.8
        condenser_f = 0.8
        heater_f = 1

        # Each heat-exchanger cost below is escalated from uo.py's 2024
        # basis to March 2026 via REMAINING_COST_ESCALATION (defined above).
        htr_area, htr_cost_value = _hx_cost_given_u(hx_htr, htr_temp, htr_u, htr_f, hx_cost_factors_htr)
        htr_cost_value = _escalate(htr_cost_value, REMAINING_COST_ESCALATION)
        htr_cost = {'area_m2': htr_area, 'u_assumed': htr_u, 'cost': htr_cost_value}

        ltr_area, ltr_cost_value = _hx_cost_given_u(hx_ltr, ltr_temp, ltr_u, ltr_f, hx_cost_factors_ltr)
        ltr_cost_value = _escalate(ltr_cost_value, REMAINING_COST_ESCALATION)
        ltr_cost = {'area_m2': ltr_area, 'u_assumed': ltr_u, 'cost': ltr_cost_value}

        bt_area, bt_cost_value = _hx_cost_given_u(hx_bt, bt_temp, bt_u, bt_f, hx_cost_factors_bt)
        bt_cost_value = _escalate(bt_cost_value, REMAINING_COST_ESCALATION)
        bt_cost = {'area_m2': bt_area, 'u_assumed': bt_u, 'cost': bt_cost_value}

        condenser_hx_area, condenser_hx_cost_value = _hx_cost_given_u(hx_condenser, condenser_temp, condenser_u, condenser_f, hx_cost_factors_cooler)
        condenser_hx_cost_value = _escalate(condenser_hx_cost_value, REMAINING_COST_ESCALATION)
        condenser_hx_cost = {'area_m2': condenser_hx_area, 'u_assumed': condenser_u, 'cost': condenser_hx_cost_value}

        heater_hx_area, heater_hx_cost_value = _hx_cost_given_u(self.heat_input, heater_temp, heater_u, heater_f, hx_cost_factors_heater)
        heater_hx_cost_value = _escalate(heater_hx_cost_value, REMAINING_COST_ESCALATION)
        heater_hx_cost = {'area_m2': heater_hx_area, 'u_assumed': heater_u, 'cost': heater_hx_cost_value}

        # Cooling tower, sized on the volumetric flow rate of the cooling
        # water it serves -- the same water stream that goes through the
        # cooler/condenser above (flowrate_cooler, kg/s, from the final,
        # mass-corrected hx2_co2_water() call). Priced with uo.py's
        # equipment_cost() power-law correlation, same pattern as every
        # other component here: Cost = Base_Cost * (Capacity/Base_Capacity)
        # ** Exponent, with Base_Cost = 5000, Base_Capacity = 1 m3/hr,
        # Exponent = 0.63 (see uo.py's COST_TABLE['cooling_tower']).
        # flowrate_cooler is a mass flow (kg/s); equipment_cost() needs
        # volumetric flow (m3/hr), so it's converted here using the density
        # of water at the cooling tower's inlet condition (liquid water is
        # close enough to incompressible that the inlet-state density is a
        # fine stand-in for the tower's actual operating density).
        cooling_water_density = CP.PropsSI('D', 'T', self.t_cooler_in, 'P', self.p_cooler_in, 'IF97::Water')
        cooling_tower_flow_m3hr = (flowrate_cooler / cooling_water_density) * 3600.0
        cooling_tower_cost_value = equipment_cost('cooling_tower', cooling_tower_flow_m3hr, cooling_tower_cost_factors)
        cooling_tower_cost_value = _escalate(cooling_tower_cost_value, REMAINING_COST_ESCALATION)
        cooling_tower_cost = {
            'flow_kgs': flowrate_cooler,
            'flow_m3hr': cooling_tower_flow_m3hr,
            'cost': cooling_tower_cost_value,
        }
        self.cooling_tower_cost = cooling_tower_cost

        # Total purchased-equipment cost for these 5 heat exchangers plus
        # the cooling tower.
        hx_costs = {
            'htr': htr_cost,
            'ltr': ltr_cost,
            'bt': bt_cost,
            'condenser': condenser_hx_cost,
            'heater': heater_hx_cost,
            'cooling_tower': cooling_tower_cost,
            'total': (
                _safe_cost(htr_cost)
                + _safe_cost(ltr_cost)
                + _safe_cost(bt_cost)
                + _safe_cost(condenser_hx_cost)
                + _safe_cost(heater_hx_cost)
                + _safe_cost(cooling_tower_cost)
            )
        }
        self.hx_costs = hx_costs
        #END UPDATE MASS




        #15

        self.net_power = self.turbine_power - self.main_comp_power - self.bypass_compressor_power
        net_efficiency = 100 * self.net_power/self.heat_input

        # 16 Turbomachinery cost (turbine, main compressor, bypass/recompression
        # compressor), via IDAES's dedicated Supercritical CO2 Costing Module
        # (NETL/QGESS Power Plant Costing Library, get_sCO2_unit_cost) -- see
        # _cost_co2_turbomachine()'s docstring for what to verify before
        # trusting these numbers. Wrapped in try/except so an equipment-
        # category mismatch against your installed IDAES version reports a
        # clear warning instead of crashing the whole cycle solve.
        # Each turbomachinery cost below is escalated from IDAES's sCO2
        # Costing Module's 2018 basis to March 2026 via
        # TURBOMACHINERY_ESCALATION (defined above), before the USD->INR
        # (*95) conversion applied later in total_plant_cost.
        try:
            self.turbine_cost = _cost_co2_turbomachine(
                self.p1, h1, p2, mass_flowrate, turbine_isen_eff, compressor=False,
                equipment=self.turbine_cost_equipment,
                hot_temp_c=self.t1 - 273.15,   # turbine inlet: hottest point of an expander
                n_equip=self.n_turbine_equip)
            self.turbine_cost = _escalate(self.turbine_cost, TURBOMACHINERY_ESCALATION)
        except Exception as err:
            print('Turbine costing failed -- check turbine_cost_equipment against your installed IDAES sCO2 costing module:', err)
            self.turbine_cost = None

        try:
            self.main_compressor_cost = _cost_co2_turbomachine(
                p7, h7, p8, flow_after_bypass, main_compressor_eff, compressor=True,
                equipment=self.main_compressor_cost_equipment,
                hot_temp_c=t8 - 273.15,   # compressor outlet: hottest point of a compressor
                n_equip=self.n_main_compressor_equip)
            self.main_compressor_cost = _escalate(self.main_compressor_cost, TURBOMACHINERY_ESCALATION)
        except Exception as err:
            print('Main compressor costing failed -- check main_compressor_cost_equipment against your installed IDAES sCO2 costing module:', err)
            self.main_compressor_cost = None

        try:
            self.bypass_compressor_cost = _cost_co2_turbomachine(
                p91, h91, p92, bypass_flow, bypass_compressor_eff, compressor=True,
                equipment=self.bypass_compressor_cost_equipment,
                hot_temp_c=t92 - 273.15,
                n_equip=self.n_bypass_compressor_equip)
            self.bypass_compressor_cost = _escalate(self.bypass_compressor_cost, TURBOMACHINERY_ESCALATION)
        except Exception as err:
            print('Bypass compressor costing failed -- check bypass_compressor_cost_equipment against your installed IDAES sCO2 costing module:', err)
            self.bypass_compressor_cost = None

        self.turbine_cost = self.turbine_cost * 95
        self.main_compressor_cost = self.main_compressor_cost * 95
        self.bypass_compressor_cost = self.bypass_compressor_cost * 95

        if None not in (self.turbine_cost, self.main_compressor_cost, self.bypass_compressor_cost):
            self.total_turbomachinery_cost = self.turbine_cost + self.main_compressor_cost + self.bypass_compressor_cost
        else:
            self.total_turbomachinery_cost = None

        # CO2 working-fluid cost (INR): required CO2 charge = 0.3 * heater
        # capacity (kW) in kg, priced at 18 INR/kg. self.heat_input is the
        # flue-gas heater duty in kW (from hx3_flue_co2).
        self.co2_mass_kg = self.co2_charge_kg_per_kw * self.heat_input
        self.co2_fluid_cost = self.co2_mass_kg * self.co2_price_inr_per_kg

        # Whole-cycle cost: turbomachinery (turbine, main compressor,
        # bypass compressor) + the 5 heat exchangers and cooling tower above
        # + the CO2 working-fluid charge.
        if None not in (self.turbine_cost, self.main_compressor_cost, self.bypass_compressor_cost):
            self.total_plant_cost = (
                (self.turbine_cost
                + self.main_compressor_cost
                + self.bypass_compressor_cost)
                + hx_costs['total']
                + self.co2_fluid_cost
            )
        else:
            self.total_plant_cost = None



        if __name__ == "__main__":
            # Define the nodes (processes/points in the exergy flow)
            # The index of the label corresponds to its position in the diagram.
            # For example, 'Input Exergy' is node 0, 'Turbine' is node 1, etc.
            nodes = [
                'Heater',  # 0
                'Turbine', # 1
                'Main Compressor',   # 2
                'Bypass Compressor',   # 3
                'HTR', # 4
                'LTR', # 5
                'Bottoming Cycle HX',   # 6
                'Cooler',   # 7
                'Mixer',   # 8
                'Shaft Work Output',   # 9
                'Exergy Destruction (Heater)', # 10
                'Exergy Destruction (Turbine)', # 11
                'Exergy Destruction (Main Compressor)', # 12
                'Exergy Destruction (Bypass Compressor)', # 13
                'Exergy Destruction (HTR)', # 14
                'Exergy Destruction (LTR)', # 15
                'Exergy Destruction (Bottoming Cycle HX)', # 16
                'Exergy Destruction (Cooler)', # 17
                'Exergy Destruction (Mixer)', # 18
                'Exergy Loss (Cooling Water)', # 19
                'Splitter', # 20
                'Flue Exergy', # 21
                'Useful Work', # 22
                'CW Loss', # 23
                '91', # 24
                'BT', # 25
                'SCO2' #26
            ]

            # Define the exergy flows (source, target, value)
            # The source and target are the indices of the nodes defined above.
            # Values represent the exergy quantities (e.g., in kW or MJ/s).
            exergy_flows = [
                # Input Exergy to Combustion Chamber
                {'source': 21, 'target': 0, 'value': heater_exergy}, # 1000 units of exergy from fuel
                {'source': 0, 'target': 10, 'value': 0},
                {'source': 0, 'target': 1, 'value': heater_ex_out},

                # Combustion Chamber outputs
                {'source': 1, 'target': 9, 'value': turbine_exergy['work']},
                {'source': 9, 'target': 2, 'value': main_comp_exergy['work']}, # Exergy to Turbine
                {'source': 9, 'target': 3, 'value': bypass_comp_exergy['work']}, # Exergy destruction in Combustion
                {'source': 9, 'target': 22, 'value': (turbine_exergy['work'] - main_comp_exergy['work'] - bypass_comp_exergy['work'])},
                {'source': 1, 'target': 11, 'value': turbine_exergy['destroyed']},
                {'source': 1, 'target': 4, 'value': htr_exergy['hot_in']},

                {'source': 4, 'target': 5, 'value': htr_exergy['hot_out']},
                {'source': 4, 'target': 14, 'value': htr_exergy['destroyed']},
                {'source': 4, 'target': 0, 'value': htr_exergy['cold_out']},


                {'source': 5, 'target': 20, 'value': ltr_exergy['hot_out']},
                {'source': 5, 'target': 15, 'value': ltr_exergy['destroyed']},
                {'source': 5, 'target': 8, 'value': ltr_exergy['cold_out']},

                {'source': 20, 'target': 6, 'value': (ltr_exergy['hot_out']) * (1 - self.bypass)},
                {'source': 20, 'target': 3, 'value': (ltr_exergy['hot_out']) *  self.bypass},

                {'source': 6, 'target': 25, 'value': bt_exergy['cold_change']},
                {'source': 6, 'target': 16, 'value': bt_exergy['destroyed']},
                {'source': 6, 'target': 7, 'value': bt_exergy['hot_out']},

                {'source': 7, 'target': 23, 'value': cooler_exergy['cold_change']},
                {'source': 7, 'target': 17, 'value': cooler_exergy['destroyed']},
                {'source': 7, 'target': 2, 'value': cooler_exergy['hot_out']},

                {'source': 2, 'target': 5, 'value': main_comp_exergy['out']},
                {'source': 2, 'target': 12, 'value': main_comp_exergy['destroyed']},

                {'source': 3, 'target': 8, 'value': bypass_comp_exergy['out']},
                {'source': 3, 'target': 13, 'value': bypass_comp_exergy['destroyed']},

                {'source': 8, 'target': 4, 'value': mixer_exergy['ex']},
                {'source': 8, 'target': 18, 'value': mixer_exergy['destroyed']}


            ]
            exergy_flows2 = [
                # Input Exergy to Combustion Chamber
                {'source': 21, 'target': 26, 'value': heater_exergy}, # 1000 units of exergy from fuel


                # Combustion Chamber outputs 
                {'source': 26, 'target': 22, 'value': (turbine_exergy['work'] - main_comp_exergy['work'] - bypass_comp_exergy['work'])},
                {'source': 26, 'target': 11, 'value': turbine_exergy['destroyed']},


                {'source': 26, 'target': 14, 'value': htr_exergy['destroyed']},




                {'source': 26, 'target': 15, 'value': ltr_exergy['destroyed']},





                {'source': 26, 'target': 25, 'value': bt_exergy['cold_change']},
                {'source': 26, 'target': 16, 'value': bt_exergy['destroyed']},


                {'source': 26, 'target': 23, 'value': cooler_exergy['cold_change']},
                {'source': 26, 'target': 17, 'value': cooler_exergy['destroyed']},



                {'source': 26, 'target': 12, 'value': main_comp_exergy['destroyed']},


                {'source': 26, 'target': 13, 'value': bypass_comp_exergy['destroyed']},


                {'source': 26, 'target': 18, 'value': mixer_exergy['destroyed']}


            ]

        # Generate and show the Sankey diagram
            sankey_figure = generate_sankey_diagram(exergy_flows, nodes)
            sankey_figure.show()
            sankey_figure = generate_sankey_diagram(exergy_flows2, nodes)
            sankey_figure.show()

        

        useful_work = (turbine_exergy['work'] - main_comp_exergy['work'] - bypass_comp_exergy['work']) + bt_exergy['cold_change']
        exergy_efficiency = self.net_power/heater_exergy
        
        total= (turbine_exergy['work'] - main_comp_exergy['work'] - bypass_comp_exergy['work']) +  bt_exergy['cold_change'] + bt_exergy['destroyed']
        #print('t_flue_in(C) = ', self.t_flue_in - 273.15, 't1 = ', self.t1-273.15, 'p1 = ', self.p1/100_000, 'hx heater = ', self.heat_input, 'm = ', mass_flowrate, 'net_power = ', self.net_power)
        #print('Turbine cost = ', self.turbine_cost, 'Main compressor cost = ', self.main_compressor_cost, 'Bypass compressor cost = ', self.bypass_compressor_cost, 'Total turbomachinery cost = ', self.total_turbomachinery_cost)
        #print('HTR cost = ', htr_cost, 'LTR cost = ', ltr_cost, 'BT cost = ', bt_cost, 'Condenser cost = ', condenser_hx_cost, 'Heater cost = ', heater_hx_cost)
        #print('Total heat exchanger cost = ', hx_costs['total'], 'Total plant cost = ', self.total_plant_cost)
        #print('CO2 charge (kg) = ', self.co2_mass_kg, 'CO2 fluid cost (INR) = ', self.co2_fluid_cost)
        flue = (self.t_flue_out, self.p_flue_out, self.m_flue_in, self.flue_fluid)
        #print(turbine_exergy['work'], main_comp_exergy['work'], bypass_comp_exergy['work'])
        #print("Cooling tower flow = ", flowrate_cooler, "cooling_tower_cost", cooling_tower_cost)
        #print('Heater input = ', self.heat_input)
        self.exergy = dict(heater=heater_exergy, turbine=turbine_exergy, main_compressor=main_comp_exergy, bypass_compressor=bypass_comp_exergy, htr=htr_exergy, ltr=ltr_exergy, bt=bt_exergy, cooler=cooler_exergy, mixer=mixer_exergy, heater_in=heater_ex_in, heater_out=heater_ex_out, bypass=self.bypass)  # for the Sankey diagrams in app.py
        return (net_efficiency, exergy_efficiency, self.net_power, flue,
                self.turbine_cost, self.main_compressor_cost, self.bypass_compressor_cost, self.total_turbomachinery_cost,
                hx_costs, self.co2_fluid_cost, self.total_plant_cost)

        
    
