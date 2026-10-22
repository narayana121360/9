import CoolProp.CoolProp as CP
import plotly.graph_objects as go
import sys
import numpy as np
from scipy.optimize import root_scalar
from scipy.optimize import least_squares
from scipy.optimize import brentq
import scipy.optimize as opt
from scipy.optimize import toms748
from CoolProp.CoolProp import PhaseSI

# ---------------------------------------------------------------------------
# Purchased-equipment costing (2024 basis)
# ---------------------------------------------------------------------------
# Power-law ("scaling exponent") cost correlations of the form:
#     Cost = Base_Cost * (Capacity / Base_Capacity) ** Exponent
# The constants below reproduce the reference "PC (2024)" column exactly
# when Capacity = 1.00 in the equipment's stated unit:
#   Compressor : 280000.00 * (1/225.00)**0.46 = 23182.17  (m3/hr)
#   Turbine    : 1100000.00 * (1/225.00)**0.46 = 91072.81 (m3/hr)
#   Pump       : 8000.00 * (1/0.37)**0.35      = 11329.73 (kW)
#   Heat Exch. : 22500.00 * (1/1.00)**0.68     = 22500.00 (m2)
COST_TABLE = {
    'compressor':     {'base_capacity': 225.00, 'base_cost': 280000.00,  'exponent': 0.46, 'unit': 'm3/hr'},
    'turbine':        {'base_capacity': 225.00, 'base_cost': 1100000.00, 'exponent': 0.46, 'unit': 'm3/hr'},
    'pump':           {'base_capacity': 0.37,   'base_cost': 8000.00,    'exponent': 0.35, 'unit': 'kW'},
    'heat_exchanger': {'base_capacity': 1.00,   'base_cost': 22500.00,   'exponent': 0.68, 'unit': 'm2'},
    'cooling_tower':  {'base_capacity': 3.00,   'base_cost': 15000.00,    'exponent': 0.63, 'unit': 'm3/hr'},
}


def apply_cost_factors(base_cost, factors):
    """
    Applies purchased-equipment cost factors to a base correlation cost:
        Adjusted Cost = Base_Cost * Material_Factor * Temperature_Factor
                         * Pressure_Factor * (1 + Piping_Factor)
    `factors` is an optional dict that may supply any of:
        'material_factor', 'temperature_factor', 'pressure_factor', 'piping_factor'.
    Any factor left out of the dict defaults to no adjustment (1.0 for the three
    multiplicative factors, 0.0 for the piping factor, so (1 + 0) leaves the
    cost unchanged). Returns None (instead of raising) if base_cost is None or
    a factor value is invalid, so an uncostable component stays uncostable
    rather than crashing.
    """
    if base_cost is None:
        return None
    if not factors:
        return base_cost
    try:
        material_factor = factors.get('material_factor', 1.0)
        temperature_factor = factors.get('temperature_factor', 1.0)
        pressure_factor = factors.get('pressure_factor', 1.0)
        piping_factor = factors.get('piping_factor', 0.0)
        return base_cost * material_factor * temperature_factor * pressure_factor * (1.0 + piping_factor)
    except (TypeError, ValueError, AttributeError):
        return None


def equipment_cost(equipment_type, capacity, factors=None):
    """
    Purchased-equipment cost (2024 basis):
        Cost = Base_Cost * (Capacity / Base_Capacity) ** Exponent
    `capacity` must be expressed in COST_TABLE[equipment_type]['unit'].
    Returns None if capacity is missing/non-positive so a bad cost is never
    silently reported as zero or a NaN.
    `factors`, if provided, is a dict of purchased-equipment cost factors
    (material_factor, temperature_factor, pressure_factor, piping_factor)
    applied to the base correlation cost via apply_cost_factors().
    """
    params = COST_TABLE.get(equipment_type)
    if params is None or capacity is None:
        return None
    try:
        if capacity <= 0:
            return None
        base_cost = params['base_cost'] * (capacity / params['base_capacity']) ** params['exponent']
    except (ValueError, TypeError):
        return None
    return apply_cost_factors(base_cost, factors)


def _lmtd(t_hot_in, t_hot_out, t_cold_in, t_cold_out):
    """
    Counter-current log-mean temperature difference. Returns None if the
    terminal temperature differences are invalid (e.g. a temperature cross
    or a non-physical result), so callers can skip costing rather than
    raising or returning a misleading number.
    """
    try:
        dt1 = t_hot_in - t_cold_out
        dt2 = t_hot_out - t_cold_in
        if dt1 <= 0 or dt2 <= 0:
            return None
        if abs(dt1 - dt2) < 1e-6:
            return dt1
        return (dt1 - dt2) / np.log(dt1 / dt2)
    except (ValueError, TypeError, ZeroDivisionError):
        return None


def _normalize_phase(phase):
    if phase in ('gas', 'supercritical_gas', 'supercritical'):
        return 'gas'
    if phase == 'liquid':
        return 'liquid'
    return 'twophase'


# Assumed overall heat-transfer coefficients (W/m2-K) by duty type. None of
# the hx functions below compute a heat-transfer area directly, so area is
# back-calculated from duty and LMTD as Area = Q / (U * LMTD). U is chosen
# from this table based on the phase of each side (liquid-liquid runs much
# higher than gas-gas, phase-change duties higher still). These are
# screening-level values (typical Perry's / Coulson & Richardson ranges);
# tune them here if better plant data is available.
U_TABLE = {
    frozenset(['gas', 'gas']):             30.0,   # gas-gas, e.g. flue-gas economizer
    frozenset(['gas', 'liquid']):           60.0,   # gas-liquid, no phase change
    frozenset(['gas', 'twophase']):        100.0,   # gas heating/cooling a boiling or condensing fluid
    frozenset(['liquid', 'liquid']):       500.0,   # liquid-liquid, no phase change
    frozenset(['liquid', 'twophase']):    1000.0,   # liquid-side heating a boiling/condensing fluid
    frozenset(['twophase', 'twophase']):  1500.0,   # condenser/evaporator, both sides phase-changing
}


def _estimate_u(hot_phase, cold_phase):
    key = frozenset([_normalize_phase(hot_phase), _normalize_phase(cold_phase)])
    return U_TABLE.get(key, 100.0)


def _fluid_phase(fluid, t, p, default='liquid'):
    """Best-effort phase lookup; falls back to `default` if CoolProp can't
    resolve the phase (e.g. an incompressible fluid string, or T/P outside
    the fluid's valid range)."""
    try:
        return CP.PhaseSI('T', t, 'P', p, fluid)
    except Exception:
        return default


def hx_cost_from_duty(q_kw, t_hot_in, t_hot_out, t_cold_in, t_cold_out, hot_phase, cold_phase, f, factors=None):
    """
    Back-calculates the heat-transfer area required for a given duty using
    Area = Q / (U * LMTD), with U assumed from the hot/cold phase
    combination (see U_TABLE), then prices that area with the
    heat-exchanger cost correlation.
    `factors`, if provided, is passed through to equipment_cost() to apply
    material/temperature/pressure/piping cost factors (see apply_cost_factors).
    Returns (area_m2, u_assumed, cost_value); any entry may be None if the
    duty or terminal temperatures don't support a valid LMTD.
    """
    if q_kw is None or q_kw <= 0:
        return None, None, None
    q_watts = q_kw * 1000.0
    dtlm = _lmtd(t_hot_in, t_hot_out, t_cold_in, t_cold_out)
    if dtlm is None:
        return None, None, None
    u = _estimate_u(hot_phase, cold_phase)
    area = q_watts / (u * dtlm * f)
    cost_value = equipment_cost('heat_exchanger', area, factors)
    return area, u, cost_value


def generate_sankey_diagram(data_flows, node_labels):

    # Extract source, target, and value from the data_flows
    sources = [d['source'] for d in data_flows]
    targets = [d['target'] for d in data_flows]
    values = [d['value'] for d in data_flows]

    # Create the Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15, # Padding between nodes
            thickness=20, # Thickness of the nodes
            line=dict(color="black", width=0.5), # Node border
            label=node_labels, # Labels for the nodes
            # color="blue" # You can set a single color for all nodes or an array of colors
        ),
        link=dict(
            source=sources, # Indices of the source nodes
            target=targets, # Indices of the target nodes
            value=values, # Values (e.g., exergy amounts) of the links
            # color="lightgray" # You can set a single color for all links or an array of colors
        ))])

    fig.update_layout(
        title_text="Sankey Diagram for Exergy Analysis",
        font_size=10,
        height=600, # Set a fixed height for better visualization
        width=900 # Set a fixed width
    )

    return fig

def pure_exergy(chemical, p, t):
    phase = CP.PhaseSI("T", t, "P", p, chemical)
    if phase == 'gas':
        chemical_exergy_data = {
        "Water": 526.68,
        "CO2": 450.294258,
        "Nitrogen": 23.90463135,
        "Oxygen": 122.6475827
        }
    if phase == 'liquid':
        chemical_exergy_data = {
        "Water": 49.83,
        "CO2": 450.294258,
        }

def tmexergy(h, s, fluid_name, m_dot):
    t0 = 273.15 + 25
    p0 = 101325
    h0 = CP.PropsSI('H', "T", t0, "P", p0, fluid_name)
    s0 = CP.PropsSI('S', "T", t0, "P", p0, fluid_name)
    ex = m_dot * ((h-h0)-(t0 * (s - s0)))
    return(ex/1000)

def turbine(p_in, h_in, p_out, efficiency, turbine_fluid, m_dot, factors=None):
    s_in = CP.PropsSI('S', "H", h_in, "P", p_in, turbine_fluid)
    t_in = CP.PropsSI('T', "H", h_in, "P", p_in, turbine_fluid)
    h_isentropic = CP.PropsSI('H', "S", s_in, "P", p_out, turbine_fluid)
    isentrophic_specific_turbine_power = h_in - h_isentropic
    actual_specific_turbine_power = isentrophic_specific_turbine_power * efficiency
    h_out = h_in - actual_specific_turbine_power
    t_out = CP.PropsSI('T', "H", h_out, "P", p_out, turbine_fluid)
    actual_turbine_power = actual_specific_turbine_power * m_dot/1000
    s_out = CP.PropsSI('S', "H", h_out, "P", p_out, turbine_fluid)
    ex_in = tmexergy(h_in, s_in, turbine_fluid, m_dot)
    ex_out = tmexergy(h_out, s_out, turbine_fluid, m_dot)
    ex_des = ex_in - ex_out - actual_turbine_power
    ex_eff = actual_turbine_power/(ex_in - ex_out)
    exergy = {
        'in': ex_in,
        'out': ex_out,
        'work': actual_turbine_power,
        'destroyed': ex_des,
        'efficiency': ex_eff
    }
    # Purchased-equipment cost: sized on inlet (suction-side) volumetric flow (m3/hr)
    try:
        d_in = CP.PropsSI('D', "P", p_in, "H", h_in, turbine_fluid)
        capacity_m3hr = (m_dot / d_in) * 3600.0
    except Exception:
        capacity_m3hr = None
    cost = {
        'capacity_m3hr': capacity_m3hr,
        'cost': equipment_cost('turbine', capacity_m3hr, factors)
    }
    return(t_out, h_out, actual_turbine_power, exergy, cost)

def hx(hot_fluid, p_hot_in, h_hot_in, flowrate_hot, hot_drop, cold_fluid, p_cold_in, h_cold_in, flowrate_cold, cold_drop, hx_eff, factors=None):
    p_hot_out = p_hot_in - hot_drop
    p_cold_out = p_cold_in - cold_drop
    t_hot_in = CP.PropsSI('T', "H", h_hot_in, "P", p_hot_in, hot_fluid)
    t_cold_in = CP.PropsSI('T', "H", h_cold_in, "P", p_cold_in, cold_fluid)
    h_hot_c = CP.PropsSI('H', "T", t_cold_in, "P", p_hot_out, hot_fluid)
    h_cold_h = CP.PropsSI('H', "T", t_hot_in, "P", p_cold_out, cold_fluid)
    q_hot = flowrate_hot * (h_hot_in - h_hot_c)
    q_cold = flowrate_cold * (h_cold_h - h_cold_in)
    q_max = min(q_hot, q_cold)
    q = q_max * hx_eff
    q_kw = q/1000
    h_hot_out = h_hot_in - (q/flowrate_hot)
    h_cold_out = h_cold_in + (q/flowrate_cold)
    t_hot_out = CP.PropsSI('T', "H", h_hot_out, "P", p_hot_out, hot_fluid)
    t_cold_out = CP.PropsSI('T', "H", h_cold_out, "P", p_cold_out, cold_fluid)
    s_hot_in = CP.PropsSI('S', "P", p_hot_in, "H", h_hot_in, hot_fluid)
    s_hot_out = CP.PropsSI('S', "P", p_hot_out, "H", h_hot_out, hot_fluid)
    s_cold_in = CP.PropsSI('S', "P", p_cold_in, "H", h_cold_in, cold_fluid)
    s_cold_out = CP.PropsSI('S', "P", p_cold_out, "H", h_cold_out, cold_fluid)
    ex_hot_in = tmexergy(h_hot_in, s_hot_in, hot_fluid, flowrate_hot)
    ex_hot_out = tmexergy(h_hot_out, s_hot_out, hot_fluid, flowrate_hot)
    ex_cold_in = tmexergy(h_cold_in, s_cold_in, cold_fluid, flowrate_cold)
    ex_cold_out = tmexergy(h_cold_out, s_cold_out, cold_fluid, flowrate_cold)
    ex_hot_change = ex_hot_in - ex_hot_out
    ex_cold_change = ex_cold_out - ex_cold_in
    ex_des = ex_hot_change - ex_cold_change
    ex_eff = ex_cold_change/ex_hot_change
    exergy = {
        'hot_in': ex_hot_in,
        'hot_out': ex_hot_out,
        'cold_in': ex_cold_in,
        'cold_out': ex_cold_out,
        'hot_change': ex_hot_change,
        'cold_change': ex_cold_change,
        'destroyed': ex_des,
        'efficiency': ex_eff
    }
    # Purchased-equipment cost: area back-calculated from duty and LMTD.
    # All four terminal temperatures are already known above; phases are
    # determined per terminal from P and H (same approach as hx2), and a
    # side is treated as 'twophase' whenever its inlet and outlet phases differ.
    hot_phase_in = PhaseSI('P', p_hot_in, 'H', h_hot_in, hot_fluid)
    hot_phase_out = PhaseSI('P', p_hot_out, 'H', h_hot_out, hot_fluid)
    cold_phase_in = PhaseSI('P', p_cold_in, 'H', h_cold_in, cold_fluid)
    cold_phase_out = PhaseSI('P', p_cold_out, 'H', h_cold_out, cold_fluid)
    hot_phase = hot_phase_in if (hot_phase_in == hot_phase_out) else 'twophase'
    cold_phase = cold_phase_in if (cold_phase_in == cold_phase_out) else 'twophase'
    area, u_used, cost_value = hx_cost_from_duty(q_kw, t_hot_in, t_hot_out, t_cold_in, t_cold_out, hot_phase, cold_phase, factors)
    cost = {
        'area_m2': area,
        'u_assumed': u_used,
        'cost': cost_value
    }
    return(t_hot_out, p_hot_out, h_hot_out, t_cold_out, p_cold_out, h_cold_out, q_kw, exergy, cost)

def find_t(p, h_target, fluid):

    def root(t, h_target):
        CP.PropsSI('H', "T|gas", t, "P", p, fluid) - h_target

def hx_pinch_orc(hot_fluid, p_hot_in, t_hot_in, flowrate_hot, hot_drop, cold_fluid, p_cold_in, h_cold_in, flowrate_cold, cold_drop, pinch_point, f, n_steps=100, tolerance=0.1, factors=None):
    """
    Calculates HX performance by discretizing to find internal or external pinch points.
    tolerance: Tolerance for the bisection search in Watts.
    n_steps: Number of nodes to discretize the heat exchanger into.
    """
    #a = np.array([])
    #b = np.array([])
    #c = np.array([])
    #print(hot_fluid, p_hot_in, t_hot_in, flowrate_hot, hot_drop, cold_fluid, p_cold_in, h_cold_in, flowrate_cold, cold_drop, pinch_point)
    p_hot_out =p_hot_in - hot_drop
    p_cold_out = p_cold_in - cold_drop

    h_hot_in = CP.PropsSI('H', "T|gas", t_hot_in, "P", p_hot_in, hot_fluid)
    t_cold_in = CP.PropsSI('T', "H", h_cold_in, "P", p_cold_in, cold_fluid)
    #print('H', "T|gas", t_hot_in-273.15, "P", p_hot_in, hot_fluid, 'Hello2')
    # 1. Determine absolute maximum theoretical Q (assuming infinite area, Delta T = 0 at ends)
    #print(('H', "T", t_cold_in-273.15, "P", p_hot_out/100_000, hot_fluid))
    #h_hot_out_abs_min = CP.PropsSI('H', "T", t_cold_in, "P", p_hot_out, hot_fluid)
    h_cold_out_abs_max = CP.PropsSI('H', "T", t_hot_in, "P", p_cold_out, cold_fluid)

    #q_max_hot = flowrate_hot * (h_hot_in - h_hot_out_abs_min)
    q_max_cold = flowrate_cold * (h_cold_out_abs_max - h_cold_in)
    #print('h2 = ', h_cold_out_abs_max, 'h1 = ', h_cold_in)

    t_flue_cond = CP.PropsSI('T', "Q|gas", 1, "P", p_hot_out, hot_fluid)
    h_flue_cond = CP.PropsSI('H', "T|gas", t_flue_cond, "P", p_hot_out, hot_fluid)
    q_max_cond = flowrate_hot * (h_hot_in - h_flue_cond)

    q_high = min(q_max_cold, q_max_cond)
    #print(q_max_cold, q_max_cond, 'Hello', 't_flue_cond = ', t_flue_cond-273.15, 'h_flue_cond = ', h_flue_cond, 'h_hot_in = ', h_hot_in)


    def get_pinch(q):

        start_q = 1
        if start_q == q:
            start_q = q/100
        intervals = 100
        step = (q - start_q) / intervals
        min_dt_found = float('inf')
        for i in range(intervals + 1):
            q_frac = i / n_steps
            h_cold_node = h_cold_in + ((q_frac * q) / flowrate_cold)
            p_cold_node = p_cold_in - (q_frac * cold_drop)
            #print('Struck inside for loop in get_pinch')
            #print('h_cold_node =', h_cold_node, "q_frac= ", h_cold_node, 'q = ', q)
            t_cold_node = CP.PropsSI('T', "H", h_cold_node, "P", p_cold_node, cold_fluid)


            h_hot_node = h_hot_in - (((1.0 -q_frac) * q) / flowrate_hot)
            #print('h_hot_in = ',h_hot_in,  'h_hot_node = ', h_hot_node)
            p_hot_node = p_hot_out + ( q_frac * hot_drop)
            t_hot_node = find_t_from_h(h_hot_node, p_hot_node, t_flue_cond-1, 702+273.15, hot_fluid)

            #a = np.array([])
            #b = np.array([])
            #c = np.array([])
            #a = np.append(a, t_cold_node-273.15)
            #b = np.append(b, t_hot_node-273.15)
            dt_node = t_hot_node - t_cold_node
            #c = np.append(c, dt_node)
            if dt_node < min_dt_found:
                min_dt_found = dt_node
        #print(a, b, c)
        return min_dt_found

    def objective_function(q):
        #print('pinch = ', get_pinch(q), 'q =', q)
        return get_pinch(q) - pinch_point

    lower_bound = 100
    upper_bound = q_high
    #initial_guess = (lower_bound + upper_bound)/2
    #result = least_squares(objective_function, x0=initial_guess, bounds=(lower_bound, upper_bound))
    #found_q = result.x[0]
    #print('f(a) = ', get_pinch(100), 'f(b) = ', get_pinch(q_high))
    #print("lower_bound = ", lower_bound, 'upper_bound = ', upper_bound)
    found_q = toms748(objective_function, a=lower_bound, b=upper_bound, xtol= 100, rtol= 0.01)
    #found_q = opt.brentq(objective_function, lower_bound, upper_bound)



    # 3. Final Calculations using the converged q_actual


    h_hot_out = h_hot_in - (found_q / flowrate_hot)
    h_cold_out = h_cold_in + (found_q / flowrate_cold)

    t_hot_out = find_t_from_h(h_hot_out, p_hot_out, t_flue_cond-1, 702+273.15, hot_fluid)
    t_cold_out = CP.PropsSI('T', "H", h_cold_out, "P", p_cold_out, cold_fluid)

    # 4. Exergy Calculations (Requires your tmexergy function)
    s_hot_in = CP.PropsSI('S', "P|gas", p_hot_in, "T", t_hot_in, hot_fluid)
    s_hot_out = CP.PropsSI('S', "P|gas", p_hot_out, "T", t_hot_out, hot_fluid)
    s_cold_in = CP.PropsSI('S', "P", p_cold_in, "H", h_cold_in, cold_fluid)
    s_cold_out = CP.PropsSI('S', "P", p_cold_out, "H", h_cold_out, cold_fluid)

    ex_hot_in = tmexergy(h_hot_in, s_hot_in, hot_fluid, flowrate_hot)
    ex_hot_out = tmexergy(h_hot_out, s_hot_out, hot_fluid, flowrate_hot)
    ex_cold_in = tmexergy(h_cold_in, s_cold_in, cold_fluid, flowrate_cold)
    ex_cold_out = tmexergy(h_cold_out, s_cold_out, cold_fluid, flowrate_cold)

    ex_hot_change = ex_hot_in - ex_hot_out
    ex_cold_change = ex_cold_out - ex_cold_in
    ex_des = ex_hot_change - ex_cold_change
    ex_eff = (ex_cold_change / ex_hot_change) if ex_hot_change != 0 else 0

    exergy = {
        'hot_in': ex_hot_in,
        'hot_out': ex_hot_out,
        'cold_in': ex_cold_in,
        'cold_out': ex_cold_out,
        'hot_change': ex_hot_change,
        'cold_change': ex_cold_change,
        'destroyed': ex_des,
        'efficiency': ex_eff
    }
    # Purchased-equipment cost: area back-calculated from duty and LMTD.
    # Hot side is explicitly forced to gas-phase properties above (flue-gas-type
    # stream), so its phase is hardcoded rather than probed with PhaseSI (avoids
    # failures near partial condensation). Cold side phase is determined per
    # terminal from P and H, same approach as hx2.
    q_kw = found_q / 1000.0
    hot_phase = 'gas'
    cold_phase_in = PhaseSI('P', p_cold_in, 'H', h_cold_in, cold_fluid)
    cold_phase_out = PhaseSI('P', p_cold_out, 'H', h_cold_out, cold_fluid)
    cold_phase = cold_phase_in if (cold_phase_in == cold_phase_out) else 'twophase'
    area, u_used, cost_value = hx_cost_from_duty(q_kw, t_hot_in, t_hot_out, t_cold_in, t_cold_out, hot_phase, cold_phase, f, factors)
    cost = {
        'area_m2': area,
        'u_assumed': u_used,
        'cost': cost_value
    }

    return (t_hot_out, p_hot_out, h_hot_out, t_cold_out, p_cold_out, h_cold_out, q_kw, exergy, cost)


def hx_pinch(hot_fluid, p_hot_in, h_hot_in, flowrate_hot, hot_drop, cold_fluid, p_cold_in, h_cold_in, flowrate_cold, cold_drop, pinch_point, n_steps=100, tolerance=0.1, factors=None):
    """
    Calculates HX performance by discretizing to find internal or external pinch points.
    tolerance: Tolerance for the bisection search in Watts.
    n_steps: Number of nodes to discretize the heat exchanger into.
    """
    p_hot_out =p_hot_in - hot_drop
    p_cold_out = p_cold_in - cold_drop

    t_hot_in = CP.PropsSI('T', "H", h_hot_in, "P", p_hot_in, hot_fluid)
    t_cold_in = CP.PropsSI('T', "H", h_cold_in, "P", p_cold_in, cold_fluid)

    # 1. Determine absolute maximum theoretical Q (assuming infinite area, Delta T = 0 at ends)
    h_hot_out_abs_min = CP.PropsSI('H', "T", t_cold_in, "P", p_hot_out, hot_fluid)
    h_cold_out_abs_max = CP.PropsSI('H', "T", t_hot_in, "P", p_cold_out, cold_fluid)

    q_max_hot = flowrate_hot * (h_hot_in - h_hot_out_abs_min)
    q_max_cold = flowrate_cold * (h_cold_out_abs_max - h_cold_in)
    q_high = min(q_max_hot, q_max_cold)

    # 2. Bisection Search for actual Q
    q_low = 0.0
    q_actual = 0.0

    while (q_high - q_low) > tolerance:
        q_guess = (q_high + q_low) / 2.0
        min_dt_found = float('inf')

        # Discretize along the heat transferred (from cold inlet / hot outlet to cold outlet / hot inlet)
        for i in range(n_steps + 1):
            q_frac = i / n_steps  # Fraction of total heat transferred at this node

            # Cold stream properties at this node (flows from q_frac = 0 to 1)
            h_cold_node = h_cold_in + (q_frac * q_guess) / flowrate_cold
            p_cold_node = p_cold_in - (q_frac * cold_drop)

            # Hot stream properties at this node (flows from q_frac = 1 to 0)
            h_hot_node = h_hot_in - ((1.0 - q_frac) * q_guess) / flowrate_hot
            p_hot_node = p_hot_out + (q_frac * hot_drop)

            # Calculate temperatures at this node
            try:
                t_cold_node = CP.PropsSI('T', "H", h_cold_node, "P", p_cold_node, cold_fluid)
                t_hot_node = CP.PropsSI('T', "H", h_hot_node, "P", p_hot_node, hot_fluid)

                dt_node = t_hot_node - t_cold_node
                if dt_node < min_dt_found:
                    min_dt_found = dt_node
            except ValueError:
                # If CoolProp fails (e.g., properties out of bounds), treat it as an invalid Q
                min_dt_found = -1
                break

        # Adjust bisection bounds
        if min_dt_found < pinch_point:
            q_high = q_guess  # Q is too high, it violated the pinch point
        else:
            q_low = q_guess   # Q is safe, try to extract more heat
            q_actual = q_guess

    # 3. Final Calculations using the converged q_actual
    q_kw = q_actual / 1000.0

    h_hot_out = h_hot_in - (q_actual / flowrate_hot)
    h_cold_out = h_cold_in + (q_actual / flowrate_cold)

    t_hot_out = CP.PropsSI('T', "H", h_hot_out, "P", p_hot_out, hot_fluid)
    t_cold_out = CP.PropsSI('T', "H", h_cold_out, "P", p_cold_out, cold_fluid)

    # 4. Exergy Calculations (Requires your tmexergy function)
    s_hot_in = CP.PropsSI('S', "P", p_hot_in, "H", h_hot_in, hot_fluid)
    s_hot_out = CP.PropsSI('S', "P", p_hot_out, "H", h_hot_out, hot_fluid)
    s_cold_in = CP.PropsSI('S', "P", p_cold_in, "H", h_cold_in, cold_fluid)
    s_cold_out = CP.PropsSI('S', "P", p_cold_out, "H", h_cold_out, cold_fluid)

    ex_hot_in = tmexergy(h_hot_in, s_hot_in, hot_fluid, flowrate_hot)
    ex_hot_out = tmexergy(h_hot_out, s_hot_out, hot_fluid, flowrate_hot)
    ex_cold_in = tmexergy(h_cold_in, s_cold_in, cold_fluid, flowrate_cold)
    ex_cold_out = tmexergy(h_cold_out, s_cold_out, cold_fluid, flowrate_cold)

    ex_hot_change = ex_hot_in - ex_hot_out
    ex_cold_change = ex_cold_out - ex_cold_in
    ex_des = ex_hot_change - ex_cold_change
    ex_eff = (ex_cold_change / ex_hot_change) if ex_hot_change != 0 else 0

    exergy = {
        'hot_in': ex_hot_in,
        'hot_out': ex_hot_out,
        'cold_in': ex_cold_in,
        'cold_out': ex_cold_out,
        'hot_change': ex_hot_change,
        'cold_change': ex_cold_change,
        'destroyed': ex_des,
        'efficiency': ex_eff
    }
    # Purchased-equipment cost: area back-calculated from duty and LMTD.
    # All four terminal temperatures are already known above; phases are
    # determined per terminal from P and H (same approach as hx2), and a
    # side is treated as 'twophase' whenever its inlet and outlet phases differ.
    hot_phase_in = PhaseSI('P', p_hot_in, 'H', h_hot_in, hot_fluid)
    hot_phase_out = PhaseSI('P', p_hot_out, 'H', h_hot_out, hot_fluid)
    cold_phase_in = PhaseSI('P', p_cold_in, 'H', h_cold_in, cold_fluid)
    cold_phase_out = PhaseSI('P', p_cold_out, 'H', h_cold_out, cold_fluid)
    hot_phase = hot_phase_in if (hot_phase_in == hot_phase_out) else 'twophase'
    cold_phase = cold_phase_in if (cold_phase_in == cold_phase_out) else 'twophase'
    area, u_used, cost_value = hx_cost_from_duty(q_kw, t_hot_in, t_hot_out, t_cold_in, t_cold_out, hot_phase, cold_phase, factors)
    cost = {
        'area_m2': area,
        'u_assumed': u_used,
        'cost': cost_value
    }

    return (t_hot_out, p_hot_out, h_hot_out, t_cold_out, p_cold_out, h_cold_out, q_kw, exergy, cost)


# Both sides of hot side are known
def hx2(hot_fluid, p_hot_in, h_hot_in, h_hot_out, flowrate_hot, hot_drop, cold_fluid, p_cold_in, h_cold_in, cold_drop, approach, f, factors=None):
    p_hot_out = p_hot_in - hot_drop
    p_cold_out = p_cold_in - cold_drop
    t_hot_in = CP.PropsSI('T', "H", h_hot_in, "P", p_hot_in, hot_fluid)
    t_cold_in = CP.PropsSI('T', "H", h_cold_in, "P", p_cold_in, cold_fluid)
    q_kw = flowrate_hot * (h_hot_in - h_hot_out)/1000
    # Required Cooling Water flowrate
    t_cold_out = t_cold_in + approach
    h_cold_out = CP.PropsSI('H', "T", t_cold_out, "P", p_cold_out, cold_fluid)
    flowrate_cold = q_kw*1000 / (h_cold_out - h_cold_in)
    s_hot_in = CP.PropsSI('S', "P", p_hot_in, "H", h_hot_in, hot_fluid)
    s_hot_out = CP.PropsSI('S', "P", p_hot_out, "H", h_hot_out, hot_fluid)
    s_cold_in = CP.PropsSI('S', "P", p_cold_in, "H", h_cold_in, cold_fluid)
    s_cold_out = CP.PropsSI('S', "P", p_cold_out, "H", h_cold_out, cold_fluid)
    ex_hot_in = tmexergy(h_hot_in, s_hot_in, hot_fluid, flowrate_hot)
    ex_hot_out = tmexergy(h_hot_out, s_hot_out, hot_fluid, flowrate_hot)
    ex_cold_in = tmexergy(h_cold_in, s_cold_in, cold_fluid, flowrate_cold)
    ex_cold_out = tmexergy(h_cold_out, s_cold_out, cold_fluid, flowrate_cold)
    ex_hot_change = ex_hot_in - ex_hot_out
    ex_cold_change = ex_cold_out - ex_cold_in
    ex_des = ex_hot_change - ex_cold_change
    ex_eff = ex_cold_change/ex_hot_change
    exergy = {
        'hot_in': ex_hot_in,
        'hot_out': ex_hot_out,
        'cold_in': ex_cold_in,
        'cold_out': ex_cold_out,
        'hot_change': ex_hot_change,
        'cold_change': ex_cold_change,
        'destroyed': ex_des,
        'efficiency': ex_eff
    }
    # Purchased-equipment cost: area back-calculated from duty and LMTD
    t_hot_out = CP.PropsSI('T', "H", h_hot_out, "P", p_hot_out, hot_fluid)
    hot_phase_in = PhaseSI('P', p_hot_in, 'H', h_hot_in, hot_fluid)
    hot_phase_out = PhaseSI('P', p_hot_out, 'H', h_hot_out, hot_fluid)
    cold_phase_in = PhaseSI('P', p_cold_in, 'H', h_cold_in, cold_fluid)
    cold_phase_out = PhaseSI('P', p_cold_out, 'H', h_cold_out, cold_fluid)
    hot_phase = hot_phase_in if (hot_phase_in == hot_phase_out) else 'twophase'
    cold_phase = cold_phase_in if (cold_phase_in == cold_phase_out) else 'twophase'
    area, u_used, cost_value = hx_cost_from_duty(q_kw, t_hot_in, t_hot_out, t_cold_in, t_cold_out, hot_phase, cold_phase, f, factors)
    cost = {
        'area_m2': area,
        'u_assumed': u_used,
        'cost': cost_value
    }
    return(t_cold_out, p_cold_out, h_cold_out, q_kw, flowrate_cold, exergy, cost)

# Both sides of cold side are known. And approach2 is between hot out and cold in temperature, not complete, gives error
def hx3(hot_fluid, p_hot_in, h_hot_in, flowrate_hot, hot_drop, cold_fluid, p_cold_in, h_cold_in, h_cold_out, flowrate_cold, cold_drop, approach2, factors=None):
    p_hot_out = p_hot_in - hot_drop
    p_cold_out = p_cold_in - cold_drop
    t_hot_in = CP.PropsSI('T', "H", h_hot_in, "P", p_hot_in, hot_fluid)
    t_cold_in = CP.PropsSI('T', "H", h_cold_in, "P", p_cold_in, cold_fluid)
    t_cold_out = CP.PropsSI('T', "H", h_cold_out, "P", p_cold_out, cold_fluid)
    q_kw = flowrate_cold * (h_cold_out - h_cold_in)/1000
    # Required hot fluid flowrate
    t_hot_out = t_cold_in + approach2
    h_hot_out = CP.PropsSI('H', "T", t_hot_out, "P", p_hot_out, hot_fluid)
    flowrate_hot_ideal = q_kw*1000 / (h_hot_in - h_hot_out_ideal)
    if flowrate_hot < flowrate_hot_ideal:
        sys.exit("The flowrate of hot fluid in the heater is not sufficient. Increase the flowrate of hot fluid in the heater")
    s_hot_in = CP.PropsSI('S', "P", p_hot_in, "H", h_hot_in, hot_fluid)
    s_hot_out = CP.PropsSI('S', "P", p_hot_out, "H", h_hot_out, hot_fluid)
    s_cold_in = CP.PropsSI('S', "P", p_cold_in, "H", h_cold_in, cold_fluid)
    s_cold_out = CP.PropsSI('S', "P", p_cold_out, "H", h_cold_out, cold_fluid)
    ex_hot_in = tmexergy(h_hot_in, s_hot_in, hot_fluid, flowrate_hot)
    ex_hot_out = tmexergy(h_hot_out, s_hot_out, hot_fluid, flowrate_hot)
    ex_cold_in = tmexergy(h_cold_in, s_cold_in, cold_fluid, flowrate_cold)
    ex_cold_out = tmexergy(h_cold_out, s_cold_out, cold_fluid, flowrate_cold)
    ex_hot_change = ex_hot_in - ex_hot_out
    ex_cold_change = ex_cold_out - ex_cold_in
    ex_des = ex_hot_change - ex_cold_change
    ex_eff = ex_cold_change/ex_hot_change
    exergy = {
        'hot_in': ex_hot_in,
        'hot_out': ex_hot_out,
        'cold_in': ex_cold_in,
        'cold_out': ex_cold_out,
        'hot_change': ex_hot_change,
        'cold_change': ex_cold_change,
        'destroyed': ex_des,
        'efficiency': ex_eff
    }
    # Purchased-equipment cost: area back-calculated from duty and LMTD.
    # All four terminal temperatures are already known above; phases are
    # determined per terminal from P and H (same approach as hx2), and a
    # side is treated as 'twophase' whenever its inlet and outlet phases differ.
    hot_phase_in = PhaseSI('P', p_hot_in, 'H', h_hot_in, hot_fluid)
    hot_phase_out = PhaseSI('P', p_hot_out, 'H', h_hot_out, hot_fluid)
    cold_phase_in = PhaseSI('P', p_cold_in, 'H', h_cold_in, cold_fluid)
    cold_phase_out = PhaseSI('P', p_cold_out, 'H', h_cold_out, cold_fluid)
    hot_phase = hot_phase_in if (hot_phase_in == hot_phase_out) else 'twophase'
    cold_phase = cold_phase_in if (cold_phase_in == cold_phase_out) else 'twophase'
    area, u_used, cost_value = hx_cost_from_duty(q_kw, t_hot_in, t_hot_out, t_cold_in, t_cold_out, hot_phase, cold_phase, factors)
    cost = {
        'area_m2': area,
        'u_assumed': u_used,
        'cost': cost_value
    }
    return(t_hot_out, p_hot_out, h_hot_out, p_cold_out, q_kw, exergy, cost)


def hx3_flue(hot_fluid, t_hot_in, p_hot_in, flowrate_hot, hot_drop, cold_fluid, p_cold_in, h_cold_in, h_cold_out, flowrate_cold, cold_drop, approach2, factors=None):
    p_hot_out = p_hot_in - hot_drop
    p_cold_out = p_cold_in - cold_drop
    h_hot_in = CP.PropsSI('H', "T|supercritical_gas", t_hot_in, "P", p_hot_in, hot_fluid)
    t_cold_in = CP.PropsSI('T', "H", h_cold_in, "P", p_cold_in, cold_fluid)
    t_cold_out = CP.PropsSI('T', "H", h_cold_out, "P", p_cold_out, cold_fluid)
    t_hot_out = t_cold_in + approach2
    h_hot_out = CP.PropsSI('H', "T|supercritical_gas", t_hot_out, "P", p_hot_out, hot_fluid)
    q_kw = flowrate_hot * (h_hot_in - h_hot_out)/1000

    # Required hot fluid flowrate



    s_hot_in = CP.PropsSI('S', "P|supercritical_gas", p_hot_in, "T", t_hot_in, hot_fluid)
    s_hot_out = CP.PropsSI('S', "P|supercritical_gas", p_hot_out, "T", t_hot_out, hot_fluid)
    s_cold_in = CP.PropsSI('S', "P", p_cold_in, "H", h_cold_in, cold_fluid)
    s_cold_out = CP.PropsSI('S', "P", p_cold_out, "H", h_cold_out, cold_fluid)
    ex_hot_in = tmexergy(h_hot_in, s_hot_in, hot_fluid, flowrate_hot)
    ex_hot_out = tmexergy(h_hot_out, s_hot_out, hot_fluid, flowrate_hot)
    ex_cold_in = tmexergy(h_cold_in, s_cold_in, cold_fluid, flowrate_cold)
    ex_cold_out = tmexergy(h_cold_out, s_cold_out, cold_fluid, flowrate_cold)
    ex_hot_change = ex_hot_in - ex_hot_out
    ex_cold_change = ex_cold_out - ex_cold_in
    ex_des = ex_hot_change - ex_cold_change
    ex_eff = ex_cold_change/ex_hot_change
    exergy = {
        'hot_in': ex_hot_in,
        'hot_out': ex_hot_out,
        'cold_in': ex_cold_in,
        'cold_out': ex_cold_out,
        'hot_change': ex_hot_change,
        'cold_change': ex_cold_change,
        'destroyed': ex_des,
        'efficiency': ex_eff
    }
    # Purchased-equipment cost: area back-calculated from duty and LMTD.
    # Hot side is explicitly forced to (supercritical) gas-phase properties above,
    # so its phase is hardcoded rather than probed with PhaseSI (avoids failures
    # near partial condensation). Cold side phase is determined per terminal
    # from P and H, same approach as hx2.
    hot_phase = 'gas'
    cold_phase_in = PhaseSI('P', p_cold_in, 'H', h_cold_in, cold_fluid)
    cold_phase_out = PhaseSI('P', p_cold_out, 'H', h_cold_out, cold_fluid)
    cold_phase = cold_phase_in if (cold_phase_in == cold_phase_out) else 'twophase'
    area, u_used, cost_value = hx_cost_from_duty(q_kw, t_hot_in, t_hot_out, t_cold_in, t_cold_out, hot_phase, cold_phase, factors)
    cost = {
        'area_m2': area,
        'u_assumed': u_used,
        'cost': cost_value
    }
    return(t_hot_out, p_hot_out, h_hot_out, p_cold_out, q_kw, exergy, cost)

# h_hot_c is not measurable. Hence, suitably modified
def hx10(hot_fluid, p_hot_in, h_hot_in, flowrate_hot, hot_drop, cold_fluid, p_cold_in, h_cold_in, flowrate_cold, cold_drop, hx_eff, f, factors=None):
    p_hot_out = p_hot_in - hot_drop
    p_cold_out = p_cold_in - cold_drop
    t_hot_in = CP.PropsSI('T', "P", p_hot_in, "H", 0.9999*h_hot_in, hot_fluid)
    t_cold_in = CP.PropsSI('T', "P", p_cold_in, "H", h_cold_in, cold_fluid)
    h_hot_c = CP.PropsSI('H', "T", t_cold_in, "P", p_hot_out, hot_fluid)
    h_cold_h = h_hot_c
    q_hot = flowrate_hot * (h_hot_in - h_hot_c)
    q_cold =flowrate_hot * (h_hot_in - h_hot_c)
    q_max = min(q_hot, q_cold)
    q = q_max * hx_eff
    q_kw = q/1000
    h_hot_out = h_hot_in - (q/flowrate_hot)
    h_cold_out = h_cold_in + (q/flowrate_cold)
    t_hot_out = CP.PropsSI('T', "H", h_hot_out, "P", p_hot_out, hot_fluid)
    t_cold_out = CP.PropsSI('T', "H", h_cold_out, "P", p_cold_out, cold_fluid)
    s_hot_in = CP.PropsSI('S', "P", p_hot_in, "H", 0.999999*h_hot_in, hot_fluid)
    s_hot_out = CP.PropsSI('S', "P", p_hot_out, "H", h_hot_out, hot_fluid)
    s_cold_in = CP.PropsSI('S', "P", p_cold_in, "H", h_cold_in, cold_fluid)
    s_cold_out = CP.PropsSI('S', "P", p_cold_out, "H", 0.999999*h_cold_out, cold_fluid)
    ex_hot_in = tmexergy(h_hot_in, s_hot_in, hot_fluid, flowrate_hot)
    ex_hot_out = tmexergy(h_hot_out, s_hot_out, hot_fluid, flowrate_hot)
    ex_cold_in = tmexergy(h_cold_in, s_cold_in, cold_fluid, flowrate_cold)
    ex_cold_out = tmexergy(h_cold_out, s_cold_out, cold_fluid, flowrate_cold)
    ex_hot_change = ex_hot_in - ex_hot_out
    ex_cold_change = ex_cold_out - ex_cold_in
    ex_des = ex_hot_change - ex_cold_change
    ex_eff = ex_cold_change/ex_hot_change
    exergy = {
        'hot_in': ex_hot_in,
        'hot_out': ex_hot_out,
        'cold_in': ex_cold_in,
        'cold_out': ex_cold_out,
        'hot_change': ex_hot_change,
        'cold_change': ex_cold_change,
        'destroyed': ex_des,
        'efficiency': ex_eff
    }
    # Purchased-equipment cost: area back-calculated from duty and LMTD.
    # All four terminal temperatures are already known above; phases are
    # determined per terminal from P and H (same approach as hx2), and a
    # side is treated as 'twophase' whenever its inlet and outlet phases differ.
    hot_phase_in = PhaseSI('P', p_hot_in, 'H', h_hot_in, hot_fluid)
    hot_phase_out = PhaseSI('P', p_hot_out, 'H', h_hot_out, hot_fluid)
    cold_phase_in = PhaseSI('P', p_cold_in, 'H', h_cold_in, cold_fluid)
    cold_phase_out = PhaseSI('P', p_cold_out, 'H', h_cold_out, cold_fluid)
    hot_phase = hot_phase_in if (hot_phase_in == hot_phase_out) else 'twophase'
    cold_phase = cold_phase_in if (cold_phase_in == cold_phase_out) else 'twophase'
    area, u_used, cost_value = hx_cost_from_duty(q_kw, t_hot_in, t_hot_out, t_cold_in, t_cold_out, hot_phase, cold_phase, f, factors)
    cost = {
        'area_m2': area,
        'u_assumed': u_used,
        'cost': cost_value
    }
    return(t_hot_out, p_hot_out, h_hot_out, t_cold_out, p_cold_out, h_cold_out, q_kw, exergy, cost)


def compressor(p_in, h_in, p_out, eff, compressor_fluid, m_dot, factors=None):
    s_in = CP.PropsSI('S', "P", p_in, "H", h_in, compressor_fluid)
    t_in = CP.PropsSI('T', "P", p_in, "H", h_in, compressor_fluid)
    h_isentropic = CP.PropsSI('H', "S", s_in, "P", p_out, compressor_fluid)
    isentrophic_specific_compressor_power = h_isentropic - h_in
    actual_specific_compressor_power = isentrophic_specific_compressor_power / eff
    h_out = h_in + actual_specific_compressor_power
    t_out = CP.PropsSI('T', "H", h_out, "P", p_out, compressor_fluid)
    actual_compressor_power = actual_specific_compressor_power * m_dot/1000
    s_out = CP.PropsSI('S', "H", h_out, "P", p_out, compressor_fluid)
    ex_in = tmexergy(h_in, s_in, compressor_fluid, m_dot)
    ex_out = tmexergy(h_out, s_out, compressor_fluid, m_dot)
    ex_des = actual_compressor_power - (ex_out - ex_in)
    ex_eff = (ex_out - ex_in)/actual_compressor_power
    exergy = {
        'in': ex_in,
        'out': ex_out,
        'work': actual_compressor_power,
        'destroyed': ex_des,
        'efficiency': ex_eff
    }
    # Purchased-equipment cost: sized on inlet (suction-side) volumetric flow (m3/hr)
    try:
        d_in = CP.PropsSI('D', "P", p_in, "H", h_in, compressor_fluid)
        capacity_m3hr = (m_dot / d_in) * 3600.0
    except Exception:
        capacity_m3hr = None
    cost = {
        'capacity_m3hr': capacity_m3hr,
        'cost': equipment_cost('compressor', capacity_m3hr, factors)
    }
    return(t_out, h_out, actual_compressor_power, exergy, cost)

def pump(p_in, h_in, p_out, eff, pump_fluid, m_dot, factors=None):
    s_in = CP.PropsSI('S', "H", h_in, "P", p_in, pump_fluid)
    t_in = CP.PropsSI('T', "H", h_in, "P", p_in, pump_fluid)
    d_in = CP.PropsSI('D', "S", s_in, "P", p_in, pump_fluid)
    del_h_ideal = (p_out - p_in)/d_in
    h_out = h_in + (del_h_ideal/eff)
    actual_specific_pump_power = h_out - h_in
    t_out = CP.PropsSI('T', "H", h_out, "P", p_out, pump_fluid)
    actual_pump_power = actual_specific_pump_power * m_dot/1000
    s_out = CP.PropsSI('S', "H", h_out, "P", p_out, pump_fluid)
    ex_in = tmexergy(h_in, s_in, pump_fluid, m_dot)
    ex_out = tmexergy(h_out, s_out, pump_fluid, m_dot)
    ex_des = actual_pump_power - (ex_out - ex_in)
    ex_eff = (ex_out - ex_in)/actual_pump_power
    exergy = {
        'in': ex_in,
        'out': ex_out,
        'work': actual_pump_power,
        'destroyed': ex_des,
        'efficiency': ex_eff
    }
    # Purchased-equipment cost: sized on actual pump shaft power (kW)
    cost = {
        'capacity_kw': actual_pump_power,
        'cost': equipment_cost('pump', actual_pump_power, factors)
    }
    return(t_out, h_out, actual_pump_power, exergy, cost)

def ref_prop(h, s):
    h10 = CP.PropsSI('H', "T", 0.7042776300716315+273.15, "Q", 0, "INCOMP::LiBr[0]")
    s10 = CP.PropsSI('S', "T", 0.7042776300716315+273.15, "Q", 0, "INCOMP::LiBr[0]")
    h010 = CP.PropsSI('H', "T", 0.7042776300716315+273.15, "P", 676, "Water")
    s010 = CP.PropsSI('S', "T", 0.7042776300716315+273.15, "P", 676, "Water")
    del_h = h-h010
    del_s = s-s010
    h2 = h10+del_h
    s2 = s10+del_s
    return(h2, s2)

def mixer(massflow1, p1, h1, massflow2, p2, h2, fluid):
    total_massflow = massflow1 + massflow2
    s1 = CP.PropsSI('S', "P", p1, "H", h1, fluid)
    s2 = CP.PropsSI('S', "P", p2, "H", h2, fluid)
    p = (p1 + p2)/2
    h = ((massflow1 * h1) + (massflow2 * h2))/total_massflow
    ex1 = tmexergy(h1, s1, fluid, massflow1)
    ex2 = tmexergy(h2, s2, fluid, massflow2)
    s = CP.PropsSI('S', "P", p, "H", h, fluid)
    t = CP.PropsSI('T', "P", p, "H", h, fluid)
    ex = tmexergy(h, s, fluid, total_massflow)
    ex_des = (ex1 + ex2) - ex
    exergy = {
        'ex1': ex1,
        'ex2': ex2,
        'ex': ex,
        'destroyed': ex_des
    }
    return(p, h, t, total_massflow, exergy)

def hx_flue(hot_fluid, p_hot_in, t_hot_in, flowrate_hot, hot_drop, cold_fluid, p_cold_in, h_cold_in, flowrate_cold, cold_drop, pinch_point, n_steps=100, tolerance=1000, factors=None):
    """
    Calculates HX performance by discretizing to find internal or external pinch points.
    tolerance: Tolerance for the bisection search in Watts.
    n_steps: Number of nodes to discretize the heat exchanger into.

    """



    p_hot_out = p_hot_in - hot_drop
    p_cold_out = p_cold_in - cold_drop

    h_hot_in = CP.PropsSI('H', "T|gas", t_hot_in, "P", p_hot_in, hot_fluid)
    t_cold_in = CP.PropsSI('T', "H", h_cold_in, "P", p_cold_in, cold_fluid)

    # 1. Determine absolute maximum theoretical Q (assuming infinite area, Delta T = 0 at ends)
    h_hot_out_abs_min = CP.PropsSI('H', "T", t_cold_in, "P", p_hot_out, hot_fluid)
    h_cold_out_abs_max = CP.PropsSI('H', "T", t_hot_in, "P", p_cold_out, cold_fluid)

    q_max_hot = flowrate_hot * (h_hot_in - h_hot_out_abs_min)
    q_max_cold = flowrate_cold * (h_cold_out_abs_max - h_cold_in)
    t_flue_cond = CP.PropsSI('T', "Q|gas", 1, "P", p_hot_out, hot_fluid)
    h_flue_cond = CP.PropsSI('H', "T|gas", t_flue_cond, "P", p_hot_out, hot_fluid)
    q_max_cond = flowrate_hot * (h_hot_in - h_flue_cond)
    q_high = min(q_max_hot, q_max_cold, q_max_cond)
    cp_hot = CP.PropsSI('C', "T|gas", t_hot_in, "P", p_hot_in, hot_fluid)
    #print("q =", q_max_hot, q_max_cold)

    # Whole New Start

    def get_pinch(q):

        start_q = 1
        if start_q == q:
            start_q = q/100
        intervals = 100
        step = (q - start_q) / intervals
        min_dt_found = float('inf')
        for i in range(intervals + 1):
            q_frac = i / n_steps
            h_cold_node = h_cold_in + ((q_frac * q) / flowrate_cold)
            p_cold_node = p_cold_in - (q_frac * cold_drop)
            #print('h_cold_node =', h_cold_node, "q_frac= ", h_cold_node, 'q = ', q)
            t_cold_node = CP.PropsSI('T', "H", h_cold_node, "P", p_cold_node, cold_fluid)
            t_hot_node = (t_hot_in - ((1.0 -q_frac) * q /(flowrate_hot * cp_hot)))
            #a = np.append(a, t_cold_node-273.15)
            #b = np.append(b, t_hot_node-273.15)
            dt_node = t_hot_node - t_cold_node
            #c = np.append(c, dt_node)
            if dt_node < min_dt_found:
                min_dt_found = dt_node
        return min_dt_found

    def objective_function(q):
        return get_pinch(q) - pinch_point

    lower_bound = 100
    upper_bound = q_high
    initial_guess = (lower_bound + upper_bound)/2
    result = least_squares(objective_function, x0=initial_guess, bounds=(lower_bound, upper_bound))
    found_q = result.x[0]
    #achieved_pinch = get_pinch(found_q)
    #print('achieved_pinch = ', achieved_pinch)






    # 3. Final Calculations using the converged q_actual
    q_kw = found_q


    h_hot_out = h_hot_in - (q_kw / flowrate_hot)
    h_cold_out = h_cold_in + (q_kw / flowrate_cold)

    state_cold_out = CP.PhaseSI('P', p_cold_out, 'H', h_cold_out, cold_fluid)
    if (state_cold_out == 'liquid' or state_cold_out == 'twophase'):
        c = 0
    else:
        c = 1
        #q_kw = found_q * 0.01
        #h_hot_out = h_hot_in - (q_kw / flowrate_hot)
        #h_cold_out = h_cold_in + (q_kw / flowrate_cold)


    t_hot_out = t_hot_in - (q_kw /(flowrate_hot * cp_hot))
    t_cold_out = CP.PropsSI('T', "H", h_cold_out, "P", p_cold_out, cold_fluid)
    #q = CP.PropsSI('S', "P|gas", p_hot_out, "T", t_hot_out, hot_fluid)
    #print(t_hot_out-273.15, t_cold_out-273.15)

    # 4. Exergy Calculations (Requires your tmexergy function)
    s_hot_in = CP.PropsSI('S', "P|gas", p_hot_in, "T", t_hot_in, hot_fluid)
    s_hot_out = CP.PropsSI('S', "P|gas", p_hot_out, "T", t_hot_out, hot_fluid)
    s_cold_in = CP.PropsSI('S', "P", p_cold_in, "H", h_cold_in, cold_fluid)
    s_cold_out = CP.PropsSI('S', "P", p_cold_out, "H", h_cold_out, cold_fluid)

    ex_hot_in = tmexergy(h_hot_in, s_hot_in, hot_fluid, flowrate_hot)
    ex_hot_out = tmexergy(h_hot_out, s_hot_out, hot_fluid, flowrate_hot)
    ex_cold_in = tmexergy(h_cold_in, s_cold_in, cold_fluid, flowrate_cold)
    ex_cold_out = tmexergy(h_cold_out, s_cold_out, cold_fluid, flowrate_cold)

    ex_hot_change = ex_hot_in - ex_hot_out
    ex_cold_change = ex_cold_out - ex_cold_in
    ex_des = ex_hot_change - ex_cold_change
    ex_eff = (ex_cold_change / ex_hot_change) if ex_hot_change != 0 else 0





    exergy = {
        'hot_in': ex_hot_in,
        'hot_out': ex_hot_out,
        'cold_in': ex_cold_in,
        'cold_out': ex_cold_out,
        'hot_change': ex_hot_change,
        'cold_change': ex_cold_change,
        'destroyed': ex_des,
        'efficiency': ex_eff
    }
    a = None
    b = None
    #c = None
    q_kw = found_q/1000.0
    #print("q_kw = ", q_kw, 't_hot_out = ', t_hot_out, 'p_hot_out = ', p_hot_out, 't_cold_out = ', t_cold_out, 'p_cold_out = ', p_cold_out, 'p_cold_in = ', p_cold_in, 'm = ', flowrate_cold)

    # Purchased-equipment cost: area back-calculated from duty and LMTD.
    # Hot side is explicitly forced to gas-phase properties above (flue-gas-type
    # stream), so its phase is hardcoded rather than probed with PhaseSI (avoids
    # failures near partial condensation). Cold side phase is determined per
    # terminal from P and H, same approach as hx2.
    hot_phase = 'gas'
    cold_phase_in = PhaseSI('P', p_cold_in, 'H', h_cold_in, cold_fluid)
    cold_phase_out = PhaseSI('P', p_cold_out, 'H', h_cold_out, cold_fluid)
    cold_phase = cold_phase_in if (cold_phase_in == cold_phase_out) else 'twophase'
    area, u_used, cost_value = hx_cost_from_duty(q_kw, t_hot_in, t_hot_out, t_cold_in, t_cold_out, hot_phase, cold_phase, factors)
    cost = {
        'area_m2': area,
        'u_assumed': u_used,
        'cost': cost_value
    }

    return (t_hot_out, p_hot_out, h_hot_out, t_cold_out, p_cold_out, h_cold_out, q_kw, exergy, a, b, c, cost)




def find_t_from_h(h_target, p, t_min, t_max, fluid):
    def objective_function(t):
        return  (CP.PropsSI('H', "T|gas", t, "P", p, fluid) - h_target)

    try:
        # brentq finds the root of the objective_function within [t_min, t_max]
        t_solution = brentq(objective_function, t_min, t_max)
        return t_solution

    except ValueError as e:
        # brentq raises a ValueError if the target h is not bounded by the outputs
        # at t_min and t_max (i.e., the function doesn't cross 0 in this range).
        print(f"Error: The target h={h_target} cannot be reached within t in ({t_min}, {t_max}).")
        # You could also calculate the h bounds to show the user why it failed
        h_at_min = calculate_h_from_t(t_min)
        h_at_max = calculate_h_from_t(t_max)
        print(f"Given range produces h values from approximately {h_at_min} to {h_at_max}.")
        return None

def hx_pinch_steam(hot_fluid, p_hot_in, t_hot_in, flowrate_hot, hot_drop, cold_fluid, p_cold_in, h_cold_in, flowrate_cold, cold_drop, pinch_point, f, n_steps=100, tolerance=0.1, factors=None):
    """
    Calculates HX performance by discretizing to find internal or external pinch points.
    tolerance: Tolerance for the bisection search in Watts.
    n_steps: Number of nodes to discretize the heat exchanger into.
    """

    #a = np.array([])
    #b = np.array([])
    #c = np.array([])

    p_hot_out =p_hot_in - hot_drop
    p_cold_out = p_cold_in - cold_drop

    h_hot_in = CP.PropsSI('H', "T|gas", t_hot_in, "P", p_hot_in, hot_fluid)
    t_cold_in = CP.PropsSI('T', "H", h_cold_in, "P", p_cold_in, cold_fluid)

    # 1. Determine absolute maximum theoretical Q (assuming infinite area, Delta T = 0 at ends)
    h_hot_out_abs_min = CP.PropsSI('H', "T", t_cold_in, "P", p_hot_out, hot_fluid)
    h_cold_out_abs_max = CP.PropsSI('H', "T", t_hot_in, "P", p_cold_out, cold_fluid)

    q_max_hot = flowrate_hot * (h_hot_in - h_hot_out_abs_min)
    q_max_cold = flowrate_cold * (h_cold_out_abs_max - h_cold_in)

    t_flue_cond = CP.PropsSI('T', "Q|gas", 1, "P", p_hot_out, hot_fluid)
    h_flue_cond = CP.PropsSI('H', "T|gas", t_flue_cond, "P", p_hot_out, hot_fluid)
    q_max_cond = flowrate_hot * (h_hot_in - h_flue_cond)

    q_high = min(q_max_hot, q_max_cold, q_max_cond)
    #print(q_max_hot, q_max_cond, q_max_cond)


    def get_pinch(q):

        start_q = 1
        if start_q == q:
            start_q = q/100
        intervals = 100
        step = (q - start_q) / intervals
        min_dt_found = float('inf')
        for i in range(intervals + 1):
            q_frac = i / n_steps
            h_cold_node = h_cold_in + ((q_frac * q) / flowrate_cold)
            p_cold_node = p_cold_in - (q_frac * cold_drop)
            #print('h_cold_node =', h_cold_node, "q_frac= ", h_cold_node, 'q = ', q)
            t_cold_node = CP.PropsSI('T', "H", h_cold_node, "P", p_cold_node, cold_fluid)


            h_hot_node = h_hot_in - (((1.0 -q_frac) * q) / flowrate_hot)
            #print('h_hot_in = ',h_hot_in,  'h_hot_node = ', h_hot_node)
            p_hot_node = p_hot_out + ( q_frac * hot_drop)
            t_hot_node = find_t_from_h(h_hot_node, p_hot_node, t_flue_cond-2, 702+273.15, hot_fluid)

            #a = np.array([])
            #b = np.array([])
            #c = np.array([])
            #a = np.append(a, t_cold_node-273.15)
            #b = np.append(b, t_hot_node-273.15)
            dt_node = t_hot_node - t_cold_node
            #c = np.append(c, dt_node)
            if dt_node < min_dt_found:
                min_dt_found = dt_node
        #print(a, b, c)
        return min_dt_found

    def objective_function(q):
        #print('pinch = ', get_pinch(q), 'q =', q)
        return get_pinch(q) - pinch_point

    lower_bound = 100
    upper_bound = q_high
    #initial_guess = (lower_bound + upper_bound)/2
    #result = least_squares(objective_function, x0=initial_guess, bounds=(lower_bound, upper_bound))
    #found_q = result.x[0]
    #print('Hello 1')
    try:
        found_q = opt.brentq(objective_function, lower_bound, upper_bound, xtol= 100, rtol= 0.01)
    except ValueError:
        found_q = 0



    # 3. Final Calculations using the converged q_actual


    h_hot_out = h_hot_in - (found_q / flowrate_hot)
    h_cold_out = h_cold_in + (found_q / flowrate_cold)

    t_hot_out = find_t_from_h(h_hot_out, p_hot_out, t_flue_cond-2, 702+273.15, hot_fluid)
    t_cold_out = CP.PropsSI('T', "H", h_cold_out, "P", p_cold_out, cold_fluid)

    # 4. Exergy Calculations (Requires your tmexergy function)
    s_hot_in = CP.PropsSI('S', "P|gas", p_hot_in, "T", t_hot_in, hot_fluid)
    s_hot_out = CP.PropsSI('S', "P|gas", p_hot_out, "T", t_hot_out, hot_fluid)
    s_cold_in = CP.PropsSI('S', "P", p_cold_in, "H", h_cold_in, cold_fluid)
    s_cold_out = CP.PropsSI('S', "P", p_cold_out, "H", h_cold_out, cold_fluid)

    ex_hot_in = tmexergy(h_hot_in, s_hot_in, hot_fluid, flowrate_hot)
    ex_hot_out = tmexergy(h_hot_out, s_hot_out, hot_fluid, flowrate_hot)
    ex_cold_in = tmexergy(h_cold_in, s_cold_in, cold_fluid, flowrate_cold)
    ex_cold_out = tmexergy(h_cold_out, s_cold_out, cold_fluid, flowrate_cold)

    ex_hot_change = ex_hot_in - ex_hot_out
    ex_cold_change = ex_cold_out - ex_cold_in
    ex_des = ex_hot_change - ex_cold_change
    ex_eff = (ex_cold_change / ex_hot_change) if ex_hot_change != 0 else 0

    exergy = {
        'hot_in': ex_hot_in,
        'hot_out': ex_hot_out,
        'cold_in': ex_cold_in,
        'cold_out': ex_cold_out,
        'hot_change': ex_hot_change,
        'cold_change': ex_cold_change,
        'destroyed': ex_des,
        'efficiency': ex_eff
    }

    state_cold_out = CP.PhaseSI('P', p_cold_out, 'H', h_cold_out, cold_fluid)
    if (state_cold_out == 'liquid' or state_cold_out == 'twophase'):
        c = 0
    else:
        c = 1

    # Purchased-equipment cost: area back-calculated from duty and LMTD.
    # Hot side is explicitly forced to gas-phase properties above (flue-gas-type
    # stream), so its phase is hardcoded rather than probed with PhaseSI (avoids
    # failures near partial condensation). Cold side phase is determined per
    # terminal from P and H, same approach as hx2.
    q_kw = found_q / 1000.0
    hot_phase = 'gas'
    cold_phase_in = PhaseSI('P', p_cold_in, 'H', h_cold_in, cold_fluid)
    cold_phase_out = PhaseSI('P', p_cold_out, 'H', h_cold_out, cold_fluid)
    cold_phase = cold_phase_in if (cold_phase_in == cold_phase_out) else 'twophase'
    area, u_used, cost_value = hx_cost_from_duty(q_kw, t_hot_in, t_hot_out, t_cold_in, t_cold_out, hot_phase, cold_phase, f, factors)
    cost = {
        'area_m2': area,
        'u_assumed': u_used,
        'cost': cost_value
    }

    return (t_hot_out, p_hot_out, h_hot_out, t_cold_out, p_cold_out, h_cold_out, q_kw, exergy, c, cost)


def sat_pressure(x, fluid):
            return CP.PropsSI('P', "T", x, "Q", 0, fluid)

        # 2. Define the objective function (f(x) - target = 0)
def find_t_from_p(x, p, fluid):
            return sat_pressure(x, fluid)-p

def hx_pinch_libr(hot_fluid, p_hot_in, t_hot_in, flowrate_hot, hot_drop, cold_fluid, p_cold_in, h_cold_in, flowrate_cold, cold_drop, pinch_point, n_steps=100, tolerance=0.1, factors=None):
    """
    Calculates HX performance by discretizing to find internal or external pinch points.
    tolerance: Tolerance for the bisection search in Watts.
    n_steps: Number of nodes to discretize the heat exchanger into.
    """

    #a = np.array([])
    #b = np.array([])
    #c = np.array([])

    p_hot_out =p_hot_in - hot_drop
    p_cold_out = p_cold_in - cold_drop

    h_hot_in = CP.PropsSI('H', "T|gas", t_hot_in, "P", p_hot_in, hot_fluid)
    t_cold_in = CP.PropsSI('T', "H", h_cold_in, "P", p_cold_in, cold_fluid)

    # 1. Determine absolute maximum theoretical Q (assuming infinite area, Delta T = 0 at ends)
    h_hot_out_abs_min = CP.PropsSI('H', "T|gas", t_cold_in, "P", p_hot_out, hot_fluid)

    sol_t_cold_sat = root_scalar(find_t_from_p, args=(p_cold_out, cold_fluid), bracket=[273, 400], method='brentq')
    t_cold_sat = sol_t_cold_sat.root
    h_cold_sat = CP.PropsSI('H', "Q", 0, "T", t_cold_sat, cold_fluid)

    q_max_hot = flowrate_hot * (h_hot_in - h_hot_out_abs_min)
    q_max_cold = flowrate_cold * (h_cold_sat - h_cold_in)

    t_flue_cond = CP.PropsSI('T', "Q|gas", 1, "P", p_hot_out, hot_fluid)
    h_flue_cond = CP.PropsSI('H', "T|gas", t_flue_cond, "P", p_hot_out, hot_fluid)
    q_max_cond = flowrate_hot * (h_hot_in - h_flue_cond)

    q_high = min(q_max_hot, q_max_cold, q_max_cond)
    #print(q_max_hot, q_max_cold, q_max_cond)



    def get_pinch(q):

        start_q = 1
        if start_q == q:
            start_q = q/100
        intervals = 100
        step = (q - start_q) / intervals
        min_dt_found = float('inf')
        for i in range(intervals + 1):
            q_frac = i / n_steps
            h_cold_node = h_cold_in + ((q_frac * q) / flowrate_cold)
            p_cold_node = p_cold_in - (q_frac * cold_drop)
            #print('h_cold_node =', h_cold_node, "q_frac= ", h_cold_node, 'q = ', q)
            t_cold_node = CP.PropsSI('T', "H", h_cold_node, "P", p_cold_node, cold_fluid)


            h_hot_node = h_hot_in - (((1.0 -q_frac) * q) / flowrate_hot)
            #print('h_hot_in = ',h_hot_in,  'h_hot_node = ', h_hot_node)
            p_hot_node = p_hot_out + ( q_frac * hot_drop)
            t_hot_node = find_t_from_h(h_hot_node, p_hot_node, t_flue_cond-1, 702+273.15, hot_fluid)

            #a = np.array([])
            #b = np.array([])
            #c = np.array([])
            #a = np.append(a, t_cold_node-273.15)
            #b = np.append(b, t_hot_node-273.15)
            dt_node = t_hot_node - t_cold_node
            #c = np.append(c, dt_node)
            if dt_node < min_dt_found:
                min_dt_found = dt_node
        #print(a, b, c)
        return min_dt_found

    def objective_function(q):
        #print('pinch = ', get_pinch(q), 'q =', q)
        return get_pinch(q) - pinch_point

    lower_bound = 100
    upper_bound = q_high
    #initial_guess = (lower_bound + upper_bound)/2
    #result = least_squares(objective_function, x0=initial_guess, bounds=(lower_bound, upper_bound))
    #found_q = result.x[0]
    #print('Hello 1')
    try:
        found_q = opt.brentq(objective_function, lower_bound, upper_bound, xtol= 100, rtol= 0.01)
    except ValueError:
        found_q = 0



    # 3. Final Calculations using the converged q_actual


    h_hot_out = h_hot_in - (found_q / flowrate_hot)
    h_cold_out = h_cold_in + (found_q / flowrate_cold)

    t_hot_out = find_t_from_h(h_hot_out, p_hot_out, t_flue_cond-1, 702+273.15, hot_fluid)
    t_cold_out = CP.PropsSI('T', "H", h_cold_out, "P", p_cold_out, cold_fluid)

    # 4. Exergy Calculations (Requires your tmexergy function)
    s_hot_in = CP.PropsSI('S', "P|gas", p_hot_in, "T", t_hot_in, hot_fluid)
    s_hot_out = CP.PropsSI('S', "P|gas", p_hot_out, "T", t_hot_out, hot_fluid)
    s_cold_in = CP.PropsSI('S', "P", p_cold_in, "H", h_cold_in, cold_fluid)
    s_cold_out = CP.PropsSI('S', "P", p_cold_out, "H", h_cold_out, cold_fluid)

    ex_hot_in = tmexergy(h_hot_in, s_hot_in, hot_fluid, flowrate_hot)
    ex_hot_out = tmexergy(h_hot_out, s_hot_out, hot_fluid, flowrate_hot)
    ex_cold_in = tmexergy(h_cold_in, s_cold_in, cold_fluid, flowrate_cold)
    ex_cold_out = tmexergy(h_cold_out, s_cold_out, cold_fluid, flowrate_cold)

    ex_hot_change = ex_hot_in - ex_hot_out
    ex_cold_change = ex_cold_out - ex_cold_in
    ex_des = ex_hot_change - ex_cold_change
    ex_eff = (ex_cold_change / ex_hot_change) if ex_hot_change != 0 else 0

    exergy = {
        'hot_in': ex_hot_in,
        'hot_out': ex_hot_out,
        'cold_in': ex_cold_in,
        'cold_out': ex_cold_out,
        'hot_change': ex_hot_change,
        'cold_change': ex_cold_change,
        'destroyed': ex_des,
        'efficiency': ex_eff
    }

    state_cold_out = CP.PhaseSI('P', p_cold_out, 'H', h_cold_out, cold_fluid)
    if (state_cold_out == 'liquid' or state_cold_out == 'twophase'):
        c = 0
    else:
        c = 1

    # Purchased-equipment cost: area back-calculated from duty and LMTD.
    # Hot side is explicitly forced to gas-phase properties above (flue-gas-type
    # stream), so its phase is hardcoded rather than probed with PhaseSI (avoids
    # failures near partial condensation). Cold side phase is determined per
    # terminal from P and H, same approach as hx2.
    q_kw = found_q / 1000.0
    hot_phase = 'gas'
    cold_phase_in = PhaseSI('P', p_cold_in, 'H', h_cold_in, cold_fluid)
    cold_phase_out = PhaseSI('P', p_cold_out, 'H', h_cold_out, cold_fluid)
    cold_phase = cold_phase_in if (cold_phase_in == cold_phase_out) else 'twophase'
    area, u_used, cost_value = hx_cost_from_duty(q_kw, t_hot_in, t_hot_out, t_cold_in, t_cold_out, hot_phase, cold_phase, factors)
    cost = {
        'area_m2': area,
        'u_assumed': u_used,
        'cost': cost_value
    }

    return (t_hot_out, p_hot_out, h_hot_out, t_cold_out, p_cold_out, h_cold_out, q_kw, exergy, c, cost)
