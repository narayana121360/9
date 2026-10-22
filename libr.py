#!/usr/bin/env python
# coding: utf-8

# In[1]:


import CoolProp.CoolProp as CP
import plotly.graph_objects as go
import sys
from scipy.optimize import root_scalar
from uo import generate_sankey_diagram, pure_exergy, tmexergy, turbine, hx, hx_pinch, hx2, hx3, hx10, compressor, pump, ref_prop, mixer, hx3_flue, hx_flue, find_t_from_h, ref_prop, find_t_from_p, equipment_cost, _lmtd


def _safe_cost(cost_dict):
    """
    Pulls the numeric 'cost' out of a component cost dict returned by uo.py,
    treating a missing dict or a None cost (e.g. invalid capacity/LMTD) as 0
    so a single un-costable component doesn't blow up the plant total.
    """
    if not cost_dict:
        return 0.0
    value = cost_dict.get('cost')
    return value if value is not None else 0.0


def _escalate(cost_value, factor):
    """
    Applies a cost-index escalation factor to a single cost value, passing
    None through unchanged -- so an un-costable component (invalid
    capacity/LMTD) doesn't turn into a spurious number. Same helper as in
    super.py/trans.py/steam.py/orc.py.
    """
    return None if cost_value is None else cost_value * factor


def _hx_cost_given_u(q_kw, temps, u, f, factors):
    """
    Prices a heat exchanger whose overall heat-transfer coefficient (U) is
    supplied directly (rather than estimated from stream phases, as in
    uo.py's hx_cost_from_duty). Area = Q / (U * LMTD).
    `temps` is a dict with keys t_hot_in, t_hot_out, t_cold_in, t_cold_out.
    Returns (area_m2, cost_value); either may be None if the duty or the
    terminal temperatures don't support a valid LMTD.
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


class libr:
    def __init__(self, t_flue_in, p_flue_in, m_flue, flue_string, t_eva, t_cond, solution_hx_eff, pump_eff, pressure_loss, circulation_ratio):
        self.t_flue_in = t_flue_in
        self.p_flue_in = p_flue_in
        self.m_flue = m_flue
        self.flue_string = flue_string
        self.t_flue_out = None
        self.p_flue_out = None
        self.h_flue_out = None
        self.t_eva = t_eva
        self.t_cond = t_cond
        self.solution_hx_eff = solution_hx_eff
        self.pump_eff = pump_eff
        self.p_drop_solution_hx_hot = pressure_loss[0]
        self.p_drop_solution_hx_cold = pressure_loss[1]
        self.p_drop_boiler_hx_hot = pressure_loss[2]
        self.p_drop_boiler_hx_cold = pressure_loss[3]
        self.circulation_ratio = circulation_ratio
        self.cop = None
        self.q_cooling = None
        self.q_heating = None
        self.flowrate_water = None
        self.cooling_tower_cost = None
        # LiBr working-fluid (solution charge) cost: 8 kg of LiBr per ton of
        # refrigeration (1 TR = 3.517 kW of cooling), priced at 600 INR/kg.
        self.libr_charge_kg_per_tr = 8.0
        self.kw_per_tr = 3.517
        self.libr_price_inr_per_kg = 600.0
        self.libr_mass_kg = None
        self.libr_fluid_cost = None

    def sat_pressure(x, fluid):
            return CP.PropsSI('P', "T", x, "Q", 0, fluid)

        # 2. Define the objective function (f(x) - target = 0)
    def find_t_from_p(x, p, fluid):
            return sat_pressure(x, fluid)-p



    def calc(self):
        m_weak = 0.05
        Weak = 0.5648
        weak = f"INCOMP::LiBr[{Weak:.4f}]"
        pinch_point = 10
        f_solution_hx = 1

        # Purchased-equipment cost factors (material, temperature, pressure,
        # piping), applied by uo.py's equipment_cost() as:
        #   Adjusted Cost = Base_Cost * material_factor * temperature_factor
        #                    * pressure_factor * (1 + piping_factor)
        # Set to no-ops here (1, 1, 1, 0) so the reported cost matches the
        # unfactored base correlation cost. Only pump() and hx10() (the
        # solution heat exchanger) take a factors argument here; the
        # desorber/evaporator/absorber/condenser go through the local
        # _hx_cost_given_u() helper instead, which isn't factor-aware.
        factors_solution_hx = {
            'material_factor': 2.1,
            'temperature_factor': 1,
            'pressure_factor': 1.3,
            'piping_factor': 0.46
        }
        factors_pump = {
            'material_factor': 3.4,
            'temperature_factor': 1,
            'pressure_factor': 2,
            'piping_factor': 0.3
        }

        m_ref = m_weak/self.circulation_ratio
        m_strong = m_weak - m_ref
        Strong = m_weak*Weak/m_strong
        strong = f"INCOMP::LiBr[{Strong:.4f}]"
        p10 = CP.PropsSI('P', "T", self.t_eva, "Q", 0, 'IF97::Water')
        p8 = CP.PropsSI('P', "T", self.t_cond, "Q", 0, 'IF97::Water')
        p1 = p10
        p2 = p8 + self.p_drop_solution_hx_cold
        p3 = p2 - self.p_drop_solution_hx_cold
        p4 = p3
        p5 = p4 - self.p_drop_solution_hx_hot
        p6 = p1
        p7 = p3
        p9 = p10
        #print(p1, p2, p3, p4, p5, p6,p7, p8, p9, p10)

        t10 = self.t_eva
        #t7 = self.t_cond

        # Absorber
        sol_t1 = root_scalar(find_t_from_p, args=(p1, weak), bracket=[274, 400], method='brentq')
        t1 = sol_t1.root
        # Desorber
        t_liquid_desorber = root_scalar(find_t_from_p, args=(p2, strong), bracket=[274, 400], method='brentq')
        t4 = t_liquid_desorber.root
        t_vapor_desorber = root_scalar(find_t_from_p, args=(p2, weak), bracket=[274, 400], method='brentq')
        t7 = t_vapor_desorber.root

        # Evaporator
        sol_t10 = root_scalar(find_t_from_p, args=(p1, 'IF97::Water'), bracket=[274, 400], method='brentq')
        t10 = sol_t10.root

        h1 = CP.PropsSI('H', "T", t1, "Q", 0, weak)
        h4 = CP.PropsSI('H', "T", t4, "Q", 0, strong)

        # Pump
        t2, h2, actual_pump_power, pump_exergy, pump_cost = pump(p1, (0.9999999*h1), p2, self.pump_eff, weak, m_weak, factors_pump)

        # Solution Heat Exchanger
        #print(m_weak)
        #print('HX = ', strong, p4, h4, m_strong, self.p_drop_solution_hx_hot, weak, p2, h2, m_weak, self.p_drop_solution_hx_cold, self.solution_hx_eff)
        t5, p5, h5, t3, p3, h3, q_sol_ex, exergy_solution_hx, solution_hx_cost = hx10(strong, p4, h4, m_strong, self.p_drop_solution_hx_hot, weak, p2, h2, m_weak, self.p_drop_solution_hx_cold, self.solution_hx_eff, f_solution_hx, factors_solution_hx)

        # Heater
        #hx_pinch_libr(self.flue_string, self.p_flue_in, self.t_flue_in, self.m_flue, self.p_drop_boiler_hx_hot, weak, p3, h3, m_weak, self.p_drop_boiler_hx_cold, pinch_point)
        flue_approach = 10
        #print(self.t_flue_in-273.15, self.p_flue_in/100_000, self.flue_string)
        h_flue_in = CP.PropsSI('H','T|gas',self.t_flue_in,'P',self.p_flue_in, self.flue_string)
        #print(h_flue_in)
        s_flue_in = CP.PropsSI('S','T|gas',self.t_flue_in,'P',self.p_flue_in, self.flue_string)
        #print(s_flue_in)
        self.t_flue_out = t4 + flue_approach
        self.p_flue_out = self.p_flue_in - self.p_drop_boiler_hx_hot
        self.h_flue_out = CP.PropsSI('H','T|gas',self.t_flue_out,'P',self.p_flue_out, self.flue_string)
        s_flue_out = CP.PropsSI('S','T|gas',self.t_flue_out,'P',self.p_flue_out, self.flue_string)
        del_q = self.m_flue * (h_flue_in - self.h_flue_out)

        # Need to find state 7 properties
        h07 = CP.PropsSI('H', "P", p7, "T", t7, 'IF97::Water')
        #print('h07 = ', h07)
        h007 = CP.PropsSI('H', "P", p7, "T", t7, 'IF97::Water')
        #print('h007 = ', h007)
        s07 = CP.PropsSI('S', "P", p7, "T", t7, 'IF97::Water')
        h7, s7 = ref_prop(h07 , s07)
        #print(h7)

        import sympy as sp

        # 1. Define the symbol 'm'
        m = sp.symbols('m')

        # 2. Define the equation
        # In SymPy, equations are set up as sp.Eq(left_side, right_side)
        equation = sp.Eq(del_q, ((m-(m/self.circulation_ratio))*h4)+((m/self.circulation_ratio)*h7)-(m*h3))

        # 3. Solve the equation for x
        m = sp.solve(equation, m)
        m_weak=m[0]

        #print(f"The exact solutions are: {m}")
        # Output: The exact solutions are: [2, 3]
        type(m)
        #print(m[0])
        #print(type(m))
        #print(del_q, m_weak, circulation_ratio, h4, h7, h3)


        # Repeat Start
        Weak = 0.5648
        weak = f"INCOMP::LiBr[{Weak:.4f}]"

        m_ref = m_weak/self.circulation_ratio
        m_strong = m_weak - m_ref
        Strong = m_weak*Weak/m_strong
        strong = f"INCOMP::LiBr[{Strong:.4f}]"
        p10 = CP.PropsSI('P', "T", self.t_eva, "Q", 0, 'IF97::Water')
        p8 = CP.PropsSI('P', "T", self.t_cond, "Q", 0, 'IF97::Water')
        p1 = p10
        p2 = p8 + self.p_drop_solution_hx_cold
        p3 = p2 - self.p_drop_solution_hx_cold
        p4 = p3
        p5 = p4 - self.p_drop_solution_hx_hot
        p6 = p1
        p7 = p3
        p9 = p10
        #print(p1, p2, p3, p4, p5, p6,p7, p8, p9, p10)

        t10 = self.t_eva
        t7 = self.t_cond

        # Absorber
        sol_t1 = root_scalar(find_t_from_p, args=(p1, weak), bracket=[274, 400], method='brentq')
        t1 = sol_t1.root
        # Desorber
        t_liquid_desorber = root_scalar(find_t_from_p, args=(p2, strong), bracket=[274, 400], method='brentq')
        t4 = t_liquid_desorber.root
        t_vapor_desorber = root_scalar(find_t_from_p, args=(p2, weak), bracket=[274, 400], method='brentq')
        t7 = t_vapor_desorber.root
        t6_sol = root_scalar(find_t_from_p, args=(p1, strong), bracket=[274, 400], method='brentq')
        t6 = t6_sol.root

        # Evaporator
        sol_t10 = root_scalar(find_t_from_p, args=(p1, 'IF97::Water'), bracket=[274, 400], method='brentq')
        t10 = sol_t10.root

        h1 = CP.PropsSI('H', "T", t1, "Q", 0, weak)
        h4 = CP.PropsSI('H', "T", t4, "Q", 0, strong)

        # Pump
        t2, h2, actual_pump_power, pump_exergy, pump_cost = pump(p1, (0.9999999*h1), p2, self.pump_eff, weak, m_weak, factors_pump)

        # Solution Heat Exchanger
        t5, p5, h5, t3, p3, h3, q_sol_ex, exergy_solution_hx, solution_hx_cost = hx10(strong, p4, h4, m_strong, self.p_drop_solution_hx_hot, weak, p2, h2, m_weak, self.p_drop_solution_hx_cold, self.solution_hx_eff, f_solution_hx, factors_solution_hx)

        # Repeat End

        #state 8
        p8 = p7
        h6 = h5
        t8 = CP.PropsSI('T', "P", p8, "Q", 0, 'IF97::Water')
        h08 = CP.PropsSI('H', "P", p8, "Q", 0, 'IF97::Water')
        s08 = CP.PropsSI('S', "P", p8, "Q", 0, 'IF97::Water')
        h8, s8 = ref_prop(h08 , s08)

        q_cond = m_ref * (h7-h8)
        #print(q_cond)
        #print(p8, t8-273.15, h8, s8)
        #print(h7, h8)

        #state 9

        p9 =p1
        h09 = h08
        t9 = CP.PropsSI('T', "P", p9, "H", h09, 'IF97::Water')
        s09 = CP.PropsSI('S', "P", p9, "H", h09, 'IF97::Water')
        q9 = CP.PropsSI('Q', "P", p9, "H", h09, 'IF97::Water')
        h9, s9 = ref_prop(h09 , s09)
        #print(p9, t9-273.15, h9, s9, q9)

        #state 10

        p10 = p9
        t10 = t9
        h010 = CP.PropsSI('H', "P", p10, "Q", 1, 'IF97::Water')
        s010 = CP.PropsSI('S', "P", p10, "Q", 1, 'IF97::Water')
        h10, s10 = ref_prop(h010 , s010)
        q_evap = m_ref * (h010-h09)
        q_absor = (m_ref * h10) + (m_strong * h6) - (m_weak * h1)
        #print(q_evap)
        #print(m_ref, h9, h10)
        #print(t10, p10)
        #print(h010, s010)

        self.cop = q_evap/del_q
        #print(h_flue_in, s_flue_in, self.flue_string, self.m_flue)

        ex_flue_in = tmexergy(h_flue_in, s_flue_in, self.flue_string, self.m_flue)
        ex_flue_out = tmexergy(self.h_flue_out, s_flue_out, self.flue_string, self.m_flue)
        ex_in = ex_flue_in - ex_flue_out


        ex9 = tmexergy(h09, s09, 'IF97::Water', m_ref)
        ex10 = tmexergy(h010, s010,  'IF97::Water', m_ref)
        ex_out = ex9 - ex10

        ex_eff = 100 * ex_out/ex_in
        #print(ex_eff)
        self.q_cooling = q_evap/1000
        self.q_heating = del_q/1000


        #print(self.cop)
        flue = (self.t_flue_out, self.p_flue_out, self.h_flue_out, self.m_flue, self.flue_string)



        hx_desorber = del_q/1000
        hx_evaporator = q_evap/1000
        hx_absorber = ((m_ref * h10) + (m_strong * h6) - (m_weak * h1))/1000
        hx_condenser = (m_ref * (h7-h8))/1000
        hx_desorber_temp = {"t_hot_in": self.t_flue_in, "t_hot_out": self.t_flue_out, "t_cold_in": t3, "t_cold_out": t4}
        hx_evaporator_temp = {"t_hot_in": 12+273.15, "t_hot_out": 7+273.15, "t_cold_in": t9, "t_cold_out": t10}
        hx_absorber_temp = {"t_hot_in": t6, "t_hot_out": t1, "t_cold_in": 25+273.15, "t_cold_out": 30+273.15}
        hx_condenser_temp = {"t_hot_in": t7, "t_hot_out": t8, "t_cold_in": 25+273.15, "t_cold_out": 30+273.15}
        hx_desorber_u = 100
        hx_evaporator_u = 1000
        hx_absorber_u = 1000
        hx_condenser_u = 1000
        
        hx_desorber_f = 1
        hx_evaporator_f = 1
        hx_absorber_f = 1
        hx_condenser_f = 1

        factors_desorber = {
            'material_factor': 2.1,
            'temperature_factor': 1,
            'pressure_factor': 1.3,
            'piping_factor': 0.46
        }
        factors_evaporator = {
            'material_factor': 1,
            'temperature_factor': 1,
            'pressure_factor': 2,
            'piping_factor': 0.46
        }
        factors_absorber = {
            'material_factor': 2.1,
            'temperature_factor': 1,
            'pressure_factor': 2,
            'piping_factor': 0.46
        }
        factors_condenser = {
            'material_factor': 1,
            'temperature_factor': 1,
            'pressure_factor': 1.3,
            'piping_factor': 0.46
        }
        cooling_tower_cost_factors = {
            'material_factor': 1,
            'temperature_factor': 1,
            'pressure_factor': 1,
            'piping_factor': 0.7
        }

        # Cost-index escalation factor (user-supplied), applied to bring
        # every component here up to a March-2026 basis. Like steam.py/
        # orc.py (and unlike super.py/trans.py), this cycle has no IDAES
        # sCO2-Costing-Module turbomachinery cost to escalate separately --
        # the pump and every heat-exchanger-type component here (solution
        # HX, desorber, evaporator, absorber, condenser, cooling tower) are
        # all priced via uo.py's generic 2024-basis correlations -- so only
        # the single "remaining cost" escalation factor applies, to all
        # seven components.
        REMAINING_COST_ESCALATION = 103.625 / 102.6166667

        # Escalate the pump and solution-HX costs (their final values, from
        # the second/"Repeat End" pass above) from uo.py's 2024 basis to
        # March 2026.
        if pump_cost:
            pump_cost['cost'] = _escalate(pump_cost.get('cost'), REMAINING_COST_ESCALATION)
        if solution_hx_cost:
            solution_hx_cost['cost'] = _escalate(solution_hx_cost.get('cost'), REMAINING_COST_ESCALATION)

        # Purchased-equipment cost for the desorber, evaporator, absorber and
        # condenser: each is priced from its duty (hx_*), terminal
        # temperatures (hx_*_temp) and an assumed overall heat-transfer
        # coefficient (hx_*_u) given above, via Area = Q / (U * LMTD),
        # same as uo.py's hx_cost_from_duty but with U supplied directly
        # instead of estimated from stream phase. Each cost below is also
        # escalated to March 2026 via REMAINING_COST_ESCALATION.
        desorber_area, desorber_cost_value = _hx_cost_given_u(hx_desorber, hx_desorber_temp, hx_desorber_u, hx_desorber_f, factors_desorber)
        desorber_cost_value = _escalate(desorber_cost_value, REMAINING_COST_ESCALATION)
        desorber_cost = {'area_m2': desorber_area, 'u_assumed': hx_desorber_u, 'cost': desorber_cost_value}

        evaporator_area, evaporator_cost_value = _hx_cost_given_u(hx_evaporator, hx_evaporator_temp, hx_evaporator_u, hx_evaporator_f, factors_evaporator)
        evaporator_cost_value = _escalate(evaporator_cost_value, REMAINING_COST_ESCALATION)
        evaporator_cost = {'area_m2': evaporator_area, 'u_assumed': hx_evaporator_u, 'cost': evaporator_cost_value}

        absorber_area, absorber_cost_value = _hx_cost_given_u(hx_absorber, hx_absorber_temp, hx_absorber_u, hx_absorber_f, factors_absorber)
        absorber_cost_value = _escalate(absorber_cost_value, REMAINING_COST_ESCALATION)
        absorber_cost = {'area_m2': absorber_area, 'u_assumed': hx_absorber_u, 'cost': absorber_cost_value}

        condenser_area, condenser_cost_value = _hx_cost_given_u(hx_condenser, hx_condenser_temp, hx_condenser_u, hx_condenser_f, factors_condenser)
        condenser_cost_value = _escalate(condenser_cost_value, REMAINING_COST_ESCALATION)
        condenser_cost = {'area_m2': condenser_area, 'u_assumed': hx_condenser_u, 'cost': condenser_cost_value}

        flowrate_water = (hx_absorber + hx_condenser)/(1.16*5) # flowrate_water is in m3/hr
        self.flowrate_water = flowrate_water

        # Cooling tower: sized directly on flowrate_water, which is already
        # a volumetric flow rate (m3/hr) here -- unlike the other cycles'
        # cooling towers, no mass-flow-to-volumetric-flow conversion (via
        # fluid density) is needed. Priced with uo.py's generic power-law
        # correlation (equipment_cost('cooling_tower', ...)), same as the
        # other cycles.
        cooling_tower_cost_value = equipment_cost('cooling_tower', flowrate_water, cooling_tower_cost_factors)
        cooling_tower_cost_value = _escalate(cooling_tower_cost_value, REMAINING_COST_ESCALATION)
        self.cooling_tower_cost = {
            'flow_m3hr': flowrate_water,
            'cost': cooling_tower_cost_value,
        }

        # LiBr working-fluid cost (INR): charge = 8 kg per TR of cooling
        # capacity, with TR = q_cooling (kW) / 3.517, priced at 600 INR/kg.
        # i.e. cost = 8 * (q_cooling / 3.517) * 600.
        # float() because q_cooling can come out as a sympy Float (m_weak is
        # solved with sympy above).
        cooling_capacity_tr = float(self.q_cooling) / self.kw_per_tr
        self.libr_mass_kg = self.libr_charge_kg_per_tr * cooling_capacity_tr
        self.libr_fluid_cost = self.libr_mass_kg * self.libr_price_inr_per_kg
        working_fluid_cost = {
            'cooling_capacity_tr': cooling_capacity_tr,
            'mass_kg': self.libr_mass_kg,
            'price_per_kg': self.libr_price_inr_per_kg,
            'cost': self.libr_fluid_cost,
        }

        # Total cost for all 7 priced components (pump, solution heat
        # exchanger (hx10), desorber, evaporator, absorber, condenser and
        # cooling tower) plus the LiBr working-fluid charge.
        total_cost = {
            'pump': pump_cost,
            'solution_hx': solution_hx_cost,
            'desorber': desorber_cost,
            'evaporator': evaporator_cost,
            'absorber': absorber_cost,
            'condenser': condenser_cost,
            'cooling_tower': self.cooling_tower_cost,
            'working_fluid': working_fluid_cost,
            'total': (
                _safe_cost(pump_cost)
                + _safe_cost(solution_hx_cost)
                + _safe_cost(desorber_cost)
                + _safe_cost(evaporator_cost)
                + _safe_cost(absorber_cost)
                + _safe_cost(condenser_cost)
                + _safe_cost(self.cooling_tower_cost)
                + _safe_cost(working_fluid_cost)
            )
        }
        self.total_cost = total_cost

        self.exergy = dict(ex_in=ex_in, ex_out=ex_out, pump=pump_exergy, solution_hx=exergy_solution_hx)  # for the Sankey diagram in app.py
        return (self.cop, ex_eff, self.q_cooling, flue, total_cost)


# In[2]:


#flue = 'HEOS::Nitrogen[0.7130895364138478]&Oxygen[0.1102407598793674]&Water[0.13729792992516215]&CO2[0.03937177378162277]'
#test1 = libr(200+273.15, 160_000, 8.5, flue, 1+273.15, 40+273.15, 0.65, 0.75, (0,0,0,0), 10.94)
#print(test1.calc())

