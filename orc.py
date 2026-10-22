
from scipy.optimize import root_scalar
import CoolProp.CoolProp as CP
import plotly.graph_objects as go
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from uo import generate_sankey_diagram, pure_exergy, tmexergy, turbine, hx2, hx3, hx10, compressor, pump, ref_prop, mixer, hx3_flue, hx_pinch, hx_pinch_orc, equipment_cost


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
    super.py/trans.py/steam.py.
    """
    return None if cost_value is None else cost_value * factor


# In[2]:


class orc:
    fluid = 'Isobutane'
    def __init__(self, t_hot_in, p_hot_in, m_hot, hot_fluid, turbine_isen_eff, pump_eff, pressure_loss):
        self.t_hot_in = t_hot_in
        self.p_hot_in = p_hot_in
        self.m_hot = m_hot
        self.hot_fluid = hot_fluid
        self.h1 = None
        self.s1 = None
        self.max_m = 1.55
        self.turbine_isen_eff = turbine_isen_eff
        self.pump_eff = pump_eff
        self.cond_drop_hot = pressure_loss[0]
        self.cond_drop_cold = pressure_loss[1]
        self.boiler_drop_hot = pressure_loss[2]
        self.boiler_drop_cold = pressure_loss[3]
        self.t1 = None
        self.p1 = None
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
        self.flowrate_water = None
        self.t_cond_cold_in = None
        self.p_cond_cold_in = None
        self.cooling_tower_cost = None
        # Isobutane working-fluid (inventory) cost: required charge is taken
        # as 0.3 kg per kW of evaporator (heater) heat input, priced at
        # 450 INR/kg.
        self.isobutane_price_inr_per_kg = 450.0
        self.isobutane_charge_kg_per_kw = 0.3
        self.isobutane_mass_kg = None
        self.isobutane_fluid_cost = None



    def calc(self):
        self.max_m = 1.55 * self.m_hot/8.5
        fluid = 'Isobutane'
        water = 'IF97::Water'
        self.t1 = 120+273.15
        self.p1 = 3_200_000  #Pa (32 bar) Turbine Inlet Pressure

        # Purchased-equipment cost factors (material, temperature, pressure,
        # piping), applied by uo.py's equipment_cost()/hx_cost_from_duty() as:
        #   Adjusted Cost = Base_Cost * material_factor * temperature_factor
        #                    * pressure_factor * (1 + piping_factor)
        # Set to no-ops here (1, 1, 1, 0) so the reported cost matches the
        # unfactored base correlation cost.
        factors_turbine = {
            'material_factor': 1,
            'temperature_factor': 1.6,
            'pressure_factor': 1.5,
            'piping_factor': 0.21
        }
        factors_pump = {
            'material_factor': 1,
            'temperature_factor': 1,
            'pressure_factor': 1.5,
            'piping_factor': 0.3
        }
        factors_heater = {
            'material_factor': 2.1,
            'temperature_factor': 1.6,
            'pressure_factor': 1.5,
            'piping_factor': 0.46
        }
        factors_condenser = {
            'material_factor': 2.1,
            'temperature_factor': 1,
            'pressure_factor': 1,
            'piping_factor': 0.46
        }
        cooling_tower_cost_factors = {
            'material_factor': 1,
            'temperature_factor': 1,
            'pressure_factor': 1,
            'piping_factor': 0.7
        }

        # Cost-index escalation factor (user-supplied), applied to bring
        # every component here up to a March-2026 basis. Like steam.py (and
        # unlike super.py/trans.py), this cycle has no IDAES sCO2-Costing-
        # Module turbomachinery cost to escalate separately -- the turbine
        # here is costed via uo.py's generic turbine() correlation, same
        # 2024 basis as the pump/condenser/heater/cooling tower -- so only
        # the single "remaining cost" escalation factor applies, to all
        # five components.
        REMAINING_COST_ESCALATION = 103.625 / 102.6166667

        def my_expression(mass_flowrate):

            #mass_flowrate = 0.1   #kg/s   Mass Flowrate
            pinch = 10
            f_heater = 0.75
            f_condenser = 0.8

            # Turbine Inputs
            condensing_temp = 40 + 273.15
            p3 = CP.PropsSI('P', "T", condensing_temp, "Q", 0, fluid)


            p2 = p3 + self.cond_drop_hot 


            #Condenser
            subcooling = 5
            t2_cond = CP.PropsSI("T", "P", p2, "Q", 0, fluid)
            t3 = condensing_temp - subcooling
            t_cold_in = 25 + 273.15 # K   (25℃)
            p_cold_in = 250_000 # Pa  (2.5 bar)
            h_cold_in = CP.PropsSI('H', "T", t_cold_in, "P", p_cold_in, water)


            # Turbine
            self.s1 = CP.PropsSI('S', "T", self.t1, "P", self.p1, fluid)
            self.h1 = CP.PropsSI('H', "T", self.t1, "P", self.p1, fluid)
            #print(self.p1, self.h1, p2, self.turbine_isen_eff, fluid, mass_flowrate)

            t2, h2, self.turbine_power, turbine_exergy, turbine_cost = turbine(self.p1, self.h1, p2, self.turbine_isen_eff, fluid, mass_flowrate, factors_turbine)


            #print('Turbine power = ', turbine_power)


            p3 = CP.PropsSI('P', "T", condensing_temp, "Q", 0, fluid)
            h3 = CP.PropsSI('H', "T", t3, "P", p3, fluid)

            #print('p3 = ', p3, 't3 = ', t3-273.15)
            t_water_out, p_water_out, h_water_out, hx_condenser, flowrate_water, condenser_exergy, condenser_cost = hx2(fluid, p2, h2, h3, mass_flowrate, self.cond_drop_hot, water, p_cold_in, h_cold_in, self.cond_drop_cold, 5, f_condenser, factors_condenser)
            self.flowrate_water = flowrate_water
            self.t_cond_cold_in = t_cold_in
            self.p_cond_cold_in = p_cold_in

            # pump
            p4 = self.p1 + self.cond_drop_hot + self.boiler_drop_cold # Pa  Pump Outlet Pressure
            t4, h4, pump_power, self.pump_exergy, pump_cost = pump(p3, h3, p4, self.pump_eff, fluid, mass_flowrate, factors_pump)

            # Heater

            h_hot_in = CP.PropsSI('H', "T|gas", self.t_hot_in, "P", self.p_hot_in, self.hot_fluid)
            self.t_hot_out, self.p_hot_out, self.h_hot_out, t01, p01, h01, self.hx_heater, self.heater_exergy, heater_cost = hx_pinch_orc(self.hot_fluid, self.p_hot_in, self.t_hot_in, self.m_hot, self.boiler_drop_hot, fluid, p4, h4, mass_flowrate, self.boiler_drop_cold, pinch, f_heater, 100, 0.1, factors_heater)

            state1 = CP.PhaseSI('P', p01, 'H', h01, fluid)
            while (state1 == 'liquid') or (state1 == 'twophase'):
                self.p1 = self.p1 - 25_000 if self.p1>300_000 else self.p1

                # Turbine Inputs


                p2 = p3 + self.cond_drop_hot # Pa (40℃ Condensing Temperature) Turbine Outlet Temperature


                #Condenser
                subcooling = 5
                t2_cond = CP.PropsSI("T", "P", p2, "Q", 0, fluid)
                t3 = condensing_temp - subcooling
                cond_drop_hot = 0
                cond_drop_cold = 0 #(0 bar)
                t_cold_in = 25 + 273.15 # K   (25℃)
                p_cold_in = 250_000 # Pa  (2.5 bar)


                # Turbine


                t2, h2, self.turbine_power, turbine_exergy, turbine_cost = turbine(self.p1, h01, p2, self.turbine_isen_eff, fluid, mass_flowrate, factors_turbine)


                #print('Turbine power = ', turbine_power)


                p3 = CP.PropsSI('P', "T", condensing_temp, "Q", 0, fluid)
                h3 = CP.PropsSI('H', "T", t3, "P", p3, fluid)
                h_cold_in = CP.PropsSI('H', "T", t_cold_in, "P", p_cold_in, water)
                t_water_out, p_water_out, h_water_out, hx_condenser, flowrate_water, condenser_exergy, condenser_cost = hx2(fluid, p2, h2, h3, mass_flowrate, self.cond_drop_hot, water, p_cold_in, h_cold_in, self.cond_drop_cold, 5, f_condenser, factors_condenser)
                self.flowrate_water = flowrate_water
                self.t_cond_cold_in = t_cold_in
                self.p_cond_cold_in = p_cold_in

                # pump
                p4 = self.p1 + self.cond_drop_hot + self.boiler_drop_cold # Pa  Pump Outlet Pressure
                t4, h4, pump_power, self.pump_exergy, pump_cost = pump(p3, h3, p4, self.pump_eff, fluid, mass_flowrate, factors_pump)

                # Heater

                h_hot_in = CP.PropsSI('H', "T|gas", self.t_hot_in, "P", self.p_hot_in, self.hot_fluid)
                self.t_hot_out, self.p_hot_out, self.h_hot_out, t01, p01, h01, self.hx_heater, self.heater_exergy, heater_cost = hx_pinch_orc(self.hot_fluid, self.p_hot_in, self.t_hot_in, self.m_hot, self.boiler_drop_hot, fluid, p4, h4, mass_flowrate, self.boiler_drop_cold, pinch, f_heater, 100, 0.1, factors_heater)
                state1 = CP.PhaseSI('P', p01, 'H', h01, fluid)
                #print('p1 = ', self.p1)
                #print('Strck inside while loop for checking turine inlet for liquid')




            #To make self.t1 and t01 equal
            while abs(h01-self.h1)>=10:
                self.h1 = h01
                #print('struck here')

                # Turbine Inputs


                p2 = p3 + self.cond_drop_hot # Pa 


                #Condenser
                subcooling = 5
                t2_cond = CP.PropsSI("T", "P", p2, "Q", 0, fluid) 
                t3 = condensing_temp - subcooling
                t_cold_in = 25 + 273.15 # K   (25℃)
                p_cold_in = 250_000 # Pa  (2.5 bar)



                t2, h2, self.turbine_power, turbine_exergy, turbine_cost = turbine(self.p1, self.h1, p2, self.turbine_isen_eff, fluid, mass_flowrate, factors_turbine)


                #print('Turbine power = ', turbine_power)


                p3 = CP.PropsSI('P', "T", condensing_temp, "Q", 0, fluid) 
                h3 = CP.PropsSI('H', "T", t3, "P", p3, fluid)
                h_cold_in = CP.PropsSI('H', "T", t_cold_in, "P", p_cold_in, water)
                t_water_out, p_water_out, h_water_out, hx_condenser, flowrate_water, condenser_exergy, condenser_cost = hx2(fluid, p2, h2, h3, mass_flowrate, self.cond_drop_hot, water, p_cold_in, h_cold_in, self.cond_drop_cold, 5, f_condenser, factors_condenser)
                self.flowrate_water = flowrate_water
                self.t_cond_cold_in = t_cold_in
                self.p_cond_cold_in = p_cold_in

                # pump
                p4 = self.p1 + self.cond_drop_hot + self.boiler_drop_cold # Pa  Pump Outlet Pressure
                t4, h4, pump_power, self.pump_exergy, pump_cost = pump(p3, h3, p4, self.pump_eff, fluid, mass_flowrate, factors_pump)

                # Heater

                h_hot_in = CP.PropsSI('H', "T|gas", self.t_hot_in, "P", self.p_hot_in, self.hot_fluid)
                self.t_hot_out, self.p_hot_out, self.h_hot_out, t01, p01, h01, self.hx_heater, self.heater_exergy, heater_cost = hx_pinch_orc(self.hot_fluid, self.p_hot_in, self.t_hot_in, self.m_hot, self.boiler_drop_hot, fluid, p4, h4, mass_flowrate, self.boiler_drop_cold, pinch, f_heater, 100, 0.1, factors_heater)
                #print('Struck inside while loop for getting correct h')

            net_power = self.turbine_power - pump_power

            self.exergy = dict(heater=self.heater_exergy, turbine=turbine_exergy, pump=self.pump_exergy, condenser=condenser_exergy)  # for the Sankey diagram in app.py
            return net_power, turbine_cost, condenser_cost, pump_cost, heater_cost



        #optimize end

        # Maximum flow rate calculation Start
        h_hot_in = CP.PropsSI('H', "T|gas", self.t_hot_in, "P", self.p_hot_in, self.hot_fluid)
        t_hot_cond = CP.PropsSI('T', "Q|gas", 1, "P", self.p_hot_in - self.boiler_drop_hot, self.hot_fluid)
        h_hot_cond = CP.PropsSI('H', "T|gas", t_hot_cond, "P", self.p_hot_in - self.boiler_drop_hot, self.hot_fluid)
        q_hot_max = self.m_hot * (h_hot_in - h_hot_cond)

        a =CP.PropsSI('H', "Q", 0, "P", 531208, fluid)
        b =CP.PropsSI('H', "Q", 1, "P", 531208, fluid)
        h_fg = b - a
        m_max = q_hot_max/h_fg
        # Maximum flow rate calculation Start

        max_y = float('-inf') # Start with the lowest possible number
        optimal_x = None

        # 3. Define the range and precision
        start_x = 0.1
        end_x = m_max
        step = 0.1 # A smaller step gives more precise results

        # Convert start and end points to integers for the 'range()' function
        # e.g., 0.1 / 0.001 = 100, and 5.0 / 0.001 = 5000
        #start_int = int(start_x / step)
        #end_int = int(end_x / step)
        num_steps = int((end_x - start_x) / step) + 1

        # 4. Use a for loop to evaluate each point
        for i in range(num_steps):
            try:
                x = start_x + (i * step)            # Convert the integer back to our decimal x
                y, turbine_cost, condenser_cost, pump_cost, heater_cost = my_expression(x)      # Calculate the output
                #print('struck z')
                #if y<0:
                    #if optimal_x == None:
                        #optimal_x = x
                    #print('y = ', y)
                    #break
                #print('m = ', x, 'Net Power = ', y, )

                # 5. Check if the current y is greater than our recorded max_y
                if y > max_y:
                    max_y = y           # Update the new maximum output
                    optimal_x = x       # Record the x value that created it
                #print(x, y)
            except ValueError:
                # If it fails, print a warning and "go past it" to the next item
                #print(f"Warning: '{x}' is not a valid number. Skipping.")
                break

        #print(f"The maximum output (y) is {max_y:.4f}")
        #print(f"This occurs at x = {optimal_x:.4f}")

        best_x = optimal_x
        max_value = max_y
        #print(f"Maximum occurs at x = {best_x:.4f}")
        #print(f"Maximum value is = {max_value:.4f}")


        #Final
        mass_flowrate = best_x
        self.t1 = 120+273.15
        self.p1 = 3_200_000  #Pa (32 bar) Turbine Inlet Pressure
        #print('m = ', mass_flowrate)
        net_power, turbine_cost, condenser_cost, pump_cost, heater_cost = my_expression(mass_flowrate)
        self.t1 = CP.PropsSI('T', "P", self.p1, "H", self.h1, fluid)

        # Escalate the turbine/condenser/pump/heater costs from uo.py's
        # 2024 basis to March 2026 via REMAINING_COST_ESCALATION (defined
        # above). Done here, once, on the final converged costs -- not
        # inside my_expression(), which is also called repeatedly during
        # the mass-flowrate optimization loop above.
        if turbine_cost:
            turbine_cost['cost'] = _escalate(turbine_cost.get('cost'), REMAINING_COST_ESCALATION)
        if condenser_cost:
            condenser_cost['cost'] = _escalate(condenser_cost.get('cost'), REMAINING_COST_ESCALATION)
        if pump_cost:
            pump_cost['cost'] = _escalate(pump_cost.get('cost'), REMAINING_COST_ESCALATION)
        if heater_cost:
            heater_cost['cost'] = _escalate(heater_cost.get('cost'), REMAINING_COST_ESCALATION)

        # Cooling tower: sized on the condenser's cooling-water volumetric
        # flow rate (self.flowrate_water, kg/s -> m3/hr via the cooling
        # water's density at the condenser's cold-side inlet), priced with
        # uo.py's generic power-law correlation (equipment_cost('cooling_tower',
        # ...)) -- the same correlation used for the other cycles' cooling
        # towers. This file already imports CoolProp, so the density comes
        # straight from CP.PropsSI, same as steam.py/super.py.
        cooling_water_density = CP.PropsSI('D', 'T', self.t_cond_cold_in, 'P', self.p_cond_cold_in, water)
        cooling_tower_flow_m3hr = (self.flowrate_water / cooling_water_density) * 3600.0
        cooling_tower_cost_value = equipment_cost('cooling_tower', cooling_tower_flow_m3hr, cooling_tower_cost_factors)
        cooling_tower_cost_value = _escalate(cooling_tower_cost_value, REMAINING_COST_ESCALATION)
        self.cooling_tower_cost = {
            'flow_kgs': self.flowrate_water,
            'flow_m3hr': cooling_tower_flow_m3hr,
            'cost': cooling_tower_cost_value,
        }

        # Isobutane working-fluid cost (INR): required charge = 0.3 * evaporator
        # (heater) heat input in kW, in kg, priced at 450 INR/kg.
        # self.hx_heater is that duty in kW (from hx_pinch_orc).
        self.isobutane_mass_kg = self.isobutane_charge_kg_per_kw * self.hx_heater
        self.isobutane_fluid_cost = self.isobutane_mass_kg * self.isobutane_price_inr_per_kg
        working_fluid_cost = {
            'mass_kg': self.isobutane_mass_kg,
            'price_per_kg': self.isobutane_price_inr_per_kg,
            'cost': self.isobutane_fluid_cost,
        }

        # Total cost for the power plant: turbine + condenser (hx2) + pump +
        # heater (hx_pinch_orc) + cooling tower + isobutane working fluid
        total_cost = {
            'turbine': turbine_cost,
            'condenser': condenser_cost,
            'pump': pump_cost,
            'heater': heater_cost,
            'cooling_tower': self.cooling_tower_cost,
            'working_fluid': working_fluid_cost,
            'total': (
                _safe_cost(turbine_cost)
                + _safe_cost(condenser_cost)
                + _safe_cost(pump_cost)
                + _safe_cost(heater_cost)
                + _safe_cost(self.cooling_tower_cost)
                + _safe_cost(working_fluid_cost)
            )
        }
        self.total_cost = total_cost

        #Cell 6

        net_efficiency = 100 * net_power/self.hx_heater
        #print("Mass Flowrate = ", mass_flowrate)

        exergy_input = self.heater_exergy['hot_change']
        exergy_output = self.turbine_power
        exergy_efficiency = 100 * net_power/exergy_input
        #print('t_flue_in = ', self.t_hot_in - 273.15, 't1 = ', self.t1-273.15, 'p1 = ', self.p1/100_000, 'h1 = ', self.h1, 'hx heater = ', self.hx_heater, 'm = ', mass_flowrate, 'net_power = ', net_power, 'self.turbine_power = ', self.turbine_power, 'Pump power = ', self.pump_exergy['work'], 'Total Plant Cost = ', total_cost['total'])
        #print(self.turbine_power, self.heater_exergy['hot_change'], self.pump_exergy['work'], exergy_output, exergy_input, exergy_output/exergy_input, 100 * exergy_output/exergy_input, exergy_efficiency)
        #if __name__ == "__main__":
            #print(exergy_efficiency)
            #print(heater_exergy['hot_change'])








        flue = (self.t_hot_out, self.p_hot_out, self.h_hot_out, self.m_hot, self.hot_fluid)
        return net_efficiency, exergy_efficiency, net_power, flue, total_cost


