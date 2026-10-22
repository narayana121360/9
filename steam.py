
from scipy.optimize import root_scalar
import CoolProp.CoolProp as CP
import plotly.graph_objects as go
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from uo import generate_sankey_diagram, pure_exergy, tmexergy, turbine, hx2, hx3, hx10, compressor, pump, ref_prop, mixer, hx3_flue, hx_flue, find_t_from_h, hx_pinch_steam, equipment_cost
fluid = 'IF97::Water'


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
    super.py/trans.py.
    """
    return None if cost_value is None else cost_value * factor


# In[2]:


class steam:
    def __init__(self, t_flue_in, p_flue_in, m_flue, flue, turbine_isen_eff, pump_eff, pressure_loss):
        self.t_flue_in = t_flue_in
        self.p_flue_in = p_flue_in
        self.m_flue = m_flue
        self.h_flue_in = None
        self.flue = flue
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
        self.quality_after_heater = None
        self.t_flue_out = None
        self.p_flue_out = None
        self.h_flue_out = None
        self.flowrate_water = None
        self.t_cond_cold_in = None
        self.p_cond_cold_in = None
        self.cooling_tower_cost = None




    def calc(self):
        self.max_m = 1.55 * self.m_flue/8.5

        pinch = 10
        c = None

        # Purchased-equipment cost factors (material, temperature, pressure,
        # piping), applied by uo.py's equipment_cost()/hx_cost_from_duty() as:
        #   Adjusted Cost = Base_Cost * material_factor * temperature_factor
        #                    * pressure_factor * (1 + piping_factor)
        # Set to no-ops here (1, 1, 1, 0) so the reported cost matches the
        # unfactored base correlation cost.
        factors_turbine = {
            'material_factor': 1,
            'temperature_factor': 2.1,
            'pressure_factor': 1.7,
            'piping_factor': 0.21
        }
        factors_pump = {
            'material_factor': 1,
            'temperature_factor': 1,
            'pressure_factor': 1.7,
            'piping_factor': 0.3
        }
        factors_heater = {
            'material_factor': 3.4,
            'temperature_factor': 2.1,
            'pressure_factor': 1.7,
            'piping_factor': 0.46
        }
        factors_condenser = {
            'material_factor': 1,
            'temperature_factor': 1,
            'pressure_factor': 2,
            'piping_factor': 0.46
        }
        cooling_tower_cost_factors = {
            'material_factor': 1,
            'temperature_factor': 1,
            'pressure_factor': 1,
            'piping_factor': 0.7
        }

        # Cost-index escalation factor (user-supplied), applied to bring
        # every component here up to a March-2026 basis. Unlike
        # super.py/trans.py, this cycle has no IDAES sCO2-Costing-Module
        # turbomachinery cost to escalate separately -- the turbine here is
        # costed via uo.py's generic turbine() correlation, same 2024 basis
        # as the pump/condenser/heater/cooling tower -- so only the single
        # "remaining cost" escalation factor applies, to all five
        # components.
        REMAINING_COST_ESCALATION = 103.625 / 102.6166667

        f_heater = 0.75
        f_condenser = 1



        #optimize start
        def steam_cycle(mass_flowrate):
            self.t1 = 458.09+273.15
            self.p1 = 7_000_000  #Pa (70 bar) Turbine Inlet Pressure
            condensing_temp = 40 + 273.15
            p3 = CP.PropsSI('P', "T", condensing_temp, "Q", 0, fluid)
            p2 = p3 + self.cond_drop_hot # Pa (40℃ Condensing Temperature) Turbine Outlet Temperature

            #Condenser
            t3 = 35+273.15
            #p_drop_hot = 0
            #p_drop_cold = 0 #(0 bar)
            t_cold_in = 25 + 273.15 # K   (25℃)
            p_cold_in = 250_000 # Pa  (2.5 bar)





            #Cell 2
            # Turbine Checkmark
            self.s1 = CP.PropsSI('S', "T", self.t1, "P", self.p1, fluid)
            self.h1 = CP.PropsSI('H', "T", self.t1, "P", self.p1, fluid)

            t2, h2, turbine_power, turbine_exergy, turbine_cost = turbine(self.p1, self.h1, p2, self.turbine_isen_eff, fluid, mass_flowrate, factors_turbine)
            t_q_out = CP.PropsSI('Q', "H", h2, "P", p2, fluid)
            t_phase_in = CP.PhaseSI("H", self.h1, "P", self.p1, fluid)
            #print("Turbine Power 0 =", f"{turbine_power:.4f}", "kW", 'Q= ', quality, 'h2 = ', h2, 'm = ', mass_flowrate)
            count = 0
            while ((t_q_out < 0.88 and t_q_out >= 0) or ((t_phase_in == 'twophase') or (t_phase_in == 'liquid'))):
                count = count + 1
                if count > 300:
                    break
                #print('count = ', count, 't_phase_in = ', t_phase_in, 't_q_out = ', t_q_out)
                #print('stuck x', 'q = ', quality, 'p1 = ', self.p1, "q_ in = ", q_in)
                self.p1 = self.p1 - 25_000 if self.p1>126_000 else self.p1
                #self.p1 = 101_325 if ((self.p1 <= 125_000) and (self.p1 >= 101_325))
                self.h1 = CP.PropsSI('H', "T", self.t1, "P", self.p1, fluid)
                t2, h2, turbine_power, turbine_exergy, turbine_cost = turbine(self.p1, self.h1, p2, self.turbine_isen_eff, fluid, mass_flowrate, factors_turbine)
                t_q_out = CP.PropsSI('Q', "H", h2, "P", p2, fluid)
            self.s1 = CP.PropsSI('S', "T", self.t1, "P", self.p1, fluid)
            #print("Turbine Power 00 =", f"{turbine_power:.4f}", "kW")
            #print('self.p1 = ', self.p1/100_000)
            #print("Turbine Power =", f"{turbine_power:.4f}", "kW")
            #print(self.p1, self.h1, p2, self.turbine_isen_eff, fluid, mass_flowrate)
            #print(t2-273.15, h2, turbine_power, turbine_exergy)
            #print("t2 =", f"{t2-273.15:.4f}", "C")
            #print("h2 =", f"{h2/1000:.4f}", "kW")
            #print(turbine_exergy)


            #Cell 3
            p3 = CP.PropsSI('P', "T", condensing_temp, "Q", 0, fluid)
            h3 = CP.PropsSI('H', "T", t3, "P", p3, fluid)
            h_cold_in = CP.PropsSI('H', "T", t_cold_in, "P", p_cold_in, fluid)
            t_water_out, p_water_out, h_water_out, hx_condenser, flowrate_water, condenser_exergy, condenser_cost = hx2(fluid, p2, h2, h3, mass_flowrate, self.cond_drop_hot, fluid, p_cold_in, h_cold_in, self.cond_drop_cold, 6, f_condenser, factors_condenser)
            self.flowrate_water = flowrate_water
            self.t_cond_cold_in = t_cold_in
            self.p_cond_cold_in = p_cold_in
            if __name__ == "__main__":
                print(t_water_out, p_water_out, h_water_out, hx_condenser, flowrate_water, condenser_exergy)

            #Cell 4
            # pump
            p4 = self.p1 + self.cond_drop_hot + self.boiler_drop_cold # Pa (72 bar) Pump Outlet Pressure
            t4, h4, pump_power, pump_exergy, pump_cost = pump(p3, h3, p4, self.pump_eff, fluid, mass_flowrate, factors_pump)
            if __name__ == "__main__":
                print("Pump Power =", f"{pump_power:.4f}", "kW")
                print("t4 =", f"{t4-273.15:.4f}", "C")
                print("p4 =", f"{p4/100000:.4f}", "bar")
                #print(pump_exergy)

            #Cell 5
            # Heater

            self.h_flue_in = CP.PropsSI('H', "T|gas", self.t_flue_in, "P", self.p_flue_in, self.flue)
            self.t_flue_out, self.p_flue_out, self.h_flue_out, t01, p01, h01, hx_heater, heater_exergy, a, heater_cost = hx_pinch_steam(self.flue, self.p_flue_in, self.t_flue_in, self.m_flue, self.boiler_drop_hot, fluid, p4, h4, mass_flowrate,  self.boiler_drop_cold, 10, f_heater, 100, 0.1, factors_heater)
            #print('INPUT ', self.p_flue_in, self.t_flue_in-273.15, self.m_flue, fluid, p4, h4, mass_flowrate)
            #print('OUTPUT', t_flue_out-273.15, p_flue_out, self.h_flue_out, t01, p01, h01, hx_heater)
            #print('h1 = ', self.h1/1000, 'h01 = ', h01/1000, 'P1 = ', p01/100_000)


            #To make self.t1 and t01 equal
            check3 = 0.1
            check4 = None
            while abs(h01-self.h1)>=10:
                #print('struck here 2')
                #print('h1 = ', self.h1, 'h01 = ', h01)
                self.h1 = h01
                self.t1 = CP.PropsSI('T', "H", self.h1, "P", self.p1, fluid)
                #print("self.t1 ", self.t1-273.15)
                p2 = p3 + self.cond_drop_hot # Pa (40℃ Condensing Temperature) Turbine Outlet Temperature

                #Condenser
                t3 = 35+273.15
                p_drop_hot = 0
                p_drop_cold = 0 #(0 bar)
                t_cold_in = 25 + 273.15 # K   (25℃)
                p_cold_in = 250_000 # Pa  (2.5 bar)






                #Cell 2
                # Turbine
                #t_turbine_sat = CP.PropsSI('T', "Q", 1, "P", self.p1, fluid)
                #if (t_turbine_sat - 0.1)  <= self.t1 <= (t_turbine_sat + 0.1):
                #    self.s1 = CP.PropsSI('S', "Q", 1, "P", self.p1, fluid)
                #    #self.h1 = CP.PropsSI('H', "Q", 1, "P", self.p1, fluid)

                #else:
                #    self.s1 = CP.PropsSI('S', "H", self.h1, "P", self.p1, fluid)
                #   #self.h1 = CP.PropsSI('H', "T", self.t1, "P", self.p1, fluid)



                t2, h2, turbine_power, turbine_exergy, turbine_cost = turbine(self.p1, self.h1, p2, self.turbine_isen_eff, fluid, mass_flowrate, factors_turbine)
                #print((self.p1, self.h1, p2, self.turbine_isen_eff, fluid, mass_flowrate))
                #print("Turbine Power 1 =", f"{turbine_power:.4f}", "kW")
                t_q_out = CP.PropsSI('Q', "H", h2, "P", p2, fluid)
                t_phase_in = CP.PhaseSI("H", self.h1, "P", self.p1, fluid)
                count = 0
                while ((t_q_out < 0.88 and t_q_out >= 0) or ((t_phase_in == 'twophase') or (t_phase_in == 'liquid'))):
                    count = count + 1
                    if count > 300:
                        #print('count = ', count)
                        break
                    #print('struck y', 'count = ', count)
                    #print('p1 = ', self.p1)
                    #print("P1 = ", self.p1, "Q = ", quality)
                    self.p1 = self.p1 - 25_000 if self.p1>126_000 else self.p1
                    #self.p1 = 101_325 if ((self.p1 <= 125_000) and (self.p1 >= 101_325))
                    #self.h1 = CP.PropsSI('H', "T", self.t1, "P", self.p1, fluid) GOT ERROR (CLOSE TO SATURATION) AND PUT BELOW CODE TO REMOVE ERROR

                    #t_turbine_sat2 = CP.PropsSI('T', "Q", 1, "P", self.p1, fluid)
                    #if (t_turbine_sat2 - 0.1)  <= self.t1 <= (t_turbine_sat2 + 0.1):
                    #   self.h1 = CP.PropsSI('H', "Q", 1, "P", self.p1, fluid)
                    #else:
                    #   self.h1 = CP.PropsSI('H', "T", self.t1, "P", self.p1, fluid)

                    t2, h2, turbine_power, turbine_exergy, turbine_cost = turbine(self.p1, self.h1, p2, self.turbine_isen_eff, fluid, mass_flowrate, factors_turbine)
                    #print("Turbine Power 2 =", f"{turbine_power:.4f}", "kW")
                    t_q_out = CP.PropsSI('Q', "H", h2, "P", p2, fluid)
                self.s1 = CP.PropsSI('S', "H", self.h1, "P", self.p1, fluid)
                #print('self.p1 = ', self.p1/100_000)



                #Cell 3
                p3 = CP.PropsSI('P', "T", condensing_temp, "Q", 0, fluid) 
                h3 = CP.PropsSI('H', "T", t3, "P", p3, fluid)
                h_cold_in = CP.PropsSI('H', "T", t_cold_in, "P", p_cold_in, fluid)
                t_water_out, p_water_out, h_water_out, hx_condenser, flowrate_water, condenser_exergy, condenser_cost = hx2(fluid, p2, h2, h3, mass_flowrate, self.cond_drop_hot, fluid, p_cold_in, h_cold_in, self.cond_drop_cold, 6, f_condenser, factors_condenser)
                self.flowrate_water = flowrate_water
                self.t_cond_cold_in = t_cold_in
                self.p_cond_cold_in = p_cold_in
                #print('flowrate_water = ', flowrate_water)


                #Cell 4
                # pump
                p4 = self.p1 + self.cond_drop_hot + self.boiler_drop_cold # Pa Pump Outlet Pressure
                t4, h4, pump_power, pump_exergy, pump_cost = pump(p3, h3, p4, self.pump_eff, fluid, mass_flowrate, factors_pump)


                #Cell 5
                # Heater

                self.h_flue_in = CP.PropsSI('H', "T|gas", self.t_flue_in, "P", self.p_flue_in, self.flue)
                self.t_flue_out, self.p_flue_out, self.h_flue_out, t01, p01, h01, hx_heater, heater_exergy, self.quality_after_heater, heater_cost = hx_pinch_steam(self.flue, self.p_flue_in, self.t_flue_in, self.m_flue, self.boiler_drop_hot, fluid, p4, h4, mass_flowrate,  self.boiler_drop_cold,  pinch, f_heater, 100, 0.1, factors_heater)
                t_test =  CP.PropsSI('T', "Q", 1, "P", p01, fluid)
                quality = 0 if (t_test -1) < t01 < (t_test + 1)  else CP.PropsSI('H', "T", t01, "P", p01, fluid)
                #check2 = -2000 if (((pinch-1) > e) or ((pinch+1) < e)) else 0
                #print('c= ', self.quality_after_heater)
                if self.quality_after_heater == 1:
                    check3 = 0
                    check4 = 1
                if self.quality_after_heater ==  0:
                    check3 = -turbine_power
                    check4 = 0



            check = -turbine_power if 0 <= self.quality_after_heater <=1 else 0
            net_power = turbine_power - pump_power if check4 == 1 else 0
            #print('check3 = ', check3, 'Net Power = ', net_power)

            net_efficiency = (100 * net_power/hx_heater) if hx_heater != 0 else 0
            #print('Input', self.flue, self.p_flue_in, self.t_flue_in-273.15, self.m_flue, 0, fluid, p4, h4, mass_flowrate, 0,  5)
            #print('Output ', t_flue_out-273.15, p_flue_out, self.h_flue_out, t01-273.15, p01, h01, hx_heater, heater_exergy)
            #print('NET POWER = ', net_power, 'Turbine Power = ', turbine_power, 'pump_power = ', pump_power)

            self.net_p = net_power
            net_efficiency = 100 * net_power/hx_heater if hx_heater != 0 else 0
            self.Heater = hx_heater
            #print("Mass Flowrate = ", mass_flowrate)

            exergy_input = heater_exergy['hot_change']
            exergy_output = turbine_power
            exergy_efficiency = 100 * net_power/exergy_input if exergy_input != 0 else 0
            self.exergy = dict(heater=heater_exergy, turbine=turbine_exergy, pump=pump_exergy, condenser=condenser_exergy)  # for the Sankey diagram in app.py
            return net_power, self.t1, self.p1, self.h1, hx_heater, self.t_flue_out, self.p_flue_out, net_efficiency, exergy_efficiency, turbine_cost, condenser_cost, pump_cost, heater_cost



        #optimize end

        max_y = float('-inf') # Start with the lowest possible number
        optimal_x = None

        # 3. Define the range and precision
        start_m = 0.1
        end_m = 3.0
        step = 0.1 # A smaller step gives more precise results

        # Convert start and end points to integers for the 'range()' function
        # e.g., 0.1 / 0.001 = 100, and 5.0 / 0.001 = 5000
        start_int = int(start_m / step)
        end_int = int(end_m / step)

        # 4. Use a for loop to evaluate each point
        for i in range(start_int, end_int + 1):
            m = i * step            # Convert the integer back to our decimal x
            net_power, t1, p1, h1, hx_heater, self.t_flue_out, self.p_flue_out, net_efficiency, exergy_efficiency, turbine_cost, condenser_cost, pump_cost, heater_cost = steam_cycle(m)      # Calculate the output
            #print('m = ', m, 'y = ', net_power, 'q = ', c)
            #print('struck z')
            #if y<0:
                #if optimal_x == None:
                    #optimal_x = x
                #print('y = ', y)
                #break
            #print('m = ', m, 'Net Power = ', net_power, 'c = ', self.quality_after_heater)

            # 5. Check if the current y is greater than our recorded max_y
            if net_power > max_y:
                max_y = net_power           # Update the new maximum output
                optimal_m = m       # Record the x value that created it
            #print(m, net_power)

        #print(f"The maximum output (y) is {max_y:.4f}")
        #print(f"This occurs at x = {optimal_x:.4f}")

        best_m = optimal_m
        max_value = max_y
        #print(f"Maximum occurs at x = {best_x:.4f}")
        #print(f"Maximum value is = {max_value:.4f}")


        #Final start
        mass_flowrate = best_m
        self.m = mass_flowrate
        net_power, self.t1, self.p1, self.h1, hx_heater, self.t_flue_out, self.p_flue_out, net_efficiency, exergy_efficiency, turbine_cost, condenser_cost, pump_cost, heater_cost = steam_cycle(mass_flowrate)

        # Escalate the turbine/condenser/pump/heater costs from uo.py's
        # 2024 basis to March 2026 via REMAINING_COST_ESCALATION (defined
        # above). Done here, once, on the final converged costs -- not
        # inside steam_cycle(), which is also called repeatedly during the
        # mass-flowrate optimization loop above.
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
        # ...)) -- the same correlation used for the sCO2/transcritical-CO2
        # cycles' cooling towers. This file already imports CoolProp, so the
        # density comes straight from CP.PropsSI, same as super.py. Also
        # escalated to March 2026 via REMAINING_COST_ESCALATION.
        cooling_water_density = CP.PropsSI('D', 'T', self.t_cond_cold_in, 'P', self.p_cond_cold_in, fluid)
        cooling_tower_flow_m3hr = (self.flowrate_water / cooling_water_density) * 3600.0
        cooling_tower_cost_value = equipment_cost('cooling_tower', cooling_tower_flow_m3hr, cooling_tower_cost_factors)
        cooling_tower_cost_value = _escalate(cooling_tower_cost_value, REMAINING_COST_ESCALATION)
        self.cooling_tower_cost = {
            'flow_kgs': self.flowrate_water,
            'flow_m3hr': cooling_tower_flow_m3hr,
            'cost': cooling_tower_cost_value,
        }

        # Total purchased-equipment cost for the power plant: turbine + condenser (hx2) + pump + heater (hx_pinch_steam) + cooling tower
        total_cost = {
            'turbine': turbine_cost,
            'condenser': condenser_cost,
            'pump': pump_cost,
            'heater': heater_cost,
            'cooling_tower': self.cooling_tower_cost,
            'total': (
                _safe_cost(turbine_cost)
                + _safe_cost(condenser_cost)
                + _safe_cost(pump_cost)
                + _safe_cost(heater_cost)
                + _safe_cost(self.cooling_tower_cost)
            )
        }
        self.total_cost = total_cost

        self.h_flue_out = self.h_flue_in - (hx_heater*1000/self.m_flue)
        #print(self.h_flue_in, self.h_flue_out, hx_heater)
        initial_t = CP.PropsSI('T', "Q|gas", 1, "P", self.p_flue_out, self.flue)
        #print(self.h_flue_out, initial_t, 700+273.15, self.flue)
        #print(find_t_from_h(self.h_flue_out, self.p_flue_out, initial_t, 700+273.15, self.flue))
        self.t_flue_out = find_t_from_h(self.h_flue_out, self.p_flue_out, initial_t-1, 702+273.15, self.flue)






        #Cell 6
        #net_power = turbine_power - pump_power

        #if __name__ == "__main__":
            #print(exergy_efficiency)
            #print(heater_exergy['hot_change'])

        #Cell 8
        if __name__ == "__main__":
            print("Total Plant Cost (2024 basis) = ", f"{total_cost['total']:.2f}")
            print("  Turbine   = ", turbine_cost)
            print("  Condenser = ", condenser_cost)
            print("  Pump      = ", pump_cost)
            print("  Heater    = ", heater_cost)
            print("  Cooling Tower = ", self.cooling_tower_cost)
            # Define the nodes (processes/points in the exergy flow)
            # The index of the label corresponds to its position in the diagram.
            # For example, 'Input Exergy' is node 0, 'Turbine' is node 1, etc.
            nodes = [
                'Heater',  # 0
                'Pump',   # 1
                'Turbine', # 2
                'Condensor',   # 3
                'Loss (Cooling Water)', # 4
                'Useful Work Output',   # 5
                'Exergy Dest (Heater)', # 6
                'Exergy Dest (Pump)', # 7
                'Exergy Dest (Turbine)', # 8
                'Exergy Dest (Condenser)', # 9
                'Exergy Input', # 10
                'Steam Cycle' #11
            ]

            # Define the exergy flows (source, target, value)
            # The source and target are the indices of the nodes defined above.
            # Values represent the exergy quantities (e.g., in kW or MJ/s).
            exergy_flows = [
                # Input Exergy to Combustion Chamber
                {'source': 10, 'target': 11, 'value': heater_exergy['hot_change']},




                {'source': 11, 'target': 6, 'value': heater_exergy['destroyed']},



                {'source': 11, 'target': 5, 'value': (turbine_power - pump_exergy['work'])}, # Useful Work Output
                {'source': 11, 'target': 8, 'value': turbine_exergy['destroyed']}, # Exergy to Heat Exchanger



                # Heat Exchanger outputs
                {'source': 11, 'target': 4, 'value': condenser_exergy['cold_change']},

                {'source': 11, 'target': 9, 'value': condenser_exergy['destroyed']},


                {'source': 11, 'target': 7, 'value': pump_exergy['destroyed']}
            ]
            exergy_flows2 = [
                # Input Exergy to Combustion Chamber
                {'source': 10, 'target': 0, 'value': heater_exergy['hot_change']},



                {'source': 0, 'target': 2, 'value': turbine_exergy['in']}, # 1000 units of exergy from fuel
                {'source': 0, 'target': 6, 'value': heater_exergy['destroyed']},

                # Combustion Chamber outputs
                {'source': 2, 'target': 1, 'value': pump_exergy['work']}, # Exergy to Turbine
                {'source': 2, 'target': 5, 'value': (turbine_power - pump_exergy['work'])}, # Useful Work Output
                {'source': 2, 'target': 8, 'value': turbine_exergy['destroyed']}, # Exergy to Heat Exchanger
                {'source': 2, 'target': 3, 'value': turbine_exergy['out']},


                # Heat Exchanger outputs
                {'source': 3, 'target': 4, 'value': condenser_exergy['cold_change']},
                {'source': 3, 'target': 1, 'value': condenser_exergy['hot_out']}, # Exergy to Exhaust Gas (lost to environment)
                {'source': 3, 'target': 9, 'value': condenser_exergy['destroyed']},   # Exergy destruction in Heat Exchanger

                {'source': 1, 'target': 0, 'value': pump_exergy['out']},
                {'source': 1, 'target': 7, 'value': pump_exergy['destroyed']}
            ]

            # Generate and show the Sankey diagram
            sankey_figure = generate_sankey_diagram(exergy_flows, nodes)
            sankey_figure.show()
            sankey_figure = generate_sankey_diagram(exergy_flows2, nodes)
            sankey_figure.show()

        #Cell 9





        flue = (self.t_flue_out, self.p_flue_out, self.h_flue_out, self.m_flue, self.flue)

        return net_efficiency, exergy_efficiency, net_power, flue, total_cost

