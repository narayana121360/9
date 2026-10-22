#!/usr/bin/env python
# coding: utf-8

import CoolProp.CoolProp as CP


# ======================================================================
# Flue gas composition of the SOFC exhaust (after the afterburner) per
# 1 kg of fuel, for any fuel defined in FUELS below.
#
# Every fuel is written as a mixture of components CxHyOzNw:
#     CxHyOzNw + (x + y/4 - z/2) O2  ->  x CO2 + y/2 H2O + w/2 N2
# Methanol/ethanol contain oxygen, so they need less air. CO2 in biogas
# needs no O2 and passes straight into the flue gas. N in ammonia
# leaves as N2.
# ======================================================================

# Atomic masses (g/mol)
M_C, M_H, M_O, M_N = 12.0107, 1.00794, 15.9994, 14.0067
M_O2 = 2 * M_O               # 31.99886
M_N2 = 2 * M_N               # 28.0134
M_H2O = 2 * M_H + M_O        # 18.01528
M_CO2 = M_C + 2 * M_O        # 44.0095

# Humid air, mass fractions (same values as the original methane code)
AIR_O2 = 0.23057397
AIR_N2 = 0.76165843
AIR_H2O = 0.0077676014

# Fuel database: {name: {component formula (C, H, O, N): mole fraction}}
# Kerosene, diesel, LPG and biogas are typical average compositions.
# Change them here if your fuel specification is different.
FUELS = {
    'methane':  {(1, 4, 0, 0): 1.0},                           # CH4
    'methanol': {(1, 4, 1, 0): 1.0},                           # CH3OH
    'ethanol':  {(2, 6, 1, 0): 1.0},                           # C2H5OH
    'kerosene': {(12, 26, 0, 0): 1.0},                         # C12H26 (n-dodecane surrogate)
    'diesel':   {(12, 23, 0, 0): 1.0},                         # C12H23 (average diesel formula)
    'co':       {(1, 0, 1, 0): 1.0},                           # CO
    'hydrogen': {(0, 2, 0, 0): 1.0},                           # H2
    'biogas':   {(1, 4, 0, 0): 0.60, (1, 0, 2, 0): 0.40},      # 60% CH4 + 40% CO2 (mol)
    'lpg':      {(3, 8, 0, 0): 0.40, (4, 10, 0, 0): 0.60},     # 40% propane + 60% butane (mol)
    'ammonia':  {(0, 3, 0, 1): 1.0},                           # NH3
}

# Other names that are accepted for the same fuel
FUEL_ALIASES = {
    'ch4': 'methane', 'ch3oh': 'methanol', 'meoh': 'methanol',
    'c2h5oh': 'ethanol', 'etoh': 'ethanol', 'carbon monoxide': 'co',
    'h2': 'hydrogen', 'nh3': 'ammonia', 'jet fuel': 'kerosene',
}


def fuel_stoichiometry(fuel):
    """Per 1 kg of fuel, return (kg O2 needed, kg H2O formed, kg CO2 formed,
    kg N2 formed, kg steam per unit S/C ratio)."""
    name = fuel.strip().lower()
    name = FUEL_ALIASES.get(name, name)
    if name not in FUELS:
        raise ValueError(f"Unknown fuel '{fuel}'. Available: {', '.join(FUELS)}")

    mix = FUELS[name]
    total = sum(mix.values())
    M_fuel = n_O2 = n_H2O = n_CO2 = n_N2 = n_C_reform = 0.0
    for (x, y, z, w), frac in mix.items():
        frac /= total
        M_fuel += frac * (x * M_C + y * M_H + z * M_O + w * M_N)
        n_O2 += frac * (x + y / 4 - z / 2)
        n_CO2 += frac * x
        n_H2O += frac * y / 2
        n_N2 += frac * w / 2
        # S/C ratio counts carbon that can be reformed (not carbon already in CO2)
        if (x, y, z, w) != (1, 0, 2, 0):
            n_C_reform += frac * x

    return (n_O2 * M_O2 / M_fuel,
            n_H2O * M_H2O / M_fuel,
            n_CO2 * M_CO2 / M_fuel,
            n_N2 * M_N2 / M_fuel,
            n_C_reform * M_H2O / M_fuel,
            name)


def mixture_string(names, fractions, backend='HEOS::'):
    """CoolProp mixture string, leaving out components that are zero
    (e.g. no CO2 for hydrogen or ammonia)."""
    parts = [f'{n}[{f}]' for n, f in zip(names, fractions) if f > 0]
    return backend + '&'.join(parts)


# To find composition of flue gas when 1 kg of fuel is utilized
class composition:
    def __init__(self, fuel, power, t, p, u_ratio, stream_carbon_ratio, excess_air):
        self.fuel = fuel
        self.u_ratio = u_ratio
        self.stream_carbon_ratio = stream_carbon_ratio
        self.excess_air = excess_air
        self.power = power
        self.t = t
        self.p = p
        self.mass_flow = None
        self.mf_nitrogen = None
        self.mf_oxygen = None
        self.mf_water = None
        self.mf_co2 = None
        self.verbose = (__name__ == "__main__")

    def calc(self):
        O2, H2O, CO2, N2_fuel, s_c_ratio_conversion, name = fuel_stoichiometry(self.fuel)

        if self.verbose:
            print(f"{name.capitalize()} is selected as Fuel Cell Fuel")
            print("Excess Air: ", self.excess_air)
            print("The H2O/C Ratio is", self.stream_carbon_ratio)
            print("The s_c_ratio_conversion is", s_c_ratio_conversion)
            print("The Fuel Cell Utilization Ratio is", self.u_ratio)
            if s_c_ratio_conversion == 0:
                print("(Fuel has no carbon, so no reforming steam is added)")

        # Same equations as the original methane code, per kg of fuel
        s_c_mass_ratio = self.stream_carbon_ratio * s_c_ratio_conversion
        stoich_air_requirement = O2 * (self.u_ratio / 100) / AIR_O2
        air_requirement = stoich_air_requirement * (1 + (self.excess_air / 100))
        nitrogen_in_flue = air_requirement * AIR_N2 + N2_fuel
        co2_in_flue = CO2
        oxygen_in_flue = (air_requirement * AIR_O2) - O2
        water_in_flue = (air_requirement * AIR_H2O) + H2O + s_c_mass_ratio

        if oxygen_in_flue < 0:
            raise ValueError(
                f"Not enough air to burn the fuel completely: O2 in flue = "
                f"{oxygen_in_flue:.4f} kg/kg fuel. Increase excess air so that "
                f"(u_ratio/100)*(1+excess_air/100) >= 1.")

        total = nitrogen_in_flue + co2_in_flue + oxygen_in_flue + water_in_flue
        moles_nitrogen = 1000 * nitrogen_in_flue / M_N2
        moles_co2 = 1000 * co2_in_flue / M_CO2
        moles_oxygen = 1000 * oxygen_in_flue / M_O2
        moles_water = 1000 * water_in_flue / M_H2O
        total_moles = moles_nitrogen + moles_co2 + moles_oxygen + moles_water

        mf_nitrogen = moles_nitrogen / total_moles
        mf_co2 = moles_co2 / total_moles
        mf_oxygen = moles_oxygen / total_moles
        mf_water = moles_water / total_moles
        flue = ['Nitrogen', 'Oxygen', 'Water', 'CO2']
        flue_mole_fractions = [mf_nitrogen, mf_oxygen, mf_water, mf_co2]

        if self.verbose:
            print("The Total Air Requirement is", air_requirement, "kg/kg fuel")
            print("Flue gas per kg fuel (kg): N2", nitrogen_in_flue, " O2", oxygen_in_flue,
                  " H2O", water_in_flue, " CO2", co2_in_flue)
            print("Steam added per kg fuel (kg):", s_c_mass_ratio)
            print('Total flue mass =', total)
            print('Total flue moles =', total_moles)
            print(flue)
            print(flue_mole_fractions)

        flue_string = mixture_string(flue, flue_mole_fractions)

        # Reference enthalpy at 25 °C: dry gas + liquid water (as in original)
        total_moles_noh2o = moles_nitrogen + moles_co2 + moles_oxygen
        flue_noh2o = ['Nitrogen', 'Oxygen', 'CO2']
        fractions_noh2o = [moles_nitrogen / total_moles_noh2o,
                           moles_oxygen / total_moles_noh2o,
                           moles_co2 / total_moles_noh2o]
        flue_string_noh2o = mixture_string(flue_noh2o, fractions_noh2o, backend='')

        h_noh2o = CP.PropsSI('H', 'T', 25 + 273.15, 'P', 101325, flue_string_noh2o)
        h_h2o = CP.PropsSI('H', 'T', 25 + 273.15, 'P', 101325, "Water")
        mass_fraction_h2o = water_in_flue / total
        mass_fraction_noh2o = 1 - mass_fraction_h2o
        h = (h_noh2o * mass_fraction_noh2o) + (h_h2o * mass_fraction_h2o)

        test1 = CP.PropsSI('H', 'T|gas', self.t, 'P', self.p, flue_string)
        heat = (test1 - h) / 1000
        self.mass_flow = self.power / heat

        self.mf_nitrogen = mf_nitrogen
        self.mf_oxygen = mf_oxygen
        self.mf_water = mf_water
        self.mf_co2 = mf_co2

        return (flue_mole_fractions, self.mass_flow)


if __name__ == "__main__":
    # Example: flue gas composition for every fuel
    # (power in kW, t in K, p in Pa, u_ratio in %, S/C ratio, excess air in %)
    print(f"{'Fuel':<10}{'N2':>9}{'O2':>9}{'H2O':>9}{'CO2':>9}{'Mass flow':>12}")
    for fuel in FUELS:
        c = composition(fuel, power=1000, t=700 + 273.15, p=101325,
                        u_ratio=85, stream_carbon_ratio=2.5, excess_air=200)
        c.verbose = False
        fr, m = c.calc()
        print(f"{fuel:<10}" + "".join(f"{v:>9.4f}" for v in fr) + f"{m:>12.4f}")
