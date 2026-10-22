import json
import os
import subprocess
import sys

import numpy as np
import streamlit as st

import sankey
from flue import FUELS, composition

st.set_page_config(page_title="SOFC Flue Gas Composition and Flow Rate", page_icon="🔥", layout="wide")

st.title("SOFC Flue gas Composition and flow rate Calculation")
st.caption("Flue gas composition and mass flow of the SOFC exhaust (after the afterburner).")

BAR = 1e5
K = 273.15

# ---------------- Inputs ----------------
with st.sidebar:
    st.header("Inputs")
    with st.form("inputs"):
        fuel = st.selectbox(
            "Fuel",
            options=list(FUELS.keys()),
            format_func=lambda f: f.upper() if f in ("lpg", "co") else f.capitalize(),
        )
        power = st.number_input("Energy Content (kW)", min_value=0.001, value=8000.0, step=100.0)
        t_c = st.number_input("Flue Gas Inlet Temperature (°C)", min_value=30.0, value=650.0, step=10.0)
        p_bar = st.number_input("Flue Gas Inlet Pressure (bar)", min_value=0.1, value=2.0, step=0.1)
        u_ratio = st.number_input("SOFC Utilization Ratio (%)", min_value=1.0, max_value=100.0, value=60.0, step=1.0)
        sc_ratio = st.number_input("Steam Carbon Ratio", min_value=0.0, value=1.2, step=0.1)
        excess_air = st.number_input("Excess Air (%)", min_value=0.0, value=300.0, step=10.0)
        submitted = st.form_submit_button("Calculate", type="primary", width="stretch")

# ---------------- Calculation ----------------
try:
    c = composition(fuel, power, t_c + K, p_bar * BAR, u_ratio, sc_ratio, excess_air)
    c.verbose = False
    mole_fractions, mass_flow = c.calc()
except ValueError as e:
    st.error(str(e))
    st.stop()
except Exception as e:  # CoolProp failures (e.g. condensing conditions)
    st.error(f"Property calculation failed for these conditions: {e}")
    st.stop()

# Same mixture string the notebook builds and passes to every cycle
flue_string = (f"HEOS::Nitrogen[{c.mf_nitrogen}]&Oxygen[{c.mf_oxygen}]"
               f"&Water[{c.mf_water}]&CO2[{c.mf_co2}]")

# ---------------- Outputs ----------------
labels = ["N₂", "O₂", "H₂O", "CO₂"]

st.subheader(f"Results for {fuel.capitalize()}")
m1, m2 = st.columns(2)
m1.metric("Flue Gas Mass Flow Rate", f"{mass_flow:.4f} kg/s")
m2.metric("Flue Gas Mass Flow Rate", f"{mass_flow * 3600:,.1f} kg/h")

st.markdown("#### Flue Gas Composition")
cols = st.columns(4)
for col, name, x in zip(cols, labels, mole_fractions):
    col.metric(name, f"{x * 100:.2f} %")


# ======================================================================
# Waste heat recovery cycles
# ======================================================================
# Each cycle is solved only when its Run button is pressed, and results are
# cached on the inputs, because the IDAES-based cycles (sCO2, tCO2) take
# minutes to solve.

def _sankey(key, cyc):
    """Exergy Sankey data; a diagram problem never fails the cycle run."""
    try:
        return sankey.build_flows(key, cyc.exergy)
    except Exception as err:
        return {"error": str(err)}


def _breakdown(cost, names):
    out = {}
    for key, label in names.items():
        d = cost.get(key)
        out[label] = None if not d else d.get("cost")
    return out


RUNNER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "co2_runner.py")
RESULT_TAG = "__CO2_RUNNER_RESULT__"


def _call_runner(task, args=None):
    """Runs co2_runner.py in a fresh Python process (see that file for why)
    and returns its JSON result."""
    proc = subprocess.run([sys.executable, RUNNER, task], input=json.dumps(args or {}),
                          capture_output=True, text=True, cwd=os.path.dirname(RUNNER))
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith(RESULT_TAG):
            return json.loads(line[len(RESULT_TAG):])
    raise RuntimeError("The cycle process stopped unexpectedly:\n" + (proc.stderr or proc.stdout)[-1500:])


def _solve_in_runner(task, args):
    out = _call_runner(task, args)
    if "error" in out:
        raise RuntimeError(out["error"])  # raised, not returned, so failures are not cached
    return out


@st.cache_data(show_spinner=False)
def run_super(flue, approach, p1, bypass, htr_eff, ltr_eff, drop_hot, drop_cold, bc, cooler):
    return _solve_in_runner("super", dict(flue=flue, approach=approach, p1=p1, bypass=bypass,
                                          htr_eff=htr_eff, ltr_eff=ltr_eff, drop_hot=drop_hot,
                                          drop_cold=drop_cold, bc=bc, cooler=cooler))


@st.cache_data(show_spinner=False)
def run_steam(t_in, p_in, m_flue, fs, turb_eff, pump_eff, losses):
    import steam
    cyc = steam.steam(t_in, p_in, m_flue, fs, turb_eff, pump_eff, list(losses))
    en, exg, net, f_out, cost = cyc.calc()
    return dict(energy=en, exergy=exg, output=net, heat_in=cyc.Heater,
                t_out=f_out[0], total=cost["total"],
                costs=_breakdown(cost, {"turbine": "Turbine", "pump": "Pump", "heater": "Heater (boiler)",
                                        "condenser": "Condenser", "cooling_tower": "Cooling tower"}),
                sankey=_sankey("steam", cyc))


@st.cache_data(show_spinner=False)
def run_libr(t_in, p_in, m_flue, fs, t_eva, t_cond, shx_eff, pump_eff, losses, circ):
    import libr
    cyc = libr.libr(t_in, p_in, m_flue, fs, t_eva, t_cond, shx_eff, pump_eff, tuple(losses), circ)
    cop, exg, cooling, f_out, cost = cyc.calc()
    costs = _breakdown(cost, {"desorber": "Desorber (generator)", "absorber": "Absorber",
                              "condenser": "Condenser", "evaporator": "Evaporator",
                              "solution_hx": "Solution HX", "pump": "Pump",
                              "cooling_tower": "Cooling tower", "working_fluid": "LiBr charge"})
    return dict(energy=float(cop), exergy=float(exg), output=float(cooling), heat_in=float(cyc.q_heating),
                t_out=float(f_out[0]), total=float(cost["total"]),
                costs={k: (None if v is None else float(v)) for k, v in costs.items()},
                sankey=_sankey("libr", cyc))


@st.cache_data(show_spinner=False)
def run_orc(t_in, p_in, m_flue, fs, turb_eff, pump_eff, losses):
    import orc
    cyc = orc.orc(t_in, p_in, m_flue, fs, turb_eff, pump_eff, tuple(losses))
    en, exg, net, f_out, cost = cyc.calc()
    return dict(energy=en, exergy=exg, output=net, heat_in=cyc.hx_heater,
                t_out=f_out[0], total=cost["total"],
                costs=_breakdown(cost, {"turbine": "Turbine", "pump": "Pump", "heater": "Evaporator",
                                        "condenser": "Condenser", "cooling_tower": "Cooling tower",
                                        "working_fluid": "Isobutane charge"}),
                sankey=_sankey("orc", cyc))


@st.cache_data(show_spinner=False)
def run_trans(t_in, p_in, m_flue, fs, turb_eff, pump_eff, heater_eff, losses):
    return _solve_in_runner("trans", dict(t_in=t_in, p_in=p_in, m_flue=m_flue, fs=fs, turb_eff=turb_eff,
                                          pump_eff=pump_eff, heater_eff=heater_eff, losses=losses))


def run(key, fn, *args):
    with st.spinner("Solving cycle… (the CO₂ cycles can take several minutes)"):
        try:
            st.session_state[key] = fn(*args)
        except Exception as err:
            st.session_state[key] = {"error": str(err)}


SANKEY_TITLES = {"res_super": "Supercritical CO₂ cycle",
                 "res_steam": "Steam Rankine cycle",
                 "res_libr": "LiBr absorption chiller",
                 "res_orc": "Organic Rankine cycle",
                 "res_trans": "Transcritical CO₂ cycle"}


def show_results(key, cooling=False):
    r = st.session_state.get(key)
    if r is None:
        return
    if "error" in r:
        st.error(f"The cycle could not be solved for these inputs: {r['error']}")
        return
    a, b, cc = st.columns(3)
    if cooling:
        a.metric("COP", f"{r['energy']:.3f}")
        b.metric("Exergy Efficiency", f"{r['exergy']:.2f} %")
        cc.metric("Cooling Capacity", f"{r['output']:,.1f} kW")
    else:
        a.metric("Energy Efficiency", f"{r['energy']:.2f} %")
        b.metric("Exergy Efficiency", f"{r['exergy']:.2f} %")
        cc.metric("Net Power Output", f"{r['output']:,.1f} kW")
    d, e, f = st.columns(3)
    d.metric("Heat Recovered from Flue Gas", f"{r['heat_in']:,.1f} kW")
    e.metric("Flue Gas Exit Temperature", f"{r['t_out'] - K:.1f} °C")
    total = r["total"]
    f.metric("Total Plant Cost", "N/A" if total is None else f"₹ {total:,.0f}")
    sk = r.get("sankey")
    if sk and "error" not in sk:
        st.markdown("#### Exergy Sankey diagrams")
        st.caption(f"Exergy efficiency = {sk['formula']} = **{r['exergy']:.2f} %** "
                   "(green ÷ orange in the overall exergy balance). "
                   "Exergy rates in kW. Orange: exergy input · Green: useful output · "
                   "Red: exergy destruction · Blue: exergy leaving with the cooling water or to the "
                   "bottoming cycle · Grey: exergy circulating inside the cycle.")
        for i, d in enumerate(sk["diagrams"]):
            st.plotly_chart(sankey.figure(d["flows"], f"{SANKEY_TITLES[key]} – {d['name']}"),
                            width="stretch", key=f"sankey_{key}_{i}")
    elif sk:
        st.warning(f"The exergy Sankey diagrams could not be drawn: {sk['error']}")
    with st.expander("Cost breakdown (INR, March 2026 basis)"):
        for name, v in r["costs"].items():
            st.write(f"**{name}:** " + ("N/A" if v is None else f"₹ {v:,.0f}"))


@st.cache_resource(show_spinner=False)
def ensure_idaes():
    """Makes sure the IDAES add-ons needed by the CO2 cycles are installed,
    downloading them automatically if not. Runs once per server (shared by
    every user of the app), so on Streamlit Cloud the first CO2 run after a
    reboot does the setup and later runs start straight away. A failed
    attempt is not cached, so the next run tries again."""
    status = _call_runner("status")
    if status["ok"]:
        return status
    result = _call_runner("install")
    status = _call_runner("status")
    if not status["ok"]:
        raise RuntimeError(status["message"] + "\n\nAutomatic install log:\n" + result.get("log", ""))
    return status


def idaes_ready():
    """Called when a CO2 cycle's Run button is pressed. Returns True when the
    cycle can run; otherwise shows what went wrong."""
    with st.spinner("Preparing the CO₂ property models… the first run after the app "
                    "starts downloads them, which can take a few minutes"):
        try:
            ensure_idaes()
            return True
        except RuntimeError as err:
            st.error("The IDAES add-ons needed for this cycle could not be set up automatically. "
                     "Press Run again to retry.")
            with st.expander("Details"):
                st.code(str(err)[-3000:])
            return False


def drop(label, default, key):
    """Pressure-drop input in bar, returned in Pa."""
    return st.number_input(f"{label} (bar)", min_value=0.0, value=default, step=0.1, format="%.2f", key=key) * BAR


st.divider()
st.header("Waste Heat Recovery Cycles")
st.caption("Each cycle uses the flue gas composition and mass flow rate calculated above. "
           "Set the flue gas inlet conditions for each cycle and press Run.")

tabs = st.tabs(["Supercritical CO₂", "Steam Rankine", "LiBr Absorption", "Organic Rankine (ORC)",
                "Transcritical CO₂"])

# ---------------- Supercritical CO2 ----------------
with tabs[0]:
    st.caption("Recompression sCO₂ Brayton cycle. The flue gas enters at the SOFC flue gas "
               "temperature and pressure set in the sidebar.")
    with st.form("super"):
        a, b, cc = st.columns(3)
        approach = a.number_input("Flue Gas HX Approach (K)", min_value=1.0, value=20.0, step=1.0)
        p1 = b.number_input("Turbine Inlet Pressure (bar)", min_value=80.0, value=239.0, step=1.0)
        bypass = cc.number_input("Bypass Fraction", min_value=0.0, max_value=0.9, value=0.348, step=0.01, format="%.3f")
        a, b = st.columns(2)
        htr = a.number_input("HTR Effectiveness", min_value=0.1, max_value=0.999, value=0.973982, step=0.01, format="%.4f")
        ltr = b.number_input("LTR Effectiveness", min_value=0.1, max_value=0.999, value=0.936902, step=0.01, format="%.4f")
        with st.expander("Bottoming cycle and cooling water"):
            a, b, cc = st.columns(3)
            t_bc = a.number_input("Bottoming Cycle CO₂ Inlet Temperature (°C)", value=40.0)
            p_bc = b.number_input("Bottoming Cycle CO₂ Inlet Pressure (bar)", value=120.0)
            m_bc = cc.number_input("Bottoming Cycle CO₂ Flow (kg/s)", min_value=0.01, value=0.33)
            a, b = st.columns(2)
            t_cw = a.number_input("Cooling Water Inlet Temperature (°C)", value=25.0, key="s_tcw")
            p_cw = b.number_input("Cooling Water Inlet Pressure (bar)", value=2.5, key="s_pcw")
        with st.expander("Pressure drops"):
            a, b = st.columns(2)
            with a:
                st.markdown("**Hot side**")
                hot = [drop("HTR", 2.0, "sh1"), drop("LTR", 2.0, "sh2"), drop("Bottoming-cycle HX", 1.0, "sh3"),
                       drop("Cooler", 1.0, "sh4"), 0.0, drop("Flue gas HX, flue side", 0.2, "sh6")]
            with b:
                st.markdown("**Cold side**")
                cold = [drop("HTR", 6.0, "sc1"), drop("LTR", 5.0, "sc2"), drop("Bottoming-cycle HX", 4.0, "sc3"),
                        drop("Cooler", 1.0, "sc4"), 0.0, drop("Flue gas HX, CO₂ side", 0.2, "sc6")]
        go = st.form_submit_button("Run Supercritical CO₂ Cycle", type="primary")
    if go and idaes_ready():
        run("res_super", run_super, (t_c + K, p_bar * BAR, mass_flow, flue_string), approach, p1 * BAR,
            bypass, htr, ltr, tuple(hot), tuple(cold), (t_bc + K, p_bc * BAR, m_bc, "CO2"),
            (t_cw + K, p_cw * BAR, "IF97::Water"))
    show_results("res_super")

# ---------------- Steam Rankine ----------------
with tabs[1]:
    st.caption("Steam Rankine cycle. The water flow rate is optimised for maximum net power.")
    with st.form("steam"):
        a, b = st.columns(2)
        t_in = a.number_input("Flue Gas Inlet Temperature (°C)", min_value=60.0, value=400.0, step=10.0, key="st_t")
        p_in = b.number_input("Flue Gas Inlet Pressure (bar)", min_value=0.5, value=1.8, step=0.1, key="st_p")
        a, b = st.columns(2)
        te = a.number_input("Turbine Isentropic Efficiency", min_value=0.1, max_value=1.0, value=0.75, step=0.01, key="st_te")
        pe = b.number_input("Pump Efficiency", min_value=0.1, max_value=1.0, value=0.75, step=0.01, key="st_pe")
        with st.expander("Pressure drops"):
            a, b = st.columns(2)
            with a:
                l1 = drop("Condenser, steam side", 0.0, "st_l1")
                l3 = drop("Boiler, flue gas side", 0.2, "st_l3")
            with b:
                l2 = drop("Condenser, cooling water side", 1.0, "st_l2")
                l4 = drop("Boiler, water side", 2.0, "st_l4")
        go = st.form_submit_button("Run Steam Rankine Cycle", type="primary")
    if go:
        run("res_steam", run_steam, t_in + K, p_in * BAR, mass_flow, flue_string, te, pe, (l1, l2, l3, l4))
    show_results("res_steam")

# ---------------- LiBr absorption ----------------
with tabs[2]:
    st.caption("Single-effect LiBr–water vapour absorption chiller driven by the flue gas.")
    with st.form("libr"):
        a, b = st.columns(2)
        t_in = a.number_input("Flue Gas Inlet Temperature (°C)", min_value=60.0, value=140.88, step=5.0, key="li_t")
        p_in = b.number_input("Flue Gas Inlet Pressure (bar)", min_value=0.5, value=1.6, step=0.1, key="li_p")
        a, b, cc = st.columns(3)
        t_eva = a.number_input("Evaporator Temperature (°C)", value=1.0, step=1.0)
        t_cond = b.number_input("Condenser Temperature (°C)", value=40.0, step=1.0)
        circ = cc.number_input("Circulation Ratio", min_value=1.1, value=10.94, step=0.1)
        a, b = st.columns(2)
        shx = a.number_input("Solution HX Effectiveness", min_value=0.1, max_value=1.0, value=0.65, step=0.01)
        pe = b.number_input("Pump Efficiency", min_value=0.1, max_value=1.0, value=0.75, step=0.01, key="li_pe")
        with st.expander("Pressure drops"):
            a, b = st.columns(2)
            with a:
                l1 = drop("Solution HX, hot side", 0.0, "li_l1")
                l3 = drop("Desorber, flue gas side", 0.2, "li_l3")
            with b:
                l2 = drop("Solution HX, cold side", 0.0, "li_l2")
                l4 = drop("Desorber, solution side", 0.0, "li_l4")
        go = st.form_submit_button("Run LiBr Absorption Cycle", type="primary")
    if go:
        run("res_libr", run_libr, t_in + K, p_in * BAR, mass_flow, flue_string, t_eva + K, t_cond + K,
            shx, pe, (l1, l2, l3, l4), circ)
    show_results("res_libr", cooling=True)

# ---------------- ORC ----------------
with tabs[3]:
    st.caption("Isobutane organic Rankine cycle. The isobutane flow rate is optimised for maximum net power.")
    with st.form("orc"):
        a, b = st.columns(2)
        t_in = a.number_input("Flue Gas Inlet Temperature (°C)", min_value=60.0, value=200.0, step=10.0, key="or_t")
        p_in = b.number_input("Flue Gas Inlet Pressure (bar)", min_value=0.5, value=1.6, step=0.1, key="or_p")
        a, b = st.columns(2)
        te = a.number_input("Turbine Isentropic Efficiency", min_value=0.1, max_value=1.0, value=0.70, step=0.01, key="or_te")
        pe = b.number_input("Pump Efficiency", min_value=0.1, max_value=1.0, value=0.75, step=0.01, key="or_pe")
        with st.expander("Pressure drops"):
            a, b = st.columns(2)
            with a:
                l1 = drop("Condenser, isobutane side", 0.0, "or_l1")
                l3 = drop("Evaporator, flue gas side", 0.0, "or_l3")
            with b:
                l2 = drop("Condenser, cooling water side", 0.0, "or_l2")
                l4 = drop("Evaporator, isobutane side", 0.0, "or_l4")
        go = st.form_submit_button("Run Organic Rankine Cycle", type="primary")
    if go:
        run("res_orc", run_orc, t_in + K, p_in * BAR, mass_flow, flue_string, te, pe, (l1, l2, l3, l4))
    show_results("res_orc")

# ---------------- Transcritical CO2 ----------------
with tabs[4]:
    st.caption("Transcritical CO₂ Rankine cycle.")
    with st.form("trans"):
        a, b = st.columns(2)
        t_in = a.number_input("Flue Gas Inlet Temperature (°C)", min_value=60.0, value=700.0, step=10.0, key="tr_t")
        p_in = b.number_input("Flue Gas Inlet Pressure (bar)", min_value=0.5, value=1.6, step=0.1, key="tr_p")
        a, b, cc = st.columns(3)
        te = a.number_input("Turbine Isentropic Efficiency", min_value=0.1, max_value=1.0, value=0.70, step=0.01, key="tr_te")
        pe = b.number_input("Pump Efficiency", min_value=0.1, max_value=1.0, value=0.75, step=0.01, key="tr_pe")
        he = cc.number_input("Heater Effectiveness", min_value=0.1, max_value=1.0, value=0.90, step=0.01)
        with st.expander("Pressure drops"):
            a, b = st.columns(2)
            with a:
                l1 = drop("Condenser, CO₂ side", 0.0, "tr_l1")
                l3 = drop("Heater, flue gas side", 0.0, "tr_l3")
            with b:
                l2 = drop("Condenser, cooling water side", 0.0, "tr_l2")
                l4 = drop("Heater, CO₂ side", 0.0, "tr_l4")
        go = st.form_submit_button("Run Transcritical CO₂ Cycle", type="primary")
    if go and idaes_ready():
        run("res_trans", run_trans, t_in + K, p_in * BAR, mass_flow, flue_string, te, pe, he, (l1, l2, l3, l4))
    show_results("res_trans")
