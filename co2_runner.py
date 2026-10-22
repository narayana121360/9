"""
Runs the IDAES-based CO2 cycles (super.py, trans.py) in a separate Python
process for app.py.

Why a separate process: IDAES looks for its compiled add-ons (Helmholtz EoS,
cubic-root functions, Ipopt) only once, when it is first imported. Running
each solve in a fresh process means add-ons installed while the app is open
are picked up immediately, without restarting Streamlit.

Usage (called by app.py):
    python co2_runner.py status            -> prints status JSON
    python co2_runner.py install           -> downloads the IDAES add-ons
    python co2_runner.py super  < args.json -> prints result JSON
    python co2_runner.py trans  < args.json -> prints result JSON
The JSON line is prefixed with RESULT_TAG so prints from the cycle code
don't get mixed up with it.
"""
import json
import sys

RESULT_TAG = "__CO2_RUNNER_RESULT__"


def _num(x):
    return None if x is None else float(x)


def status():
    try:
        import idaes
        from idaes.models.properties.general_helmholtz import helmholtz_available
        from idaes.models.properties.modular_properties.eos.ceos import cubic_roots_available
        from pyomo.environ import SolverFactory
    except ImportError as err:
        return {"ok": False, "message": f"IDAES is not installed in this Python environment ({err})."}
    missing = []
    if not helmholtz_available():
        missing.append("Helmholtz EoS functions (CO₂ / water properties)")
    if not cubic_roots_available():
        missing.append("cubic-root functions (Peng-Robinson flue gas)")
    if not SolverFactory("ipopt").available(exception_flag=False):
        missing.append("Ipopt solver")
    if missing:
        return {"ok": False,
                "message": "IDAES is installed, but its add-ons are missing: " + ", ".join(missing) + "."}
    return {"ok": True, "message": ""}


def _linux_distro_override():
    """IDAES only publishes add-on builds for Ubuntu, RHEL-family, macOS and
    Windows. Streamlit Community Cloud (and other Debian-based servers) is
    rejected as an 'Unsupported platform', so pick the Ubuntu build whose
    glibc is compatible: ubuntu2204 needs glibc >= 2.35, ubuntu2004 >= 2.31.
    Returns None where IDAES' own detection already works."""
    import platform
    if platform.system() != "Linux":
        return None
    try:
        with open("/etc/os-release") as f:
            info = dict(line.rstrip().split("=", 1) for line in f if "=" in line)
    except OSError:
        info = {}
    distro = info.get("ID", "").strip('"').lower()
    version = info.get("VERSION_ID", "").strip('"').replace(".", "")
    supported = {"ubuntu1804", "ubuntu2004", "ubuntu2204", "ubuntu2404", "rhel8", "el8", "centos8",
                 "rocky8", "almalinux8", "el9", "rhel7", "centos7", "debian9", "debian10"}
    if f"{distro}{version}" in supported:
        return None
    libc, ver = platform.libc_ver()
    try:
        major, minor = (int(x) for x in ver.split(".")[:2])
    except ValueError:
        major, minor = 2, 31
    return "ubuntu2204" if (major, minor) >= (2, 35) else "ubuntu2004"


def install():
    """Downloads the IDAES add-ons (same as `idaes get-extensions`), choosing a
    compatible build on Debian-based servers such as Streamlit Cloud."""
    import contextlib
    import io
    from idaes.commands.base import command_base
    argv = ["idaes", "get-extensions"]
    distro = _linux_distro_override()
    if distro:
        argv += ["--distro", distro]
    sys.argv = argv
    buf = io.StringIO()
    code = 0
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        try:
            command_base()
        except SystemExit as e:
            code = e.code or 0
    return {"ok": code == 0, "distro": distro or "auto", "log": buf.getvalue()[-3000:]}


def _sankey(key, cyc):
    """Exergy Sankey data for the app; a diagram problem never fails the run."""
    try:
        import sankey
        return sankey.build_flows(key, cyc.exergy)
    except Exception as err:
        return {"error": str(err)}


def run_super(a):
    import numpy as np
    import super as sup
    cyc = sup.sco2(a["approach"], a["p1"], a["bypass"], a["htr_eff"], a["ltr_eff"],
                   np.array(a["drop_hot"]), np.array(a["drop_cold"]),
                   list(a["flue"]), list(a["bc"]), list(a["cooler"]))
    (en, exg, net, f_out, turb_c, mc_c, bc_c, _, hx_costs, fluid_c, total) = cyc.calc()
    costs = {
        "Turbine": turb_c, "Main compressor": mc_c, "Bypass compressor": bc_c,
        "HTR": hx_costs["htr"]["cost"], "LTR": hx_costs["ltr"]["cost"],
        "Bottoming-cycle HX": hx_costs["bt"]["cost"], "Cooler": hx_costs["condenser"]["cost"],
        "Flue gas heater": hx_costs["heater"]["cost"], "Cooling tower": hx_costs["cooling_tower"]["cost"],
        "CO₂ charge": fluid_c,
    }
    return dict(energy=_num(en), exergy=_num(100 * exg), output=_num(net), heat_in=_num(cyc.heat_input),
                t_out=_num(f_out[0]), total=_num(total), costs={k: _num(v) for k, v in costs.items()},
                sankey=_sankey("super", cyc))


def run_trans(a):
    import trans
    cyc = trans.tco2(a["t_in"], a["p_in"], a["m_flue"], a["fs"], a["turb_eff"], a["pump_eff"],
                     a["heater_eff"], tuple(a["losses"]))
    en, exg, net, f_out, cost = cyc.calc()
    costs = {"Turbine": cost["turbine_cost"], "Pump": cost["pump_cost"],
             "Heater": cost["heater_cost"], "Condenser": cost["condenser_cost"],
             "Cooling tower": cost["cooling_tower_cost"], "CO₂ charge": cost["co2_fluid_cost"]}
    return dict(energy=_num(en), exergy=_num(exg), output=_num(net), heat_in=_num(cyc.hx_heater),
                t_out=_num(f_out[0]), total=_num(cost["total"]), costs={k: _num(v) for k, v in costs.items()},
                sankey=_sankey("trans", cyc))


if __name__ == "__main__":
    task = sys.argv[1]
    try:
        if task == "status":
            out = status()
        elif task == "install":
            out = install()
        else:
            args = json.loads(sys.stdin.read())
            out = {"super": run_super, "trans": run_trans}[task](args)
    except Exception as err:
        out = {"error": str(err)}
    print(RESULT_TAG + json.dumps(out), flush=True)
