"""
Exergy Sankey diagrams for the five waste heat recovery cycles.

Each cycle's calc() stores its exergy results in `cycle.exergy` (one line
added before calc()'s return). build_flows() turns those into plain,
JSON-safe diagram data -- so the IDAES cycles can send it back from
co2_runner.py -- and figure() draws one diagram with Plotly.

Two diagrams per cycle (LiBr has only the first):
  1. Overall exergy balance: exergy input(s) -> cycle -> useful output,
     exergy destruction in each component, exergy lost to cooling water.
     The useful output divided by the input(s) is the cycle's exergy
     efficiency, as defined in its .py file:
       steam, orc : net power / flue gas exergy
       trans      : (turbine work - pump work) / flue gas exergy
       super      : net power / exergy added in the flue gas heater
       libr       : cooling exergy / flue gas exergy
  2. Component exergy flow: exergy passed from component to component
     around the cycle (the second diagram in steam.py / super.py).

All values are exergy rates in kW.
"""

USEFUL = "#2e7d32"      # useful output
LOSS = "#1e88e5"        # exergy carried away (cooling water, bottoming cycle)
DESTROYED = "#e53935"   # exergy destruction
INPUT = "#fb8c00"       # exergy input
CYCLE = "#546e7a"       # cycle / component nodes
INTERNAL = "#90a4ae"    # exergy circulating inside the cycle


def _f(x):
    return float(x) if x is not None else 0.0


def _kw(value):
    return f"{value:,.1f} kW" if value >= 1 else f"{value:.3f} kW"


class _Builder:
    def __init__(self):
        self.labels, self.colors, self.pos = [], [], []
        self.sources, self.targets, self.values, self.link_colors = [], [], [], []

    def node(self, label, color=CYCLE, pos=None):
        """pos=(x, y) pins a component node (0-1, left-to-right / top-to-bottom)."""
        self.labels.append(label)
        self.colors.append(color)
        self.pos.append(pos)
        return len(self.labels) - 1

    def connect(self, src, dst, value, color=INTERNAL):
        """Link between two existing nodes. Zero or negative flows are left
        out (Sankey links must be positive)."""
        value = _f(value)
        if value <= 1e-6:
            return
        self.sources.append(src)
        self.targets.append(dst)
        self.values.append(value)
        self.link_colors.append(color)

    def link(self, src, label, value, color):
        """New end node `label` fed from `src`, with its value in the label."""
        value = _f(value)
        if value <= 1e-6:
            return
        self.connect(src, self.node(f"{label}: {_kw(value)}", color), value, color)

    def feed(self, label, value, dst, color=INPUT):
        """New start node `label` feeding `dst`, with its value in the label."""
        value = _f(value)
        if value <= 1e-6:
            return
        self.connect(self.node(f"{label}: {_kw(value)}", color), dst, value, color)

    def result(self):
        out = dict(labels=self.labels, node_colors=self.colors, sources=self.sources,
                   targets=self.targets, values=self.values, link_colors=self.link_colors)
        if any(self.pos):
            # Pinned layout (component diagrams): inputs in a column on the
            # left, end nodes (output, destruction, losses) in a column on
            # the right, components where they were pinned.
            used = set(self.sources) | set(self.targets)
            starts = [i for i in range(len(self.labels)) if self.pos[i] is None and i in used
                      and i not in self.targets]
            ends = [i for i in range(len(self.labels)) if self.pos[i] is None and i in used
                    and i not in starts]
            # order end nodes by the height of the component feeding them,
            # so their links cross as little as possible
            feeder = {t: s for s, t in zip(self.sources, self.targets)}
            ends.sort(key=lambda i: (self.pos[feeder[i]] or (0, 0))[::-1])
            xs, ys = [0.5] * len(self.labels), [0.5] * len(self.labels)
            for i, p in enumerate(self.pos):
                if p:
                    xs[i], ys[i] = p
            for col, x in ((starts, 0.001), (ends, 0.999)):
                for k, i in enumerate(col):
                    xs[i], ys[i] = x, 0.06 + 0.88 * (k + 0.5) / len(col)
            out["x"], out["y"] = xs, ys
        return out


def _condenser_split(hx):
    """Splits the exergy the working fluid gives up in a condenser/cooler
    (hot_change) into (destroyed, carried away by cooling water). When the
    cooling water enters below the 25 °C reference temperature its exergy
    change is negative, so all of it counts as destruction -- this keeps the
    diagrams balanced and free of negative flows."""
    released = _f(hx["hot_change"])
    to_water = min(max(_f(hx["cold_change"]), 0.0), released)
    return released - to_water, to_water


# ---------------------------------------------------------------------------
# Steam, ORC, transcritical CO2: heater -> turbine -> condenser -> pump
# ---------------------------------------------------------------------------

def _rankine_overall(ex, cycle_name):
    """Exergy input = flue gas exergy only; the pump is driven by the turbine
    inside the cycle, so only the net power leaves as useful output."""
    b = _Builder()
    cyc = b.node(cycle_name)
    turbine_w, pump_w = _f(ex["turbine"]["work"]), _f(ex["pump"]["work"])
    b.feed("Exergy from flue gas", ex["heater"]["hot_change"], cyc)
    b.link(cyc, "Net power output", turbine_w - pump_w, USEFUL)
    b.link(cyc, "Destruction – heater", ex["heater"]["destroyed"], DESTROYED)
    b.link(cyc, "Destruction – turbine", ex["turbine"]["destroyed"], DESTROYED)
    b.link(cyc, "Destruction – pump", ex["pump"]["destroyed"], DESTROYED)
    destroyed, to_water = _condenser_split(ex["condenser"])
    b.link(cyc, "Destruction – condenser", destroyed, DESTROYED)
    b.link(cyc, "Lost to cooling water", to_water, LOSS)
    return b.result()


def _rankine_components(ex):
    """Same flows as exergy_flows2 in steam.py."""
    heater, turbine, pump, cond = ex["heater"], ex["turbine"], ex["pump"], ex["condenser"]
    b = _Builder()
    n_heater, n_turbine = b.node("Heater", pos=(0.18, 0.45)), b.node("Turbine", pos=(0.36, 0.25))
    n_cond, n_pump = b.node("Condenser", pos=(0.52, 0.55)), b.node("Pump", pos=(0.64, 0.68))
    b.feed("Exergy from flue gas", heater["hot_change"], n_heater)
    b.connect(n_pump, n_heater, pump["out"])
    b.connect(n_heater, n_turbine, turbine["in"])
    b.link(n_heater, "Destruction – heater", heater["destroyed"], DESTROYED)
    b.link(n_turbine, "Net power output", _f(turbine["work"]) - _f(pump["work"]), USEFUL)
    b.connect(n_turbine, n_pump, pump["work"])                 # shaft work to the pump
    b.link(n_turbine, "Destruction – turbine", turbine["destroyed"], DESTROYED)
    b.connect(n_turbine, n_cond, turbine["out"])
    b.connect(n_cond, n_pump, cond["hot_out"])
    destroyed, to_water = _condenser_split(cond)
    b.link(n_cond, "Destruction – condenser", destroyed, DESTROYED)
    b.link(n_cond, "Lost to cooling water", to_water, LOSS)
    b.link(n_pump, "Destruction – pump", pump["destroyed"], DESTROYED)
    return b.result()


# ---------------------------------------------------------------------------
# Recompression sCO2 cycle
# ---------------------------------------------------------------------------

def _net_super(ex):
    return (_f(ex["turbine"]["work"]) - _f(ex["main_compressor"]["work"])
            - _f(ex["bypass_compressor"]["work"]))


def _super_overall(ex):
    """Same scheme as exergy_flows2 in super.py."""
    b = _Builder()
    cyc = b.node("sCO₂ cycle")
    b.feed("Exergy added in flue gas heater", ex["heater"], cyc)
    b.link(cyc, "Net power output", _net_super(ex), USEFUL)
    b.link(cyc, "To bottoming cycle", ex["bt"]["cold_change"], LOSS)
    b.link(cyc, "Destruction – turbine", ex["turbine"]["destroyed"], DESTROYED)
    b.link(cyc, "Destruction – HTR", ex["htr"]["destroyed"], DESTROYED)
    b.link(cyc, "Destruction – LTR", ex["ltr"]["destroyed"], DESTROYED)
    b.link(cyc, "Destruction – bottoming-cycle HX", ex["bt"]["destroyed"], DESTROYED)
    b.link(cyc, "Destruction – main compressor", ex["main_compressor"]["destroyed"], DESTROYED)
    b.link(cyc, "Destruction – bypass compressor", ex["bypass_compressor"]["destroyed"], DESTROYED)
    b.link(cyc, "Destruction – mixer", ex["mixer"]["destroyed"], DESTROYED)
    destroyed, to_water = _condenser_split(ex["cooler"])
    b.link(cyc, "Destruction – cooler", destroyed, DESTROYED)
    b.link(cyc, "Lost to cooling water", to_water, LOSS)
    return b.result()


def _super_components(ex):
    """Same flows as exergy_flows in super.py: heater -> turbine -> HTR ->
    LTR -> splitter -> (bottoming-cycle HX -> cooler -> main compressor ->
    LTR) and (bypass compressor) -> mixer -> HTR -> heater."""
    htr, ltr, bt, cooler = ex["htr"], ex["ltr"], ex["bt"], ex["cooler"]
    turbine, mc, bc, mixer = ex["turbine"], ex["main_compressor"], ex["bypass_compressor"], ex["mixer"]
    bypass = _f(ex["bypass"])
    b = _Builder()
    layout = {"Heater": (0.16, 0.55), "Turbine": (0.23, 0.4), "HTR": (0.3, 0.55), "LTR": (0.36, 0.35),
              "Splitter": (0.44, 0.25), "Bottoming-cycle HX": (0.52, 0.12), "Bypass compressor": (0.52, 0.5),
              "Cooler": (0.59, 0.08), "Main compressor": (0.66, 0.2), "Mixer": (0.62, 0.7)}
    n = {name: b.node(name, pos=xy) for name, xy in layout.items()}
    b.feed("Exergy added in flue gas heater", ex["heater"], n["Heater"])
    b.connect(n["HTR"], n["Heater"], htr["cold_out"])
    b.connect(n["Heater"], n["Turbine"], ex["heater_out"])
    # Turbine work: net output + shaft work to the two compressors
    b.link(n["Turbine"], "Net power output", _net_super(ex), USEFUL)
    b.connect(n["Turbine"], n["Main compressor"], mc["work"])
    b.connect(n["Turbine"], n["Bypass compressor"], bc["work"])
    b.link(n["Turbine"], "Destruction – turbine", turbine["destroyed"], DESTROYED)
    b.connect(n["Turbine"], n["HTR"], htr["hot_in"])
    # Recuperators
    b.connect(n["HTR"], n["LTR"], htr["hot_out"])
    b.link(n["HTR"], "Destruction – HTR", htr["destroyed"], DESTROYED)
    b.connect(n["LTR"], n["Splitter"], ltr["hot_out"])
    b.connect(n["LTR"], n["Mixer"], ltr["cold_out"])
    b.link(n["LTR"], "Destruction – LTR", ltr["destroyed"], DESTROYED)
    # Split: main flow to the bottoming-cycle HX, bypass to the bypass compressor
    b.connect(n["Splitter"], n["Bottoming-cycle HX"], _f(ltr["hot_out"]) * (1 - bypass))
    b.connect(n["Splitter"], n["Bypass compressor"], _f(ltr["hot_out"]) * bypass)
    b.link(n["Bottoming-cycle HX"], "To bottoming cycle", bt["cold_change"], LOSS)
    b.link(n["Bottoming-cycle HX"], "Destruction – bottoming-cycle HX", bt["destroyed"], DESTROYED)
    b.connect(n["Bottoming-cycle HX"], n["Cooler"], bt["hot_out"])
    destroyed, to_water = _condenser_split(cooler)
    b.link(n["Cooler"], "Destruction – cooler", destroyed, DESTROYED)
    b.link(n["Cooler"], "Lost to cooling water", to_water, LOSS)
    b.connect(n["Cooler"], n["Main compressor"], cooler["hot_out"])
    b.connect(n["Main compressor"], n["LTR"], mc["out"])
    b.link(n["Main compressor"], "Destruction – main compressor", mc["destroyed"], DESTROYED)
    b.connect(n["Bypass compressor"], n["Mixer"], bc["out"])
    b.link(n["Bypass compressor"], "Destruction – bypass compressor", bc["destroyed"], DESTROYED)
    b.connect(n["Mixer"], n["HTR"], mixer["ex"])
    b.link(n["Mixer"], "Destruction – mixer", mixer["destroyed"], DESTROYED)
    return b.result()


# ---------------------------------------------------------------------------
# LiBr absorption chiller
# ---------------------------------------------------------------------------

def _libr_overall(ex):
    """libr.py only evaluates component exergy for the pump and solution
    heat exchanger, so the desorber, absorber, condenser and evaporator are
    shown together as the balance."""
    b = _Builder()
    cyc = b.node("LiBr chiller")
    ex_in, pump_w = _f(ex["ex_in"]), _f(ex["pump"]["work"])
    b.feed("Exergy from flue gas", ex_in, cyc)
    b.feed("Pump work", pump_w, cyc)
    cooling = _f(ex["ex_out"])
    pump_d, shx_d = _f(ex["pump"]["destroyed"]), _f(ex["solution_hx"]["destroyed"])
    b.link(cyc, "Cooling exergy output", cooling, USEFUL)
    b.link(cyc, "Destruction – solution HX", shx_d, DESTROYED)
    b.link(cyc, "Destruction – pump", pump_d, DESTROYED)
    b.link(cyc, "Destruction & losses – desorber, absorber, condenser, evaporator",
           ex_in + pump_w - cooling - pump_d - shx_d, DESTROYED)
    return b.result()


# ---------------------------------------------------------------------------

FORMULAS = {
    "steam": "net power ÷ flue gas exergy",
    "orc": "net power ÷ flue gas exergy",
    "trans": "(turbine work − pump work) ÷ flue gas exergy",
    "super": "net power ÷ exergy added in the flue gas heater",
    "libr": "cooling exergy ÷ flue gas exergy",
}

CYCLE_NAMES = {"steam": "Steam Rankine cycle", "orc": "Organic Rankine cycle",
               "trans": "Transcritical CO₂ cycle"}


def build_flows(cycle_key, exergy):
    """Returns {"formula": str, "diagrams": [{"name": str, "flows": {...}}, ...]}."""
    if cycle_key == "super":
        diagrams = [("Overall exergy balance", _super_overall(exergy)),
                    ("Exergy flow between components", _super_components(exergy))]
    elif cycle_key == "libr":
        diagrams = [("Overall exergy balance", _libr_overall(exergy))]
    else:
        diagrams = [("Overall exergy balance",
                     _rankine_overall(exergy, CYCLE_NAMES[cycle_key])),
                    ("Exergy flow between components", _rankine_components(exergy))]
    return {"formula": FORMULAS[cycle_key],
            "diagrams": [{"name": name, "flows": flows} for name, flows in diagrams]}


def figure(flows, title):
    import plotly.graph_objects as go
    fig = go.Figure(go.Sankey(
        arrangement="snap",
        valueformat=",.1f",
        valuesuffix=" kW",
        node=dict(pad=18, thickness=18, line=dict(color="rgba(0,0,0,0.3)", width=0.5),
                  label=flows["labels"], color=flows["node_colors"],
                  **({"x": flows["x"], "y": flows["y"]} if "x" in flows else {})),
        link=dict(source=flows["sources"], target=flows["targets"], value=flows["values"],
                  color=[c + "55" for c in flows["link_colors"]]),
    ))
    fig.update_layout(title_text=title, font_size=12, height=560, margin=dict(l=10, r=10, t=50, b=50))
    return fig
