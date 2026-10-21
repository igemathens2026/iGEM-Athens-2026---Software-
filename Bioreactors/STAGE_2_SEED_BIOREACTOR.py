import numpy as np
from scipy.integrate import solve_ivp
import itertools

"""STAGE 2 - SEED BIOREACTOR """

"""Constants"""
mu_max = 0.40   # maximum growth rate [1/h]
K_S    = 0.01   # glycerol half-saturation [g/L]

Y_XS   = 0.44   # Biomass yield on glycerol [g DCW/g glycerol]
Y_XO   = 0.72   # Biomass yield on oxygen [g DCW/g O2]

m_O2   = 0.057                 # oxygen maintenance [g O2/(g DCW*h)]
Y_OS   = 3.5 * 32.00 / 92.09   # express the oxygen consumed per unit of substrate consumed
                               # C3H8O3 + 3.5 O2 -> 3 CO2 + 4 H2O
                               # Stoichiometric parameter
m_S    = m_O2 / Y_OS           # substrate maintenance [g glycerol/(g DCW*h)]


C_star = 0.0160                # saturation dissolved O2 sparging 50% O2 at 37 C,
C      = 0.0021                # C  dissolved O2 held by the controller = 30% of AIR


O_sat  = C_star-C             # DRIVING FORCE (C_star - C), not a saturation value [g O2/L]
                              # Transfer is driven by the DIFFERENCE between what the liquid could hold and what it actually holds.

S_HARVEST = 5.0               # residual glycerol required at harvest [g/L]
                              # The culture is transferred here, NOT run to exhaustion:
                              # carbon-starved cells carry a lag into the next reactor.

V_max = 30.0            # seed bioreactor total volume [L]
V_liq = 0.50 * V_max    # working volume = actual culture volume [L] = 15 L
X_0   = 0.115           # Inoculum Concentration [g DCW/L]
culture_time = (0, 50)  # maximum cultivation window [h]


def calculate_rates(X, S):
    """Rates Determination"""
    S = max(S, 0)
    mu   = mu_max * (S / (S + K_S))
    q_S  = mu / Y_XS + m_S           # Herbert - Pirt for Substrate : growth-associated and maintenance
    q_O2 = mu / Y_XO + m_O2          # Herbert - Pirt for Oxygen : growth-associated and maintenance
    OUR  = q_O2 * X                  # volumetric O2 uptake [g O2/(L*h)]
    return mu, q_S, q_O2, OUR


def seed_bioreactor(t, y):
    """ODE SYSTEM (Biomass and Glycerol Balance)"""
    X, S = y
    mu, q_S, q_O2, OUR = calculate_rates(X, S)
    dXdt = mu * X
    dSdt = -q_S * X                  # = -(mu/Y_XS + m_S)*X
    return (dXdt, dSdt)


"""Run one (kLa, S_0) combination"""
def run_case(kLa, S_0):
    OTR_MAX    = kLa * O_sat
    OTR_thresh = 0.8 * OTR_MAX   # DESIGN THRESHOLD


    def oxygen_capacity_limit(t, y):          # Event detector -> Oxygen Threshold
        X, S = y
        mu, q_S, q_O2, OUR = calculate_rates(X, S)
        return OTR_thresh - OUR
    oxygen_capacity_limit.terminal = True
    oxygen_capacity_limit.direction = -1

    def harvest_point(t, y):                  # Event detector -> Harvest
        X, S = y
        return S - S_HARVEST
    harvest_point.terminal = True
    harvest_point.direction = -1

    # Oxygen uptake rate demanded by the starting inoculum, before any
    # integration happens -> lets us compare directly against 0.8*OTR_max to
    # see whether this (kLa, S_0) pair could ever support aerobic growth.
    _, _, _, OUR_t0 = calculate_rates(X_0, S_0)

    # A crossing-based event can never catch a constraint that is already
    # violated at t=0 (there is no sign CHANGE to detect), so check the
    # oxygen condition explicitly before integrating.
    O2_margin_t0 = OTR_thresh - OUR_t0
    o2_short_from_start = O2_margin_t0 < 0

    if o2_short_from_start:
        # The reactor cannot supply enough O2 even for the starting inoculum.
        # Integrating anyway would report glycerol depletion as the cause later
        # and bury the real problem, so report t_end = 0 instead.
        X_end, S_end, t_end = X_0, S_0, 0.0
        o2_limited, harvested = True, False
        calc_thresh_O2 = O2_margin_t0
    else:
        sol = solve_ivp(seed_bioreactor, culture_time, [X_0, S_0], method='Radau',
                        events=[oxygen_capacity_limit, harvest_point],
                        rtol=1e-8, atol=1e-10)

# solving initial value problem

        X_end = sol.y[0][-1]     # biomass at termination [g/L]
        S_end = sol.y[1][-1]     # glycerol at termination [g/L]
        t_end = sol.t[-1]        # time at termination [h]

        o2_limited = sol.t_events[0].size > 0   # O2 threshold crossed during the run
        harvested  = sol.t_events[1].size > 0   # reached the harvest spec

        # Value of the O2-threshold event function at the stopping point.
        # ~0 when O2 was the active/binding constraint (that is literally the
        # root solve_ivp located). Positive on a clean harvest.
        calc_thresh_O2 = oxygen_capacity_limit(t_end, [X_end, S_end])

    CELL_mass = (X_end - X_0) * V_liq          # NET biomass produced [g]
                                               # (X_end - X_0), not X_end alone,


    return (kLa, S_0, OTR_thresh, OUR_t0, t_end, S_end, X_end, CELL_mass,
            o2_limited, harvested, calc_thresh_O2)


#Substrate Screening
"""SWEEP over all combinations of kLa and S_0"""
kLa_values = np.array([675])                             # specific values, alter as needed
S_0_values = np.arange(10,36,1)    # 10 ... 35 g/L glycerol loading


#printing instructions
print(f"{'kLa[1/h]':>9} {'S_0[g/L]':>9} {'0.8*OTR_max':>12} {'OUR(t=0)':>9} "
      f"{'t_end[h]':>9} {'S_end[g/L]':>11} {'X_end[g/L]':>11} {'CELL_mass[g]':>13} "
      f"{'O2_lim':>7} {'harvest':>8} {'calc_thresh_O2':>15}")
print("-" * 122)


# Rows that ended on the oxygen ceiling are excluded: there the culture never
# reached S_HARVEST, so a residual-glycerol trigger would never have fired.
best = {'CELL_mass': -np.inf}

#running the run_case function
for kLa, S_0 in itertools.product(kLa_values, S_0_values):
    (kLa, S_0, OTR_thresh, OUR_t0, t_end, S_end, X_end, CELL_mass,
     o2_limited, harvested, calc_thresh_O2) = run_case(kLa, S_0)

    print(f"{kLa:>9.2f} {S_0:>9.1f} {OTR_thresh:>12.5f} {OUR_t0:>9.5f} "
          f"{t_end:>9.2f} {S_end:>11.2e} {X_end:>11.3f} {CELL_mass:>13.4f} "
          f"{str(o2_limited):>7} {str(harvested):>8} {calc_thresh_O2:>15.6f}")

#It Saves the greatest CELL_mass
    if harvested and CELL_mass > best['CELL_mass']:
        best = {'kLa': kLa, 'S_0': S_0, 'OTR_thresh': OTR_thresh,
                'OUR_t0': OUR_t0, 't_end': t_end, 'S_end': S_end, 'X_end': X_end,
                'CELL_mass': CELL_mass, 'o2_limited': o2_limited,
                'harvested': harvested, 'calc_thresh_O2': calc_thresh_O2}

"""Summary of the best combination"""
print("\n" + "=" * 122)
if best['CELL_mass'] == -np.inf:
    print("No charge reached the harvest spec - every row hit the oxygen ceiling first.")
    raise SystemExit
print(f"For kLa = {best['kLa']:.2f} /h and S_0 = {best['S_0']:.1f} g/L "
      f"we have maximum CELL_mass = {best['CELL_mass']:.4f} g "
      f"(X_end = {best['X_end']:.4f} g/L)")
print(f"  (0.8*OTR_max = {best['OTR_thresh']:.5f} g O2/L/h, "
      f"OUR(t=0) = {best['OUR_t0']:.5f} g O2/L/h, t_end = {best['t_end']:.2f} h, "
      f"S_end = {best['S_end']:.2f} g/L, "
      f"O2_lim = {best['o2_limited']}, harvest = {best['harvested']}, "
      f"calc_thresh_O2 = {best['calc_thresh_O2']:.6f})")

