import numpy as np
from scipy.integrate import solve_ivp
import itertools

"""STAGE 1 - BAFFLED FLASK : SWEEP OF n (rpm) AND V_max (flask size)"""

"""Constants"""
mu_max = 0.45   # maximum growth rate [1/h]
K_S    = 0.01   # glycerol half-saturation [g/L]
Y_XS   = 0.45   # biomass yield on glycerol [g DCW/g glycerol]
Y_XO   = 0.71   # biomass yield on oxygen [g DCW/g O2]
m_O2   = 0.057  # oxygen maintenance [g O2/(g DCW*h)]
O_sat  = 0.007  # saturation dissolved oxygen at 37C [g O2/L]
S_MIN  = 1e-6   # glycerol concentration treated as "depleted" [g/L]
                # (Monod kinetics only let S approach 0 asymptotically, so an exact S == 0 crossing may never be detected
                # numerically;a  small epsilon makes the depletion event actually triggerable)

"""Gaussian kLa correlation (Meier et al. 2013, 10% filling)"""
x_0 = 223.56
y_0 = 259.62
a   = 386.72
b   = 63.13
c   = 247.92

def calc_kLa(n, V_max):
    """Oxygen transfer coefficient from Gaussian correlation [1/h]"""
    return a * np.exp(-0.5 * (((n - x_0) / b) ** 2 + ((V_max - y_0) / c) ** 2))

def calculate_rates(X, S):
    """Rates Determination"""
    S = max(S, 0)
    mu   = mu_max * (S / (S + K_S))
    q_O2 = mu / Y_XO + m_O2
    OUR  = q_O2 * X
    return mu, q_O2, OUR


def flask(t, y):
    """ODE SYSTEM (Biomass and Glycerol Balance)"""
    X, S = y
    mu, q_O2, OUR = calculate_rates(X, S)
    dXdt = mu * X
    dSdt = -(1 / Y_XS) * mu * X
    return (dXdt, dSdt)

"""Run one (n, V_max) combination"""
def run_case(n, V_max):    #For a particular n and V_max run my fermentation simulation
    kLa = calc_kLa(n, V_max)
    OTR_MAX = kLa * O_sat
    OTR_thresh = 0.8 * OTR_MAX   # the 80%-of-capacity O2 ceiling itself

    def oxygen_capacity_limit(t, y):          #Event detector -> Oxygen Threshold
        X, S = y
        mu, q_O2, OUR = calculate_rates(X, S)
        return OTR_thresh - OUR
    oxygen_capacity_limit.terminal = True
    oxygen_capacity_limit.direction = -1

    def substrate_depletion_limit(t, y):      #Event detector -> Glycerol Threshold
        X, S = y
        return S - S_MIN
    substrate_depletion_limit.terminal = True
    substrate_depletion_limit.direction = -1

    S_0 = 8.0     # Initial Glycerol Concentration

    X_0 = 0.02    # Inoculum Concentration


    culture_time = (0, 13)

    # Oxygen uptake rate demanded by the starting inoculum, before any
    # integration happens -> lets us compare directly against 0.8*OTR_max to
    # see whether this (n, V_max) pair could ever support aerobic growth.
    _, _, OUR_t0 = calculate_rates(X_0, S_0)

    # solve_ivp watches BOTH event functions continuously, at every internal
    # step, for the whole run -> whichever crosses zero first stops it.
    # The one thing a crossing-based event can never catch is a constraint
    # that is already violated at t=0 (there is no sign CHANGE to detect),
    # so check the oxygen condition explicitly at t=0 before integrating.
    O2_margin_t0 = OTR_thresh - OUR_t0
    o2_short_from_start = O2_margin_t0 < 0   # True -> flask can't meet even the
                                              # starting O2 demand; growth never
                                              # really had adequate aeration

    if o2_short_from_start:
        # The flask can't supply enough O2 even for the tiny starting
        # inoculum. There is no "crossing" for solve_ivp to find here (the
        # constraint is already broken at t=0), and letting the ODE run
        # anyway would just report glycerol depletion as the cause later,
        # burying the real problem. So: don't integrate, report this
        # (n, V_max) pair as oxygen-limited from the very start, t_end = 0.
        X_end, S_end, t_end = X_0, S_0, 0.0
        o2_limited, s_limited = True, False
        calc_thresh_O2 = O2_margin_t0
    else:
        lysis = solve_ivp(flask, culture_time, [X_0, S_0], method='Radau',
                        events=[oxygen_capacity_limit, substrate_depletion_limit],
                        rtol=1e-8, atol=1e-10)

        X_end = lysis.y[0][-1]     # biomass at termination [g/L]
        S_end = lysis.y[1][-1]     # glycerol at termination [g/L]
        t_end = lysis.t[-1]        # time at termination [h]

        o2_limited = lysis.t_events[0].size > 0   # O2 threshold was crossed during the run
        s_limited  = lysis.t_events[1].size > 0   # glycerol ran out during the run


        calc_thresh_O2 = oxygen_capacity_limit(t_end, [X_end, S_end])

    V_liq = 0.10 * V_max                       # 10% fill, in litres
    CELL_mass = (X_end - X_0) * V_liq          # NET biomass produced [g]
                                               # (X_end - X_0), not X_end alone,


    return (n, V_max, kLa, OTR_thresh, OUR_t0, t_end, S_end, X_end, CELL_mass,
            o2_limited, s_limited, calc_thresh_O2)

"""SWEEP over all combinations of n and V_max""" # Stucturing the combinations between n and V_max.

n_values    = np.arange(50, 301, 10)          # 50 ... 300 rpm
Vmax_values = np.arange(0.500, 1.001, 1)    # 50 ... 500 Lt


# printing instructions
print(f"{'n[rpm]':>7} {'V[mL]':>6} {'kLa[1/h]':>9} {'0.8*OTR_max':>12} {'OUR(t=0)':>9} "
      f"{'t_end[h]':>9} {'S_end[g/L]':>11} {'X_end[g/L]':>11} {'CELL_mass[g]':>13} "
      f"{'O2_lim':>7} {'S_lim':>6} {'calc_thresh_O2':>15}")
print("-" * 132)



best = {'CELL_mass': -np.inf}


#screening the combinations
for n, V_max in itertools.product(n_values, Vmax_values):
    (n, V_max, kLa, OTR_thresh, OUR_t0, t_end, S_end, X_end, CELL_mass,
     o2_limited, s_limited, calc_thresh_O2) = run_case(n, V_max)


    print(f"{n:>7} {V_max:>6} {kLa:>9.3f} {OTR_thresh:>12.5f} {OUR_t0:>9.5f} "
          f"{t_end:>9.2f} {S_end:>11.2e} {X_end:>11.3f} {CELL_mass:>13.4f} "
          f"{str(o2_limited):>7} {str(s_limited):>6} {calc_thresh_O2:>15.6f}")

    if CELL_mass > best['CELL_mass']:
        best = {'n': n, 'V_max': V_max, 'kLa': kLa, 'OTR_thresh': OTR_thresh,
                'OUR_t0': OUR_t0, 't_end': t_end, 'S_end': S_end, 'X_end': X_end,
                'CELL_mass': CELL_mass, 'o2_limited': o2_limited,
                's_limited': s_limited, 'calc_thresh_O2': calc_thresh_O2}

"""Summary of the best combination"""
print("\n" + "=" * 132)
print(f"For n = {best['n']} rpm and V_max = {best['V_max']} Lt "
      f"we have maximum CELL_mass = {best['CELL_mass']:.4f} g "
      f"(X_end = {best['X_end']:.4f} g/L)")
print(f"  (kLa = {best['kLa']:.3f} /h, 0.8*OTR_max = {best['OTR_thresh']:.5f} g O2/L/h, "
      f"OUR(t=0) = {best['OUR_t0']:.5f} g O2/L/h, t_end = {best['t_end']:.2f} h, "
      f"S_end = {best['S_end']:.2e} g/L, "
      f"O2_lim = {best['o2_limited']}, S_lim = {best['s_limited']}, "
      f"calc_thresh_O2 = {best['calc_thresh_O2']:.6f})")

