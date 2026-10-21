import numpy as np
from scipy.integrate import solve_ivp

"""STAGE 3 - FED-BATCH FERMENTER """

"""Phase 1 is closed form: the exponential feed imposes mu = MU_SET, so q_O2 and
q_S are constant and every state has an analytical expression.
Phase 2 is integrated: q_p decays with time, so the rates are not constant.
"""

# ---------------------------------------------------------------- parameters
Y_XS = 0.44                   # Biomass yield on glycerol [g DCW/g glycerol]
Y_XO   = 0.72                 # Biomass yield on oxygen [g DCW/g O2]

Y_OS = 3.5 * 32.0 / 92.09     # express the oxygen consumed per unit of substrate consumed
                              # C3H8O3 + 3.5 O2 -> 3 CO2 + 4 H2O
                              # Stoichiometric parameter

m_O2 = 0.057                   # oxygen maintenance [g O2/(g DCW*h)]
m_S  = m_O2 / Y_OS             # substrate maintenance [g glycerol/(g DCW*h)]

MU_SET   = 0.055               # imposed by the feed profile [1/h]
MU_RATIO = 0.10                # post-induction mu as a fraction of MU_SET due to burden
S_F      = 711.0               # glycerol in the feed [g/L]

kLa = 275                      # [1/h]
C_star = 0.0334                # saturation dissolved O2 sparging 50% O2 at 37 C,
C      = 0.0021                # C  dissolved O2 held by the controller = 30% of AIR
OTR_FRACTION = 0.8             # design threshold on the transfer available at C_L

V_TOTAL = 150
FILL = 0.70
V_FILL_MAX = FILL * V_TOTAL          # 105 L

X_SEED = 12.0                        # Inoculum Concentration
V_SEED = 15.0
V_1 = 75.0                           # Volume at the start of the fed-batch
X_1 = X_SEED * V_SEED / V_1          # Ostwald's Dilution Law

BETA    = np.exp(-2.7302) / 1000.0   # q_p at induction [g P/(g DCW*h)]  [WET LAB - exp. data]
K_DECAY = 0.12                       # decay of q_p [1/h] [WET LAB - exp. data]
Y_P     = 0.25                       # assumed value
GAMMA_P = 1.526                      # depends on assumption
Y_OP    = Y_OS / Y_P - GAMMA_P       # depends on assumption
TAU_MAX = 24.0                       # production window [h]

# ------------------------------------------------------------------- phase 1 - Production
q_O2 = MU_SET / Y_XO + m_O2          # Herbert - Pirt for Substrate : growth-associated and maintenance
q_S  = MU_SET / Y_XS + m_S           # Herbert - Pirt for Oxygen : growth-associated and maintenance

OTR_max    = kLa * (C_star - C)      # oxygen transfer availability
OTR_thresh = OTR_FRACTION * OTR_max  # carbon-limited growth valid below this

b = q_S * X_1 * V_1 / (MU_SET * S_F)          # V(t) = c + b*exp(MU_SET*t)
c = V_1 - b

X_ind = OTR_thresh / q_O2                     # oxygen-limited density
denom = X_1 * V_1 - X_ind * b
u     = X_ind * c / denom                     # = exp(MU_SET * t_ind)
t_ind = np.log(u) / MU_SET
V_ind = c + b * u

# The closed form returns an answer even for a reactor that cannot run,
# so the two physical constraints are checked explicitly.
if q_O2 * X_1 > OTR_thresh:
    raise SystemExit("FAIL: the inoculum alone exceeds the oxygen threshold.")
if denom <= 0:
    raise SystemExit("FAIL: X_ind unreachable - the feed dilutes as fast as "
                     "the cells grow.")
if V_ind > V_FILL_MAX:
    raise SystemExit(f"FAIL: vessel fills at {V_FILL_MAX:.0f} L before the "
                     f"oxygen threshold (needs {V_ind:.1f} L).")

print("=" * 70)
print("PHASE 1 - fed-batch to induction")
print("=" * 70)
print(f"  mu_set : {MU_SET:.3f} 1/h    q_S : {q_S:.4f}    q_O2 : {q_O2:.4f}")
print(f"  OTR available : {OTR_max:.3f}, threshold : {OTR_thresh:.3f} g O2/(L*h)")
print(f"  t_ind  :    {t_ind:8.2f} h")
print(f"  X_ind  :   {X_ind:8.3f} g DCW/L")
print(f"  V_ind  :    {V_ind:8.2f} L      (fill limit {V_FILL_MAX:.0f} L)")
print(f"  biomass :   {X_ind*V_ind - X_1*V_1:8.0f} g net")
print(f"  glycerol :  {(V_ind - V_1)*S_F:8.0f} g fed")

# ------------------------------------------------------------------- phase 2
mu = MU_RATIO * MU_SET               # imposed; cells divide slowly under burden

def phase2(tau, y):
    """States: volume, biomass, product. tau = time since induction."""
    V, X, P = y
    q_p = BETA * np.exp(-K_DECAY * tau)             # via [WET LAB - exp. data]
    q_S2 = mu / Y_XS + m_S + q_p / Y_P              # Herbert - Pirt for Substrate : growth-associated, maintenance, protein production
    F    = q_S2 * X * V / S_F                       # feed matches consumption, so S is not a state
    D    = F / V                                    # Dillution Parameter
    return (F, mu * X - D * X, q_p * X - D * P)

def fill_limit(tau, y):
    return V_FILL_MAX - y[0]
fill_limit.terminal, fill_limit.direction = True, -1

#Solution of Initial Value Problem
sol = solve_ivp(phase2, (0.0, TAU_MAX), [V_ind, X_ind, 0.0], method='Radau',
                events=[fill_limit], rtol=1e-8, atol=1e-10, max_step=0.25)

#logging the values of acetyl - muramidase estimation at the end of simulation
tau_end = sol.t[-1]
V_end, X_end, P_end = sol.y[:, -1]
q_p_end = BETA * np.exp(-K_DECAY * tau_end)
OUR_end = (mu / Y_XO + m_O2 + q_p_end * Y_OP) * X_end


#Printing instructions
print("\n" + "=" * 70)
print("PHASE 2 - production after induction")
print("=" * 70)
print(f"  mu {mu:.4f} 1/h   beta {1000*BETA:.5f} mg P/(gDCW*h)   "
      f"q_p half-life {np.log(2)/K_DECAY:.1f} h")
print(f"  stopped by: {'vessel fill limit' if sol.t_events[0].size else 'production window'}")
print(f"  tau_end    {tau_end:8.2f} h      (process clock {t_ind+tau_end:.2f} h)")
print(f"  X          {X_end:8.3f} g DCW/L  (from {X_ind:.3f})")
print(f"  V          {V_end:8.2f} L        ({V_FILL_MAX-V_end:.1f} L spare)")
print(f"  titre      {1000*P_end:8.2f} mg/L")
print(f"  ENZYME     {P_end*V_end:8.3f} g total")
print(f"  glycerol   {(V_end - V_ind)*S_F:8.0f} g fed")
print(f"  OUR        {OUR_end:8.3f} g O2/(L*h) of {OTR_max:.3f} available "
      f"({100*OUR_end/OTR_max:.0f} %)   [diagnostic]")
print("=" * 70)