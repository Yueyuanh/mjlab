# Actuator response tests

Run limit-aware actuator response tests against the actuator and robot model from
any registered task:

~~~bash
uv run response Mjlab-Tracking-Flat-Pikachu-Walk
~~~

The runner builds a one-robot MJWarp simulation from the task's robot entity. It
removes the floating-base joint, fixes the root at --suspend-height-m, omits
terrain, and disables all geom collisions. This makes ground contact impossible
while preserving gravity loads, link dynamics, joint damping/friction, armature,
PD gains, effort saturation, and configured actuator command delay.

Every selected hinge joint is reset before each case. Positive and negative
steps are sized from the available travel on each side of the nominal pose.
Sine amplitudes use the smaller symmetric travel. Both protocols reserve
--limit-margin-fraction at the hard stops.

Useful examples:

~~~bash
# One joint, both step directions.
uv run response Mjlab-Tracking-Flat-Pikachu-Walk \
  --mode step --joint-pattern left_knee_joint

# Sine frequency response at custom frequencies.
uv run response Mjlab-Tracking-Flat-Pikachu-Walk \
  --mode sine --sine-frequencies-hz 0.5,1,2,4,8,12

# Run on CUDA instead of the one-world CPU default.
uv run response Mjlab-Tracking-Flat-Pikachu-Walk --device cuda:0
~~~

Results are written below logs/response/<task>/<UTC timestamp>/:

- response.csv: raw long-form samples with commands, position, velocity,
  actuator torque, power, saturation/utilization, limits, motor metadata, PD
  gains, armature, home-pose inertia, gearbox data, and a zero-contact check;
- summary.csv: rise/settling time, overshoot, steady-state error, error
  integrals, sine gain/phase/THD, measured bandwidth, and motor sizing demand;
- metadata.json: run configuration, suspension guarantee, per-joint model,
  and output-file manifest.

Start the dashboard service from the repository root:

~~~bash
uv run python scripts/plot_logs/serve.py
~~~

Open http://localhost:8000/scripts/plot_logs/response_csv_dashboard.html. It
discovers every logs/response/**/response.csv and loads the newest run
automatically. The page shows raw step and per-frequency sine time histories,
Bode/THD fits, torque-speed demand, LaTeX model derivations, and explicit NaN
diagnostics. Unknown motor catalogue values remain unavailable by design; do
