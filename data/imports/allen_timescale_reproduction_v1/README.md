# Historical Allen V1 response-timescale reproduction

**Raw reproduction gate: PASS**

- Raw sessions: 4 (226 historical VISp fits).
- Exact spike counts: 226/226 units.
- Maximum absolute timescale difference: 470.962 ms.
- Maximum absolute valid-session mean difference: 0.0825677 ms.
- Validity-gate disagreements: 0.
- Full frozen-table support: 5,370 fits, 1,615 valid units, 54 sessions.

The raw gate requires exact spike counts, identical historical validity decisions,
and <0.1-ms drift in every valid-session mean. Per-unit rejected fits are not
constrained because bounded nonlinear solutions are unstable at the fit limits.
