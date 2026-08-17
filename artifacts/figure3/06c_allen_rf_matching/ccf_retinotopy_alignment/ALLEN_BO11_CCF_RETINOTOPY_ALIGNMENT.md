# Allen BO 1.1 V1 CCF→RF registration

V1 CCF coordinates were available in **23/31** simultaneous V1/HVA sessions.
Each session was predicted by a robust, session-balanced CCF→RF model trained only on the other sessions.
The selected RF-only model was **linear**; SF, TF, and HVA units were not used for fitting or selection.
The robust median V1 prediction residual supplies one bounded translation shared by that session's V1 and HVA maps.
Sessions without reconstructed CCF coordinates are marked unavailable and assigned identity only in the transform table; the four-row comparison excludes them from every row.

Median V1 RF RMSE before session translation: **12.27°**.
Median V1 RF RMSE after session translation: **10.81°**.
Translations reaching the ±15° bound: **5/23**.
