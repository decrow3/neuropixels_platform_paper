# Allen BO 1.1 non-center feature registration

Transforms were fitted without SF, TF, RF-center consensus, or modulation index.
The scalar fields are log2 RF area, dorsal–ventral CCF position, probe-horizontal
position, RF response time-to-peak, and flash first-spike latency. Feature weights
follow the independent spatial-gradient audit; the two latency fields are weak
regularizers.

Translation-only and tightly bounded similarity models were both fitted. Similarity
is selected only when its median regularized non-center objective improves by at least
0.020.

Selected model: **similarity**. Median objective gain over translation: +0.026.

The selected transform is an exploratory landmark-based registration. SF/TF were not
used for fitting or model selection and can therefore be used as independent outcomes.
