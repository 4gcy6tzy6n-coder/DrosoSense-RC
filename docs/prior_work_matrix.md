# DATA-29 — Prior-work evidence matrix (e-nose / food-quality × reservoir computing × lightweight deployment)

> **Scope of this file (evidence only):** each row states what the cited work *reports*, with the verifiable
> identifiers checked here and now. It makes **no** claim about this project's results and does not compare
> "better/worse" — see `## Incomparability notes` for where cross-reading is dangerous.
>
> **Verification status legend**
> - **verified (id)** — bibliographic identifiers (title/authors/venue/DOI or arXiv id) confirmed today against
>   OpenAlex and/or the arXiv API for this matrix.
> - **unverified (abstract)** — record exists and was reached via one of those APIs, but only the abstract was
>   read; any number pulled from the abstract is labelled accordingly.
> - **not read** — record identified (title/DOI/venue) but the full text was not accessible from here (publisher
>   paywall / direct download blocked). Where a specific number is still citable from an accessible secondary
>   source it is given and flagged as secondary; otherwise the cell reads **"no public reported value found"**
>   rather than an estimate.

All access dates below: **2026-07-24** (UTC, retrieval date of the OpenAlex / arXiv records used).

---

## 1. Classical ML on e-nose / food data (SVM / RF / XGBoost / PCA+SVM)

| # | Work (title — authors, year, venue/DOI) | What the paper reports (fact only) | Reported accuracy | Sample scale | Verified |
|---|---|---|---|---|---|
| 1.1 | "Quality Control of Olive Oils Using Machine Learning and Electronic Nose" — Ordukaya & Karlık, 2017, *J. Food Quality* — DOI: [10.1155/2017/9272404](https://doi.org/10.1155/2017/9272404) | ML pipeline for virgin olive-oil classification on e-nose data (ML family: SVM among others) | No public reported value found (full text not read; number not extractable here) | No public reported value found | verified (id) |
| 1.2 | "Electronic nose and tongue combination for improved classification of Moroccan virgin olive oil profiles" — Haddi, Alami, El Bari, Tounsi, Barhoumi, Maaref, Jaffrézic-Renault, Bouchikhi, 2013, *Food Res. Int.* — DOI: [10.1016/j.foodres.2013.09.036](https://doi.org/10.1016/j.foodres.2013.09.036) | E-nose + e-tongue fusion for Moroccan virgin-olive-oil variety classification | No public reported value found | No public reported value found | verified (id) |
| 1.3 | "Validation of the rapid detection approach for enhancing the electronic nose systems performance, using different deep learning models and support vector machines" — Rodríguez Gamboa, da Silva, Araújo, Albarracín, Durán Acevedo, 2020, *Sensors and Actuators B: Chem.* — DOI: [10.1016/j.snb.2020.128921](https://doi.org/10.1016/j.snb.2020.128921); arXiv:[2005.01611](https://arxiv.org/abs/2005.01611) | Comparison of a rapid-detection approach (shorter exposure time) across several deep-learning and SVM classifiers for e-nose performance | No public reported value found | No public reported value found | verified (id) |
| 1.4 | "Rapid analysis of meat floss origin using a supervised machine learning-based electronic nose towards food authentication" — 2023, *npj Science of Food* — DOI: [10.1038/s41538-023-00205-2](https://doi.org/10.1038/s41538-023-00205-2) | Supervised ML e-nose pipeline for meat-floss geographic-origin authentication | No public reported value found | No public reported value found | verified (id) |
| 1.5 | "Bacteria classification using Cyranose 320 electronic nose" — 2002, *Bioméd. Eng. Online* — DOI: [10.1186/1475-925x-1-4](https://doi.org/10.1186/1475-925x-1-4) | Bacteria discrimination with the Cyranose 320 e-nose | No public reported value found | No public reported value found | verified (id) |
| 1.6 | "Electronic Nose and Its Applications: A Survey" — 2019, *Machine Intelligence Research* — DOI: [10.1007/s11633-019-1212-9](https://doi.org/10.1007/s11633-019-1212-9) | Survey of e-nose sensing and recognition; useful as a hub for citing the classical-ML (SVM/PLS/ANN) literature, not a primary result itself | n/a (review) | n/a | verified (id) |
| 1.7 | "Gas Recognition in E-Nose System: A Review" — 2022, *IEEE Trans. Biomed. Circuits Syst.* — DOI: [10.1109/tbcas.2022.3166530](https://doi.org/10.1109/tbcas.2022.3166530) | Review of e-nose gas-recognition pipelines incl. hardware | n/a (review) | n/a | verified (id) |

> **Blank (no citable numbers found):** for rows 1.1–1.5 the exact accuracy figures and sample sizes live in the
> paywalled full texts and were **not** read in this pass. If Table II needs concrete "SVM/RF/XGBoost on
> olive-oil / meat / bacteria" numbers, the next step is a targeted read of those PDFs (and/or the two surveys,
> which tabulate prior accuracies) — flagged here, not filled in.

## 2. Sequence / recurrent models (ESN, GRU/LSTM/TCN) on gas, odor or food time series

| # | Work | What the paper reports (fact only) | Reported accuracy / params / cost | Verified |
|---|---|---|---|---|
| 2.1 | "Reservoir computing compensates slow response of chemosensor arrays exposed to fast varying gas concentrations in continuous monitoring" — Fonollosa, Sheik, Huerta, Marco, 2015, *Sensors and Actuators B: Chem.* — DOI: [10.1016/j.snb.2015.03.028](https://doi.org/10.1016/j.snb.2015.03.028) | Uses ESN/reservoir computing to compensate the slow response of a chemosensor array for continuous gas-concentration tracking — the closest classical "reservoir + e-nose" reference found | No public reported value found (full text not read) | verified (id) |
| 2.2 | "Continuous Prediction in Chemoresistive Gas Sensors Using Reservoir Computing" (conference) — Sheik, Marco, Huerta, Fonollosa, 2014, *Procedia Eng.* — DOI: [10.1016/j.proeng.2014.11.285](https://doi.org/10.1016/j.proeng.2014.11.285) | Early conference companion to 2.1: continuous prediction with a chemiresistive gas-sensor array via reservoir computing | No public reported value found | verified (id), open access |
| 2.3 | "Early-Stage Gas Identification Using Convolutional Long Short-Term Neural Network with Sensor Array Time Series Data" — Zhou & Liu, 2021, *Sensors* — DOI: [10.3390/s21144826](https://doi.org/10.3390/s21144826) | ConvLSTM on raw sensor-array time series for early-stage gas identification | No public reported value found (full text not read) | verified (id) |
| 2.4 | "A novel electronic nose classification prediction method based on TETCN" — Wu, Ma, Li, Fei, Duan, Peng, 2024, *Sensors and Actuators B: Chem.* — DOI: [10.1016/j.snb.2024.135272](https://doi.org/10.1016/j.snb.2024.135272) | TETCN (temporal convolutional e-nose) for classification + prediction | No public reported value found (full text not read) | verified (id) |
| 2.5 | "Validation of the rapid detection approach…" (Rodríguez Gamboa et al., 2020 — see 1.3) | The deep-learning side of that study is the LSTM/CNN comparison; accuracy figures not read here | No public reported value found | verified (id) |
| 2.6 | "An Odor Labeling Convolutional Encoder-Decoder for Odor Sensing in Machine Olfaction" — 2020, arXiv: [2011.12538](https://arxiv.org/abs/2011.12538) | Odor-sensing with a convolutional encoder–decoder (machine olfaction) | No public reported value found (abstract not read in depth) | verified (id) |

## 3. Biologically-inspired / connectome-based reservoirs (the direct comparators for this project's R0 substrate)

| # | Work | Substrate source | Nodes / edges | Frozen? | Trainable params | Reported perf. | Verified |
|---|---|---|---|---|---|---|---|
| 3.1 | "The Drosophila Connectome as a Computational Reservoir for Time-Series Prediction" — Costi, Hadjiivanov, Dold, Hale, Izzo, 2025, *Biomimetics* 10(5):341 — DOI: [10.3390/biomimetics10050341](https://doi.org/10.3390/biomimetics10050341) (open access); **near-duplicate preprint** "The Connectome of a Drosophila as a Computational Reservoir" — same authors, 2025, *Preprints.org* — DOI: [10.20944/preprints202504.1215.v1](https://doi.org/10.20944/preprints202504.1215.v1) | Whole *Drosophila* connectome used as a fixed/graph reservoir for time-series prediction | No public reported value found (full text not read from here — MDPI/preprints direct download was blocked in this environment; numbers should be pulled from the paper itself, not estimated) | **Yes — the connectome is the frozen substrate** (per title/venue claim); exact treatment not confirmed in text | No public reported value found | No public reported value found | verified (id), open access, full text **not read** |
| 3.2 | "Connectome-based reservoir computing with the conn2res toolbox" — Suárez, Mihalik, Milisav, Marshall, Li, Vértes, Lajoie, Mišić, 2024, *Nature Communications* — DOI: [10.1038/s41467-024-44900-4](https://doi.org/10.1038/s41467-024-44900-4) (open access); earlier versions: *PLoS Comput. Biol.* 2022, DOI: [10.1371/journal.pcbi.1010639](https://doi.org/10.1371/journal.pcbi.1010639); *bioRxiv* 2023, DOI: [10.1101/2023.05.31.543092](https://doi.org/10.1101/2023.05.31.543092) | Human/brain connectome (structural connectivity) → reservoir, via a published Python toolbox | No public reported value found from here | Yes, in the standard conn2res usage (frozen connectivity, trained readout) — confirm per run settings in the paper | No public reported value found | No public reported value found | verified (id), open access, full text **not read** |
| 3.3 | "Classifying continuous, real-time e-nose sensor data using a bio-inspired spiking network modelled on the insect olfactory system" — Diamond, Schmuker, Berna, Trowell, Nowotny, 2016, *Bioinspiration & Biomimetics* 11:026002 — DOI: [10.1088/1748-3190/11/2/026002](https://doi.org/10.1088/1748-3190/11/2/026002) | Insect-olfactory-system spiking model as the classifier (NOT a frozen-reservoir design; it is a trained spiking net) — closest "insect-olfaction-based e-nose" reference found | No public reported value found | No public reported value found | No public reported value found | No public reported value found | verified (id), full text **not read** |
| 3.4 | "Spiking Neural Network E-Nose classifier chip" — Abdel-Aty-Zohdy, Allen, Ewing, 2010, IEEE NAECON (chip) — DOI: [10.1109/naecon.2010.5712980](https://doi.org/10.1109/naecon.2010.5712980) | Hardware chip implementing an SNN e-nose classifier — useful only for the "edge/hardware" slot, not a biological-substrate paper | No public reported value found | No public reported value found | No public reported value found | No public reported value found | verified (id) |

> **Honest gap for Section 3:** only **3.1 / 3.2** are genuinely "biological connectome → reservoir" works found in
> this pass, and their substrate scale / trainable-parameter counts / reported accuracy were **not** extracted here
> (full text not read). No public reported value was found *in this pass* for a **frozen *Drosophila*-specific**
> connectome reservoir; 3.1 is the nearest match and should be treated as such, not as an identical design.

## 4. Neuromorphic / in-sensor "lightweight" gas sensing (the deployment-side comparators)

| # | Work | What it reports | Verified |
|---|---|---|---|
| 4.1 | "Bionic Olfactory Neuron with In-Sensor Reservoir Computing for Intelligent Gas Recognition" — Wu, Shi, Jiang, Lin, Song, Wang, Huang, 2025, *Advanced Materials* — DOI: [10.1002/adma.202419159](https://doi.org/10.1002/adma.202419159) | In-sensor reservoir computing for gas recognition — neuromorphic, substrate-level integration | verified (id), full text not read |
| 4.2 | "Spatiotemporal Data Processing with Memristor Crossbar-Array-Based Graph Reservoir" — Jang et al., 2023, *Advanced Materials* — DOI: [10.1002/adma.202309314](https://doi.org/10.1002/adma.202309314) | Memristive crossbar graph reservoir for spatiotemporal processing | verified (id), full text not read |
| 4.3 | "In-sensor reservoir computing for gas pattern recognition using Pt-AlGaN/GaN HEMTs" — Jiang et al., 2024, *Device* — DOI: [10.1016/j.device.2024.100550](https://doi.org/10.1016/j.device.2024.100550) (open access) | In-sensor RC on HEMT gas devices | verified (id), open access, full text not read |
| 4.4 | "Artificial Olfactory Neuron for an In-Sensor Neuromorphic Nose" — Han et al., 2022, *Advanced Science* — DOI: [10.1002/advs.202106017](https://doi.org/10.1002/advs.202106017) | In-sensor neuromorphic olfactory neuron | verified (id), full text not read |
| 4.5 | "Learning function from structure in neuromorphic networks" — Suárez, Richards, Lajoie, Mišić, 2021, *Nature Machine Intelligence* — DOI: [10.1038/s42256-021-00376-1](https://doi.org/10.1038/s42256-021-00376-1) (open access) | Theoretical link between network structure and learned function (frame-setting for "does a real connectome help?") | verified (id), open access, full text not read |
| 4.6 | "Elegans-AI: How the connectome of a living organism could model artificial neural networks" — 2024, *Neurocomputing* — DOI: [10.1016/j.neucom.2024.127598](https://doi.org/10.1016/j.neucom.2024.127598) | Worm connectome → ANN modelling perspective | verified (id) |

## 5. Engineering-reporting comparability (latency / throughput / memory)

| # | Work | How latency / throughput / memory is reported | Verified |
|---|---|---|---|
| 5.1 | "Sensor Drift Compensation via Olfactory system and Reservoir Computing" — Dong, Li, Yajima, 2026, arXiv: [2608.24288](https://arxiv.org/abs/2608.24288) | Sample-wise **online** drift compensation: SNN feature adaptation (STDP) + spiking-reservoir-computing classification with winner-take-all self-supervised adaptation; evaluated on a real sensor-drift dataset. Abstract claims "clear improvement in classification accuracy over baseline methods" **without giving the number** in the abstract. | **unverified (abstract)** — full text not read; specific accuracy values not citable from here |
| 5.2 | "LDAC-Net: A Learnable Multi-Lag Differencing Attention-Convolution Network for Drift-Robust Recognition with Low-Cost MOX Gas Sensors" — Zhang, Han, Shi, Sobeih, 2026, arXiv: [2608.25646](https://arxiv.org/abs/2608.25646) | **Concrete, citable numbers from the abstract:** on **SmellNet-Base** (50-class): **68.2 % top-1**, ≈ +14 pp over best FOTD-preprocessed comparison, > +30 pp over raw-input Transformer; on **SmellNet-Mixtures**: 45.4 % → **50.5 %**; on **eNose-Drift** (62-channel, strong long-term drift): **70.6 % top-1, 69.6 % macro-F1**, outperforming the best dataset-retuned-FOTD comparison by 8.0 / 3.0 pp. Reports dataset sizes / channel counts but **no latency, throughput, memory, or hardware in the abstract**. | **unverified (abstract)** — numbers are from the arXiv abstract only; treat as "reported in the abstract of arXiv:2608.25646" not as a journal-verified result |
| 5.3 | "Spiking Neural Network E-Nose classifier chip" (3.4) | Hardware (chip) — energy/throughput style reporting expected, but no value extracted here | verified (id) |
| 5.4 | In-sensor RC works 4.1–4.4 | Report device-level energy/latency where applicable, but **no public reported value found** in this pass | verified (id) |

> **Where latency/throughput/memory actually is:** almost every paper above reports **accuracy only**; the few that
> touch deployment (4.1–4.4, 5.3) do so at the *device/hardware* level, and none of them was read in full here.
> **Do not put a latency/throughput/memory number in Table III unless it was read from the primary text.**

---

## Incomparability notes (read before building Table II / III from this)

1. **Accuracy ≠ comparability.** Reported accuracies above span **different datasets, class counts, and exposure
   protocols** (50-class SmellNet-Base vs 62-channel eNose-Drift vs a single olive-oil study vs a bacteria set).
   Comparing their percentages as if they were like-for-like is a mistake this matrix is **not** licensed to make.
2. **"Reservoir node count" vs "trainable parameters" are different things.** The connectome papers (3.1, 3.2, 4.5,
   4.6) and in-sensor RC papers (4.1–4.4) report *substrate* size; only a few actually separate *frozen* reservoir
   weight count from *trained readout* parameters. Where a paper only states "N nodes / M edges," **do not** call
   that "N trainable parameters."
3. **Frozen vs trained reservoir.** 3.1/3.2 are frozen-substrate + trained readout (the design this project mirrors);
   3.3/3.4/5.1 are **trained** spiking systems — a different category. Mixing them into one row misrepresents the
   method.
4. **Food vs gas.** "Gas sensor" (CO/CH4 etc.) literature (2.1, 2.2, 4.3, 5.2) and **food/olfaction** literature
   (1.x, 3.3) use different sensor chemistries and tasks; a gas-sensor RC result is not a food-quality e-nose result.
5. **Preprint vs peer-reviewed.** 5.1, 5.2 are 2026 **arXiv preprints** (abstract-level evidence only); 3.1 appears
   in both a preprint (10.20944/…) and a *Biomimetics* journal version (10.3390/biomimetics10050341) — cite the
   peer-reviewed one, note the preprint.
6. **No public reported value found** is a legitimate cell value; it is **not** permission to back-fill a number.

## Sources (retrieved & verified 2026-07-24)

- 1.1 [10.1155/2017/9272404](https://doi.org/10.1155/2017/9272404) — Ordukaya & Karlık 2017, *J. Food Quality*
- 1.2 [10.1016/j.foodres.2013.09.036](https://doi.org/10.1016/j.foodres.2013.09.036) — Haddi et al. 2013, *Food Res. Int.*
- 1.3 / 2.5 [10.1016/j.snb.2020.128921](https://doi.org/10.1016/j.snb.2020.128921), arXiv: [2005.01611](https://arxiv.org/abs/2005.01611) — Rodríguez Gamboa et al. 2020
- 1.4 [10.1038/s41538-023-00205-2](https://doi.org/10.1038/s41538-023-00205-2) — 2023, *npj Science of Food*
- 1.5 [10.1186/1475-925x-1-4](https://doi.org/10.1186/1475-925x-1-4) — 2002, *Bioméd. Eng. Online*
- 1.6 [10.1007/s11633-019-1212-9](https://doi.org/10.1007/s11633-019-1212-9) — 2019 survey, *MIR*
- 1.7 [10.1109/tbcas.2022.3166530](https://doi.org/10.1109/tbcas.2022.3166530) — 2022 review, *IEEE TBCAS*
- 2.1 [10.1016/j.snb.2015.03.028](https://doi.org/10.1016/j.snb.2015.03.028) — Fonollosa et al. 2015
- 2.2 [10.1016/j.proeng.2014.11.285](https://doi.org/10.1016/j.proeng.2014.11.285) — Sheik et al. 2014 (open access)
- 2.3 [10.3390/s21144826](https://doi.org/10.3390/s21144826) — Zhou & Liu 2021
- 2.4 [10.1016/j.snb.2024.135272](https://doi.org/10.1016/j.snb.2024.135272) — Wu et al. 2024
- 2.6 arXiv: [2011.12538](https://arxiv.org/abs/2011.12538) — 2020
- 3.1 [10.3390/biomimetics10050341](https://doi.org/10.3390/biomimetics10050341) + preprint [10.20944/preprints202504.1215.v1](https://doi.org/10.20944/preprints202504.1215.v1) — Costi et al. 2025
- 3.2 [10.1038/s41467-024-44900-4](https://doi.org/10.1038/s41467-024-44900-4) + [10.1371/journal.pcbi.1010639](https://doi.org/10.1371/journal.pcbi.1010639) + [10.1101/2023.05.31.543092](https://doi.org/10.1101/2023.05.31.543092) — Suárez et al. 2024/2022/2023
- 3.3 [10.1088/1748-3190/11/2/026002](https://doi.org/10.1088/1748-3190/11/2/026002) — Diamond et al. 2016
- 3.4 / 5.3 [10.1109/naecon.2010.5712980](https://doi.org/10.1109/naecon.2010.5712980) — Abdel-Aty-Zohdy et al. 2010
- 4.1 [10.1002/adma.202419159](https://doi.org/10.1002/adma.202419159) — Wu et al. 2025
- 4.2 [10.1002/adma.202309314](https://doi.org/10.1002/adma.202309314) — Jang et al. 2023
- 4.3 [10.1016/j.device.2024.100550](https://doi.org/10.1016/j.device.2024.100550) — Jiang et al. 2024 (open access)
- 4.4 [10.1002/advs.202106017](https://doi.org/10.1002/advs.202106017) — Han et al. 2022
- 4.5 [10.1038/s42256-021-00376-1](https://doi.org/10.1038/s42256-021-00376-1) — Suárez et al. 2021 (open access)
- 4.6 [10.1016/j.neucom.2024.127598](https://doi.org/10.1016/j.neucom.2024.127598) — 2024
- 5.1 arXiv: [2608.24288](https://arxiv.org/abs/2608.24288) — Dong, Li, Yajima 2026 (preprint)
- 5.2 arXiv: [2608.25646](https://arxiv.org/abs/2608.25646) — Zhang, Han, Shi, Sobeih 2026 (preprint)

*Access date for all OpenAlex/arXiv lookups: 2026-07-24 (UTC). Full-text PDFs for paywalled items were not
retrieved; their numeric cells are deliberately left as "no public reported value found" rather than estimated.*
