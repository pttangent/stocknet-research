# StockNet Final Research Report

## Executive Summary

StockNet studies whether U.S. equities form non-preset intraday co-evolution communities from `5m`, `15m`, and `30m` price and volume behavior, whether those communities exhibit observable lifecycles, and whether parts of their evolution can be predicted and organized into community rotation signals.

The current repository supports a strong research prototype and a mostly answered set of research questions. The strongest confirmed results are non-preset community discovery, multi-resolution comparability, edge-persistence prediction, and a first formal edge-emergence benchmark. The newest addition is a first `Community Rotation Detection v1` layer that converts lifecycle, migration, and emergence outputs into candidate source-to-target rotation events.

## Research Scope

- market: `U.S. equities`
- 15m parquet universe: `3808` symbols
- frequencies: `5m / 15m / 30m`
- snapshot dataset: `1536` windows
- active nodes per snapshot: `199`
- compute backend for graph snapshot build: `torch`
- historical span: `roughly two months`

This scope is enough for structure discovery, cross-resolution comparison, lifecycle experiments, and short-horizon prediction tasks. It is not enough for strong production trading claims.

## Methodology Summary

### Data construction

The repository now follows a `5m`-first intraday architecture:

`5m raw -> 15m resample -> 30m resample`

This matters because the three resolutions now come from a single raw source rather than three independently fetched datasets.

### Graph construction

Each snapshot is a stock graph where:

- nodes are stocks
- edges capture intraday co-evolution
- node features include return, residual return, abnormal volume, volatility, liquidity, and graph-position features
- edge features include return correlation, residual correlation, volume correlation, edge strength, and persistence semantics

### Research outputs

The current pipeline produces:

- graph snapshots
- temporal labels and lifecycle artifacts
- consensus and null-model outputs
- multi-resolution consistency outputs
- edge-persistence TGNN results
- community-survival TGNN results
- node-migration TGNN results
- edge-emergence baseline results
- community-rotation detection outputs

## RQ1: Do non-preset co-evolution communities exist?

### Evidence

- 5m communities: `291`
- 15m communities: `155`
- 30m communities: `98`
- rotation-lifecycle timeseries rows: `7691`

### Answer

> Yes. The current system repeatedly discovers non-preset intraday co-evolution communities from graph structure without predefining sectors or themes.

The positive answer is currently strongest at the prototype-research level rather than publication-grade significance, but the structure is clearly not empty.

## RQ2: What distinct roles do 5m, 15m, and 30m play?

### Evidence

- NMI 5m vs 15m: `0.34843628533592874`
- NMI 15m vs 30m: `0.48012162414876774`
- NMI 5m vs 30m: `0.3019278314742274`
- persistent / confirmed / emerging communities: `1 / 2 / 11`

### Answer

> `5m` behaves like an earlier and noisier discovery layer, `15m` is the most useful main analytical frequency, and `30m` behaves like a confirmation or denoising layer.

This is a meaningful but still provisional conclusion. The current numbers support the role split rather than proving it as a final theorem.

## RQ3: Do communities exhibit lifecycles?

### Evidence

- lifecycle-aware temporal outputs now exist and are used in downstream rotation detection:
- `lifecycle_communities.csv`
- `lifecycle_events.csv`
- `node_membership_timeline.csv`
- `node_migration_labels.csv` based on `lifecycle_id` rather than local `community_id`

### Answer

> Communities appear to have observable lifecycles, and the repository now has the correct identity model to study them.

This is a major methodological improvement, but it should still be treated as an active validation area rather than a final end-state conclusion.

## RQ4: Are real communities stronger than random structure?

### Evidence

- consensus communities: `32`
- time-shuffle p-value: `0.0`
- label-shuffle p-value: `0.88`
- real persistence score: `271.0`

The null-validation outputs now also include more research-meaningful structure metrics such as internal coherence, node coverage, member confidence, structure score, and per-community significance tables.

### Answer

> Real communities are clearly stronger than naive time-shuffled null structure, but the current evidence is not yet strong enough to claim full significance against more difficult structure-preserving null baselines.

This remains one of the main open research tasks.

## RQ5: Can models learn community evolution?

### Snapshot Edge Persistence

- TGNN AUC / AP / F1: `0.7527 / 0.9364 / 0.9160`
- TGNN test rows: `466671.0000`
- device: `cuda`
- GPU enabled: `True`

### Community Survival

- TGNN AUC / AP / F1: `0.7816 / 0.9638 / 0.9488`
- TGNN test rows: `2085.0000`
- device: `cuda`

### Node Migration

- TGNN AUC / AP / F1: `0.5025 / 0.2352 / 0.3588`
- TGNN test rows: `60894.0000`
- device: `cuda`

### Edge Emergence Baseline

- XGBoost AUC / AP / F1: `0.9345 / 0.5169 / 0.5741`
- baseline test rows: `756787.0000`

### Answer

> Yes, the system can learn several forms of network evolution. Edge persistence is the strongest confirmed task, community survival is promising, edge emergence is now a real benchmarked task, and node migration remains weak.

## Baseline Comparison

| model | metric | value |
| --- | --- | --- |
| tgnn_snapshot | auc | 0.752693 |
| static_graph_logistic | auc | 0.690802 |
| logistic_regression | auc | 0.682427 |
| edge_strength | auc | 0.628952 |
| persistence | auc | 0.611598 |
| tgnn_snapshot | average_precision | 0.936367 |
| static_graph_logistic | average_precision | 0.916836 |
| logistic_regression | average_precision | 0.913467 |
| edge_strength | average_precision | 0.899253 |
| persistence | average_precision | 0.876090 |
| logistic_regression | f1 | 0.916115 |
| tgnn_snapshot | f1 | 0.916039 |
| static_graph_logistic | f1 | 0.915859 |
| persistence | f1 | 0.880523 |
| edge_strength | f1 | 0.622045 |

## RQ6: Can the system detect community rotation rather than only isolated community behavior?

### Evidence

- community timeseries rows: `7691`
- raw rotation event rows: `1168`
- qualified rotation event rows: `1116`

Rotation in this report is defined as a structural transfer pattern rather than a traditional sector label switch:

- source community decay
- target community expansion
- member migration, edge rewiring, or relative-strength transfer between source and target
- optional multi-resolution support on the target side

### Answer

> The repository now has a first formal `Community Rotation Detection v1` layer. It can generate candidate source-to-target rotation events, but it should still be treated as an early detection system rather than a final economic-interpretation engine.

This is enough to support research observation of structural rotation, but not yet enough to claim full capital-flow inference.

## Top Rotation Candidates

| timestamp | source_lifecycle_id | target_lifecycle_id | source_stage | target_stage | migrated_members | migrated_symbols | rewired_edges | rewired_edge_pairs | rotation_confidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-05-28 19:45:00+00:00 | L1205 | L1197 | maturity | expansion | 14 | AAON;ABCB;ABX;ACET;ACLS;ACM;AEG;AEM;AER;AG;AGEN;AGI;AMCR;AME | 120 | AA-AMAT;AAL-ALC;AAMI-AER;AAMI-AME;AAOI-AEBI;AAOI-AEO;AAOI-AGL;AAOI-AIRJ;AAON-ACMR;AAON-AFG;AAPL-ALMS;AAUC-AEBI;AAUC-AEO;ABCB-AFG;ABCB-AFJK;ABCB-ALH;ABCB-ALMU;ABCB-AMAL;ABEV-AEBI;ABEV-AEO;ABEV-AIRG;ABEV-AIRJ;ABG-AEG;ABR-ACU;ACA-AER;ACA-AME;ACAD-QQQ;ACET-ADNT;ACET-AFYA;ACET-AGL;ACHC-AEBI;ACLS-ACMR;ACLS-AGM;ACLS-ALGN;ACLS-AMD;ACM-AFG;ACMR-AEIS;ACR-ADAG;ACR-QQQ;ACRE-AEG;ACRE-AGNC;ACT-AEO;ACT-AHR;ACTG-AEHR;ACTG-AGO;ACTG-AMLX;ACU-AGL;ADEA-ALGM;ADNT-AER;ADTN-AFJK;ADTN-AIR;ADTN-AIT;ADTN-ALMU;ADUR-AEO;ADUR-AFG;ADUS-AHRT;ADUS-AMBA;AEBI-AEC;AEBI-AEHR;AEBI-AEIS;AEBI-AGNC;AEBI-AGO;AEBI-AHRT;AEBI-AMBA;AEBI-AMPG;AEC-AEO;AEC-AFG;AEG-AIR;AEG-AKR;AEG-ALH;AEG-AMAL;AEG-AMD;AEHR-AEO;AEHR-AGL;AEHR-AIR;AEHR-AIRJ;AEHR-ALMU;AEIS-AFG;AEIS-ALH;AEIS-ALMU;AEM-AMAT;AEO-AGNC;AEO-AMBA;AEO-AMPG;AER-AFJK;AER-ALV;AER-AMG;AER-AMP;AFG-AGNC;AFG-AGO;AFG-AGX;AFG-ALRS;AFG-AMBA;AFG-AMPG;AFYA-AGI;AGEN-AKR;AGEN-ALV;AGL-AMLX;AGM-AMAN;AGNC-AIRG;AGNC-AIRJ;AGNC-ALMU;AGO-AHR;AGX-ALH;AHR-AHRT;AHR-AMBA;AHRT-AIR;AHRT-AIRJ;AHRT-ALTG;AIR-AME;AIRJ-AMBA;AIRJ-AMPG;AKAM-AMP;ALC-AMG;ALGN-AMCR;ALGT-AME;ALK-AMKR;ALMU-AMBA;AMAN-QQQ;AMD-AME | 1.879235 |
| 2026-03-20 19:45:00+00:00 | L0137 | L0150 | expansion | maturity | 6 | AA;AAPG;ABSI;ACNT;ALM;AMLX | 87 | AA-AAOI;AA-ACRS;AA-ADM;AA-AEHR;AAL-AAP;AAL-AAPG;AAL-ABG;AAL-AEG;AAL-AGCC;AAL-ALRS;AAOI-ABSI;AAOI-ALKS;AAOI-ALM;AAOI-AMLX;AAON-ADV;AAON-AEVA;AAON-ALB;AAON-ALGM;AAP-ABUS;AAP-ADV;AAP-AGEN;AAPG-ADV;AAPG-AERO;AAPG-AGI;AARD-ACIC;AARD-ACNT;AARD-AD;AARD-AKO-B;ABAT-ABSI;ABCB-ACAD;ABCB-ACIC;ABCB-ACNB;ABEO-ACDC;ABEO-ACRS;ABEO-AESI;ABG-ADTN;ABSI-ACB;ABSI-ACDC;ABSI-ACHR;ABSI-ADM;ABSI-AEHR;ABSI-ALMU;ABSI-ALOY;ACAD-ALB;ACAD-ALGM;ACAD-ALTG;ACB-AGRO;ACDC-AGRO;ACIC-ACLS;ACIC-ADV;ACIC-AEM;ACIC-AIP;ACIC-ALTG;ACIC-QQQ;ACNT-ADM;ACNT-AEHR;ACNT-QQQ;AD-ADV;AD-QQQ;ADI-AEVA;ADI-ALB;ADI-ALGM;ADI-AMD;AEG-ALB;AEHR-ALM;AEHR-AMLX;AEM-ALKS;AER-ALB;AER-ALGM;AERO-AGCC;AESI-AGRO;AGEN-AKO-B;AGIO-AKAM;AGIO-ALB;AGIO-QQQ;AGRO-ALMU;AGRO-ALOY;AIP-AMBQ;AIRO-ALM;ALB-AMLX;ALGM-AMBQ;ALGM-AMP;ALH-AMP;ALM-ALMU;ALM-ALOY;ALM-AMKR;AMKR-AMLX | 1.774049 |
| 2026-05-27 13:45:00+00:00 | L1164 | L1177 | expansion | confirmation | 9 | AAMI;AAON;ACOG;AGRO;AIAI;AIRG;AIZ;AKO-B;AMH | 46 | AAMI-AMBA;AAMI-AME;AAON-AFYA;AAON-AIP;AAON-ALAB;AAON-ALVO;AARD-ACA;AARD-AD;AAUC-AIRS;ACA-AEXA;ACA-AIR;ACB-AIAI;ACB-AIZ;ACDC-ACNT;ACDC-AHR;ACLS-ACOG;ACNT-AEXA;ACNT-AIP;ACNT-AIR;ACNT-ALAR;ACNT-ALVO;ACNT-AMP;ACOG-AFYA;ACOG-AMAT;ACR-AEBI;ACR-AEIS;ACR-AIP;ACR-AMAT;ACU-AESI;ACU-AM;AD-AIR;ADAG-AMG;AEXA-AIV;AEXA-AMG;AGRO-AIR;AGRO-ALVO;AI-AIRG;AI-AMH;AIAI-ALAB;AIIR-AMG;AIP-AKO-B;AIP-AMG;AIR-AMG;AIRG-ALAB;ALVO-AMG;AMBA-AMG | 1.763246 |
| 2026-06-03 14:30:00+00:00 | L1274 | L1278 | expansion | maturity | 8 | ABEV;ACHV;ADMA;ADV;AGI;AGL;AIRS;ALAR | 52 | AA-AAMI;AA-AER;AA-AFYA;AAMI-ALGT;AAOI-ALX;ABCB-AGPU;ABCB-ALKT;ABEV-AFYA;ABEV-ALV;ABG-AGI;ABG-AIAI;ABNB-ADP;ABNB-AFRM;ABNB-AGI;ABNB-ALGT;ABSI-AGNT;ABX-AFYA;ACEL-ADP;ACET-AEM;ACIW-ALLY;ACMR-AII;ACN-ALLY;ACRE-AEBI;ACTG-AEO;ACVA-AEBI;ACVA-AGNT;ACVA-AII;ACVA-ALV;ADMA-ALV;ADPT-AMG;ADUS-AEM;ADV-ALV;ADV-ALX;AEG-AEM;AEG-AGL;AEM-AFYA;AEM-ALV;AEYE-AGNT;AEYE-AIN;AFRM-AFYA;AFRM-AMG;AFYA-AIAI;AFYA-AIRS;AGBK-ALGT;AGNT-ALGT;AI-ALX;AIAI-ALLY;AIRS-AMG;AKAM-ALX;ALK-ALLY;ALV-AMCR;AMCR-AMG | 1.749817 |
| 2026-04-02 13:45:00+00:00 | L0356 | L0355 | confirmation | confirmation | 10 | A;AAOI;ABCL;ABSI;ADMA;ADV;AGIO;AKTS;ALB;AMPG | 81 | A-AA;A-ACM;A-AD;A-AG;A-AGI;A-AIRO;A-ALGM;A-ALOY;A-AMKR;AA-ABUS;AA-AKTS;AA-AMBP;AAOI-ALAB;AAON-ACAD;AAPG-ALAB;ABCL-ACHR;ABCL-ACLS;ABCL-ADTN;ABCL-ALAB;ABEV-ADTN;ABEV-AEG;ABEV-AGI;ABEV-AGX;ABSI-ACLS;ABSI-ALGM;ABSI-ALOY;ABVX-ALOY;ABX-AII;ACAD-AIT;ACAD-ALGT;ACB-AEM;ACB-AFRM;ACB-AG;ACB-AIT;ACB-ALGN;ACB-ALK;ACB-AMKR;ACHC-AGNC;ACHR-AI;ACHR-AMLX;ACHV-ACLS;ACHV-ALAB;ACLS-AMLX;ACM-AEXA;ACM-AGNC;ACM-AI;ACMR-ADV;ADMA-ALAB;ADMA-ALGT;ADMA-ALK;ADMA-ALMU;ADMA-AMD;ADMA-AMKR;ADTN-AGNC;ADTN-ALLT;ADV-AG;ADV-AGI;ADV-AGX;ADV-AIP;ADV-AIRO;ADV-ALAB;AEVA-AI;AEVA-AMLX;AEXA-AG;AEXA-AGL;AFRI-AIRO;AGIO-AIRO;AGIO-ALMS;AGNC-ALGM;AGNC-ALLY;AGNC-QQQ;AGX-ALB;AI-QQQ;AII-ALM;AIP-AMPG;AIR-AMPG;ALAB-AMLX;ALGM-AMPG;ALMU-AMLX;ALMU-AMPG;ALTO-AMBP | 1.705385 |
| 2026-03-30 19:45:00+00:00 | L0317 | L0318 | expansion | maturity | 6 | AA;ACDC;ADAG;AEXA;AGRO;ALVO | 127 | AA-AAON;AA-ABAT;AA-ABEV;AA-ACU;AA-AEHR;AA-AEIS;AA-AEVA;AA-AIR;AA-ALNT;AA-AMAT;AA-AMCI;AA-AMKR;AA-QQQ;AAL-ACDC;AAL-AMG;AAMI-ABCL;AAMI-AEG;AAOI-ADAG;AAON-ADM;AAON-AEM;AAON-AEO;AAON-AGRO;AAP-ADNT;AAP-ADUR;ABCL-ACB;ABCL-ADNT;ABCL-AEHR;ABCL-AG;ABCL-AIP;ABCL-ALGM;ABVX-AERO;ABVX-ALSN;ABX-ADUR;ABX-AIP;ABX-ALAB;ACB-ADM;ACB-AEG;ACB-AEM;ACB-AEYE;ACB-AFRM;ACB-AGBK;ACDC-ACMR;ACDC-ADTN;ACDC-AEIS;ACDC-AGCO;ACDC-ALNT;ACDC-AMAT;ACDC-AMKR;ACHR-ALKT;ACHR-ALVO;ACM-ACNT;ACM-AEM;ACM-AFCG;ACM-AGBK;ACM-ALNY;ACMR-ADAG;ACTG-AD;ACVA-ADNT;ACVA-ADUR;ACVA-AG;AD-AGL;AD-AGX;ADAG-ADTN;ADAG-AGX;ADAG-AMAT;ADI-AGRO;ADM-AERO;ADM-AG;ADM-ALG;ADM-ALSN;ADM-AME;ADNT-AEG;ADNT-AEO;ADNT-AFRM;ADNT-AGBK;ADNT-AGM;ADNT-AIRS;ADNT-ALNY;ADNT-ALVO;ADUR-AEG;ADUR-AFRM;ADUR-AGBK;ADUR-AGMB;ADUR-AMLX;AEC-AGPU;AEG-AEHR;AEG-AG;AEG-ALH;AEHR-AEM;AEHR-AEO;AEHR-AGBK;AEHR-AGRO;AEHR-AMLX;AEM-ALGM;AEM-ALSN;AENT-AIRO;AEO-AGI;AEO-ALAB;AEXA-AGX;AEXA-AIR;AEXA-ALH;AEXA-ALLT;AEXA-ALM;AFRM-AIP;AFRM-ALH;AG-AGBK;AG-AGMB;AG-ALNY;AG-ALTO;AG-AMLX;AGBK-AGI;AGBK-ALGM;AGCO-AMG;AGI-ALNY;AGL-ALKT;AGL-AMCX;AGMB-AIR;AGMB-ALAB;AGMB-ALAR;AGMB-ALSN;AGRO-AMAT;AGRO-AMBA;AGRO-AMKR;AGRO-QQQ;AIV-ALSN;ALGM-AMLX;ALVO-AMBA | 1.691215 |
| 2026-05-22 19:45:00+00:00 | L1144 | L1126 | expansion | maturity | 0 |  | 205 | AA-ABEV;AA-AEG;AA-AEM;AA-AG;AA-AGI;AA-AKO-B;AA-AMCI;AAMI-AHRT;AAOI-AGCC;AAON-AIRJ;AAON-QQQ;AAP-ABT;AAP-ACN;AAP-AII;AAP-ALAB;AAP-ALLT;AAP-ALRM;AAUC-ABEV;AAUC-AEM;AAUC-AHR;ABAT-AEG;ABAT-AG;ABCB-ACEL;ABCB-AHRT;ABCB-AIRJ;ABEV-ADI;ABEV-AEC;ABEV-ALSN;ABR-AMAN;ABT-ADUS;ABT-AEXA;ABT-ALHC;ABT-ALOT;ABT-AMPG;ABUS-AEM;ABUS-AG;ABUS-AGI;ACA-AIRO;ACEL-ADNT;ACEL-AGI;ACEL-AIN;ACEL-ALRS;ACGL-ACI;ACI-ADUS;ACI-AJG;ACI-AMLX;ACIC-ALL;ACIW-ADUS;ACIW-AIP;ACIW-ALHC;ACIW-AM;ACLS-AIRJ;ACM-ADUS;ACMR-ADI;ACN-ADUS;ACN-AEXA;ACN-AIP;ACN-ALHC;ACN-ALOT;ACN-AMPG;ACNB-ALSN;ACRS-AMP;ACVA-ADAM;ACVA-AGNC;ACVA-AHCO;ACVA-AIR;ACVA-ALM;ACVA-ALRS;ADAM-ADI;ADAM-AEC;ADAM-AHRT;ADAM-AIRO;ADBE-ALHC;ADBE-ALOT;ADI-ADNT;ADI-AEG;ADI-AEIS;ADI-AEM;ADI-AENT;ADI-AEO;ADI-AFRI;ADI-AGI;ADI-AGX;ADI-AIG;ADI-AIN;ADI-AIR;ADI-AIT;ADI-AKO-B;ADI-ALM;ADI-ALNT;ADI-ALRS;ADI-ALTG;ADI-AMG;ADI-AMKR;ADNT-AHRT;ADNT-AIRJ;ADP-ADUS;ADP-AIP;ADP-ALHC;ADP-ALOT;ADP-AMPG;ADPT-ADUS;ADUS-AMGN;AEC-AEG;AEC-AGI;AEC-AKO-B;AEC-ALRS;AEE-AGCC;AEG-AI;AEG-AIRO;AEG-ALB;AEG-ALSN;AEG-AMD;AEG-QQQ;AEIS-AIRS;AEM-AGMB;AEM-AIRO;AEM-ALB;AEM-ALSN;AEM-AMD;AEM-QQQ;AENT-AKTS;AEO-AIRJ;AEO-AIRO;AEO-AKTS;AEP-AGCC;AER-AHRT;AEXA-AGCC;AEXA-AKAM;AEXA-AMGN;AFBI-AII;AFBI-AMP;AFJK-AGCC;AFL-ALLT;AFRI-AGMB;AFRI-QQQ;AFYA-AMAN;AG-AI;AG-AIRO;AG-QQQ;AGCC-AIRG;AGCC-AIV;AGCC-AMH;AGI-AGMB;AGI-AHRT;AGI-AIRO;AGI-ALSN;AGI-AMD;AGI-QQQ;AGL-AII;AGMB-AGX;AGMB-ALNT;AGNC-AHRT;AGNC-ALL;AGO-ALL;AGX-ALB;AGX-ALSN;AGYS-AJG;AGYS-ALHC;AGYS-AM;AHCO-AIOT;AHR-ALB;AHRT-ALNT;AI-ALM;AI-AMCI;AIG-ALL;AIG-ALSN;AII-ALOT;AIIR-AJG;AIIR-AKR;AIN-AIRJ;AIR-AIRO;AIRJ-AIT;AIRO-AKO-B;AIRO-ALNT;AIRO-ALTG;AIRO-AMKR;AIRS-AMH;AIT-QQQ;AIV-AMP;AIZ-ALL;AJG-ALL;AJG-ALLT;AJG-AMGN;AKO-B-AMD;AKO-B-QQQ;AKR-ALSN;ALAB-ALOT;ALAB-AMPG;ALB-ALM;ALB-AMCI;ALHC-ALLT;ALKS-ALRM;ALLE-AMAN;ALLT-ALOT;ALNT-ALSN;ALNT-AMD;ALNT-QQQ;ALTO-QQQ;AMAN-AMBQ;AMCI-AMD;AMCI-QQQ;AMD-AMKR;AMH-AMP;AMKR-QQQ | 1.673973 |
| 2026-05-27 18:30:00+00:00 | L1190 | L1181 | confirmation | maturity | 20 | AARD;ABCB;ACA;ACB;ACNT;AD;ADAG;ADSE;AEXA;AEYE;AFCG;AFRI;AHR;AIR;AIZ;AKR;ALRS;ALVO;AMCI;AMG | 16 | AARD-ABX;AAUC-AMPH;ABCB-AGM;ABX-ADAG;ABX-AIAI;ACNT-ADM;AD-AHCO;AD-ALLY;AD-AMPH;ADAG-ALGN;ADM-AEXA;AFL-AHR;AGO-ALVO;AIV-ALRS;AIV-AMCI;ALC-ALX | 1.670669 |
| 2026-05-15 19:45:00+00:00 | L1034 | L1030 | maturity | expansion | 0 |  | 166 | A-ABT;A-ACGL;A-ACI;A-ACM;A-ACN;A-ACNT;A-ACVA;A-ADBE;A-ADP;A-AEBI;A-AFL;A-AIG;A-ALL;A-AM;AAMI-AHCO;AAP-AFG;AAP-ALCO;AAUC-AIG;ABBV-ABG;ABBV-AFJK;ABBV-AIT;ABBV-ALKS;ABBV-ALSN;ABBV-ALVO;ABBV-AME;ABBV-AMGN;ABCB-ABT;ABCB-ACVA;ABCB-ADP;ABCB-AFL;ABCB-ALC;ABEO-ADV;ABM-ABT;ABM-ACGL;ABM-ALC;ABM-ALKT;ABT-AD;ABT-ADC;ABT-AEE;ABT-AEG;ABT-AIT;ABT-AIV;ABT-ALLE;ABT-ALSN;ABT-AMG;ABT-AMH;ABUS-AGYS;ACAD-AFG;ACDC-ACI;ACDC-AFL;ACDC-AM;ACGL-AD;ACGL-ADMA;ACGL-ADUS;ACGL-AEP;ACGL-AES;ACGL-AGO;ACGL-AIT;ACGL-ALKS;ACGL-AMG;ACGL-AMP;ACI-ADMA;ACI-ALKS;ACIW-AKR;ACIW-AMH;ACM-ADMA;ACN-AFJK;ACNB-ADP;ACNB-ALC;ACNT-ADUS;ACNT-AESI;ACR-AJG;ACRE-ACT;ACRS-AIAI;ACT-AGO;ACT-AHR;ACT-ALOT;ACT-ALRS;ACVA-AGO;ACVA-AHR;ACVA-ALLY;AD-ADP;AD-AFL;AD-ALC;ADBE-ADMA;ADC-ADP;ADC-AGNT;ADC-AHCO;ADC-AJG;ADC-ALC;ADC-ALL;ADC-ALRM;ADMA-ADP;ADMA-ADSK;ADMA-AFL;ADMA-ALL;ADMA-AM;ADP-AEE;ADP-AEG;ADP-AEP;ADP-AES;ADP-AGO;ADP-AHR;ADP-ALHC;ADP-ALLE;ADP-AMAL;ADP-AMCI;ADP-AMH;ADSE-ALTO;ADSK-AGM;ADT-ALLE;ADT-AMG;ADT-AMH;ADV-AGPU;AEBI-AES;AEE-AFL;AEG-AFG;AEG-AFL;AEG-AIG;AEG-AJG;AEG-ALC;AEP-AFG;AEP-AFL;AEP-ALC;AEP-ALL;AES-AFL;AES-ALC;AESI-AM;AFCG-ALRS;AFL-AGO;AFL-AIT;AFL-ALKS;AFL-ALLE;AFL-ALLY;AFL-AMCI;AFL-AMG;AFL-AMP;AGM-ALRM;AGNT-ALRS;AGO-AHCO;AGO-AJG;AGO-ALC;AHCO-ALSN;AHCO-AMH;AHR-ALC;AHR-ALL;AHR-AM;AIG-AKR;AIG-ALLE;AIG-ALLY;AIV-ALC;AJG-ALHC;AJG-AMH;AKR-ALCO;ALC-ALHC;ALC-ALLE;ALC-ALLY;ALC-ALRS;ALC-AMAL;ALC-AMG;ALC-AMH;ALC-AMP;ALCO-AMAL;ALL-ALLY;ALL-AMP;ALTO-AMLX | 1.668159 |
| 2026-05-04 19:45:00+00:00 | L0868 | L0863 | decay | expansion | 0 |  | 103 | A-ABCL;A-ABEV;A-ABUS;A-ABVX;A-AESI;A-AIRS;A-ALAB;A-AMAT;A-AMKR;A-QQQ;AAMI-ALX;AAOI-AAPL;AAON-ABT;AAON-ACHR;AAON-ADAM;AAON-ADNT;AAON-AMBP;AAPG-ABR;AAPG-AIRG;AAPL-AEHR;AAPL-AHR;AAPL-AIRG;AAPL-ALAB;AAPL-ALGT;ABCL-AEG;ABCL-AER;ABEO-AIG;ABEV-AEG;ABEV-AER;ABG-ALGT;ABG-ALTG;ABR-ACEL;ABR-AIG;ABT-ADEA;ABT-AEIS;ABT-AFYA;ABT-AKO-B;ABUS-AENT;ABUS-AMD;ABVX-AMD;ACB-ADT;ACB-AGL;ACB-AGPU;ACB-AMD;ACEL-ACTG;ACEL-ALMR;ACLS-AMD;ACMR-AEG;ACMR-AMD;ACRE-AMGN;ADAG-AIG;ADAM-ADEA;ADAM-AEIS;ADAM-AFYA;ADAM-AJG;ADAM-AMPG;ADEA-AEVA;ADEA-AGL;ADEA-ALGN;ADNT-ALGM;ADT-ALAB;AEG-AERO;AEG-AGEN;AEG-AIR;AEG-ALM;AEG-ALNT;AEG-QQQ;AEIS-AEVA;AEIS-ALGN;AEIS-AMBP;AENT-AKTS;AENT-ALAB;AENT-ALKS;AER-ALNT;AER-AMAT;AER-QQQ;AERO-AIP;AESI-AGIO;AEVA-AEYE;AFYA-ALX;AGEN-AMD;AGIO-ALVO;AGIO-AMAT;AGPU-AMPG;AHR-AIP;AIG-AIRS;AIP-AKTS;AIP-ALNT;AIP-AMN;AIR-AMD;AIRS-AMCR;AJG-AMBP;AKTS-AMCR;ALAB-AMD;ALGT-AMCR;ALKS-AMD;ALNT-AMCR;ALTG-AMD;AMAT-AMD;AMBP-AMN;AMD-AMKR;AMD-AMLX;AMD-QQQ | 1.666101 |

## Top Incoming Communities

| timestamp | lifecycle_id | stage | rotation_in_score | relative_return | volume_expansion | breadth | coherence | cross_resolution_support |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-04-02 17:30:00+00:00 | L0359 | expansion | 1.258930 | 0.001392 | 0.328673 | 0.592593 | 0.640300 | 0.018898 |
| 2026-05-06 18:45:00+00:00 | L0914 | maturity | 1.256680 | 0.002110 | 0.590415 | 0.600000 | 0.406022 | 0.007947 |
| 2026-05-05 14:00:00+00:00 | L0883 | confirmation | 1.123866 | 0.007015 | -0.270848 | 0.695652 | 0.456630 | 0.007463 |
| 2026-05-04 13:30:00+00:00 | L0843 | expansion | 1.033211 | 0.002952 | 0.068504 | 0.589744 | 0.430906 | 0.020270 |
| 2026-03-16 17:15:00+00:00 | L0051 | maturity | 1.031153 | 0.001881 | -0.174040 | 0.666667 | 0.502539 | 0.007752 |
| 2026-03-17 15:15:00+00:00 | L0080 | confirmation | 1.030662 | 0.000684 | 0.275791 | 0.608696 | 0.553041 | 0.008519 |
| 2026-05-14 19:45:00+00:00 | L1025 | maturity | 1.007850 | 0.001509 | 1.961266 | 0.666667 | 0.483336 | 0.007821 |
| 2026-05-29 17:30:00+00:00 | L1229 | confirmation | 1.006152 | 0.002114 | 0.400907 | 0.560000 | 0.442511 | 0.007692 |
| 2026-05-21 15:00:00+00:00 | L1101 | maturity | 1.000528 | 0.001663 | -0.035897 | 0.585366 | 0.605330 | 0.017021 |
| 2026-05-04 19:45:00+00:00 | L0863 | expansion | 0.981132 | 0.000678 | 1.660226 | 0.615385 | 0.437515 | 0.022642 |

## Top Outgoing Communities

| timestamp | lifecycle_id | stage | rotation_out_score | relative_return | member_outflow | edge_death_rate | breadth_delta | coherence_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-04-10 14:30:00+00:00 | L0472 | expansion | 1.882166 | -0.003974 | 47 | 0.189573 | -0.528483 | -0.036210 |
| 2026-05-28 16:30:00+00:00 | L1201 | expansion | 1.666581 | -0.001950 | 33 | 0.147059 | -0.375291 | -0.025429 |
| 2026-05-08 19:15:00+00:00 | L0943 | decay | 1.642211 | -0.001908 | 28 | 0.180723 | -0.557143 | -0.084077 |
| 2026-03-19 14:00:00+00:00 | L0120 | decay | 1.611116 | -0.002675 | 40 | 0.342657 | -0.554386 | -0.090924 |
| 2026-06-02 17:45:00+00:00 | L1265 | expansion | 1.517560 | -0.004801 | 13 | 0.151515 | -0.423077 | -0.009398 |
| 2026-03-24 19:15:00+00:00 | L0203 | confirmation | 1.445314 | -0.006011 | 14 | 0.171429 | -0.593985 | -0.049220 |
| 2026-05-27 18:30:00+00:00 | L1190 | confirmation | 1.367537 | -0.001385 | 24 | 0.111111 | -0.090116 | -0.071030 |
| 2026-05-04 17:45:00+00:00 | L0870 | confirmation | 1.339447 | -0.000922 | 29 | 0.142857 | -0.091278 | -0.000724 |
| 2026-05-15 15:45:00+00:00 | L1031 | decay | 1.338887 | -0.001900 | 24 | 0.257143 | -0.170045 | -0.072437 |
| 2026-05-01 13:30:00+00:00 | L0799 | expansion | 1.305631 | -0.010754 | 37 | 0.384058 | -0.186574 | -0.020829 |

## Additional Findings

### Short-horizon backtest observations

- The current backtest comparison remains informative but should not be over-interpreted.
- It is useful as a sanity check that communities can be translated into signals, but it is not strong enough to support a durable alpha claim.

### Strongest current conclusions

1. Non-preset intraday communities can be detected from graph structure.
2. `15m` is currently the strongest main analytical frequency.
3. `5m`, `15m`, and `30m` produce meaningfully different but comparable structures.
4. Edge persistence is a valid and learnable prediction task with `AUC = 0.7527`.
5. Community survival is promising with `AUC = 0.7816`.
6. Edge emergence is now a real predictive benchmark with `AUC = 0.9345`.
7. Community rotation can now be expressed as lifecycle-to-lifecycle structural transfer candidates.

### Remaining prototype areas

1. node migration remains weak with `AUC = 0.5025` and should not be treated as solved.
2. null significance against stronger structure-preserving shuffles remains incomplete.
3. lifecycle case validation still needs more case-by-case audit.
4. community rotation still needs richer frontend interpretation and longer-horizon case studies.

## Existing Detailed Reports

- experiment report: `D:\DEV\stocknetwork\StockNet\artifacts\final_report\experiment_report.md`
- backtest comparison: `D:\DEV\stocknetwork\StockNet\artifacts\backtest_comparison.md`
- edge emergence report: `D:\DEV\stocknetwork\StockNet\artifacts\edge_emergence_final_v2\edge_emergence_report.md`
- rotation score report: `D:\DEV\stocknetwork\StockNet\artifacts\rotation_detection_v1\rotation_score_report.md`

## Final Conclusion

> Intraday U.S. equity data does appear to contain non-preset co-evolution community structure that can be detected, compared across `5m / 15m / 30m`, partially organized into lifecycle and rotation semantics, and predicted for several tasks.

The main hypothesis is therefore supported at a strong prototype-research level.

The most reliable completed findings are:

- non-preset community discovery
- multi-resolution structure comparison
- strong edge-persistence prediction
- a first formal edge-emergence benchmark
- an initial community-rotation detection layer

The main results that still require caution are:

- full lifecycle-grade validation
- stronger null significance against structure-preserving baselines
- reliable node-migration prediction
- deeper case validation of rotation events

So the fairest complete answer to the research agenda is:

> StockNet already demonstrates that non-preset intraday market structure is detectable and partially predictable. It has not yet finished all validation required for a final academic or production-grade conclusion, but it now supports a coherent research story across community discovery, multi-resolution confirmation, lifecycle reasoning, edge emergence, and early community rotation detection.
