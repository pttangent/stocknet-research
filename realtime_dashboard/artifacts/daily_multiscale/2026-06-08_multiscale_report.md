# 2026-06-08 Intraday Multiscale Community Scan

Using local formal `1m` archive and derived `5m` / `15m` aggregation from the same `1m` source.

## Summary

- Trade date: 2026-06-08
- 1m communities: 8
- 5m communities: 4
- 15m communities: 4

## Top 1m Communities

| Community | Radar | Members | Top Members | Breadth | Coherence | Rel Return |
|---|---:|---:|---|---:|---:|---:|
| C005 | 0.6481 | 26 | ASX,BE,BABA,APLD,BB | 0.4615 | 0.4398 | 0.0002 |
| C002 | 0.6405 | 131 | APXT,BGSF,BEAG,ASPS,AIV | 0.4122 | 0.5229 | -0.0003 |
| C001 | 0.5702 | 101 | ALX,ANPA,ACOG,AVAH,AZ | 0.4653 | 0.4079 | 0.0003 |
| C006 | 0.5236 | 33 | ARXS,RTX,AIG,AMWL,AXR | 0.3333 | 0.4247 | -0.0003 |

## Top 5m Communities

| Community | Radar | Members | Top Members | Breadth | Coherence | Rel Return |
|---|---:|---:|---|---:|---:|---:|
| C002 | 0.6340 | 134 | AMPX,AA,AEHR,ASTS,AXTI | 0.4851 | 0.4334 | -0.0000 |
| C001 | 0.5881 | 288 | BGSF,APXT,BEAG,ASPS,AKO-B | 0.4236 | 0.4215 | -0.0000 |
| C003 | 0.4707 | 47 | BAC,ATI,APG,BEN,BKU | 0.4043 | 0.4138 | -0.0001 |
| C004 | 0.3126 | 36 | SMR,MU,MS,MSFT,META | 0.1944 | 0.4217 | -0.0019 |

## Top 15m Communities

| Community | Radar | Members | Top Members | Breadth | Coherence | Rel Return |
|---|---:|---:|---|---:|---:|---:|
| C004 | 0.5970 | 185 | AKO-B,AGCC,ASPS,AXR,ALX | 0.5189 | 0.4302 | 0.0006 |
| C003 | 0.5897 | 95 | ARCC,AXON,AMRC,AIRJ,ADBE | 0.5263 | 0.5019 | 0.0001 |
| C001 | 0.4566 | 84 | ASR,ANDG,AGX,ANF,BBWI | 0.4762 | 0.4491 | 0.0001 |
| C002 | 0.3585 | 138 | TSLA,ASX,AMD,ASML,AVGO | 0.3841 | 0.4435 | -0.0004 |

## Cross-Scale Overlap

| Left | Right | Jaccard | Overlap | Shared Symbols |
|---|---|---:|---:|---|
| five_minute:C001 | fifteen_minute:C004 | 0.5662 | 171 | AAPL,AARD,AAT,AAUC,ABEV,ABG,ABX,ACA,ACB,ACCO,ACEL,ACIC,ACNB,ACNT,ACOG,ACR,ACTG,ACU,ADAG,ADC |
| one_minute:C008 | five_minute:C004 | 0.3333 | 15 | ASPI,BEKE,CCJ,GOOGL,GS,JPM,META,MRVL,MSFT,MU,NVDA,OKLO,SMR,TSLA,TSM |
| five_minute:C002 | fifteen_minute:C002 | 0.3204 | 66 | AA,AAOI,ABAT,ABBV,ABSI,ACHC,ACHR,ACMR,ADI,ADP,AEHR,AEM,AFL,AG,AGI,AIOT,AIP,AIRO,ALAB,ALGM |
| five_minute:C002 | fifteen_minute:C003 | 0.3011 | 53 | ABCL,ABR,ABVX,ACDC,ADBE,ADEA,ADMA,ADPT,ADTN,AEIS,AEO,AEVA,AFRM,AGNC,AGNT,AHCO,AI,AIRJ,AKAM,ALB |
| one_minute:C002 | five_minute:C001 | 0.2075 | 72 | AAUC,ABUS,ACI,ACR,ACRE,ACT,ACTG,ADAG,ADNT,ADUR,ADUS,AEG,AER,AESI,AFG,AGBK,AGIO,AGMB,AGO,AGRO |
| one_minute:C003 | five_minute:C001 | 0.2070 | 65 | AARD,AAT,ABG,ABM,ACCO,ACIC,ACLS,ACNB,ACNT,ADC,AES,AESPU,AEYE,AFBI,AFCG,AFYA,AGCC,AHRT,AIAI,AIRG |
| one_minute:C001 | five_minute:C001 | 0.1969 | 64 | A,AAPL,ABX,ACET,ACM,ACOG,ADAM,ADM,AEC,AGL,AGPU,AIN,AJG,ALCO,ALGN,ALHC,ALMR,ALTG,ALX,AMCX |
| one_minute:C005 | five_minute:C002 | 0.1940 | 26 | ABAT,ACHR,ADI,AG,AI,ALAB,AMAT,AMD,AMKR,AMPX,ANET,APH,APLD,ARM,ASML,ASST,ASTS,ASX,AUR,AVGO |
| one_minute:C003 | fifteen_minute:C004 | 0.1897 | 44 | AARD,AAT,ABG,ACCO,ACIC,ACNB,ACNT,ADC,AEYE,AFBI,AFCG,AFYA,AGCC,AHRT,AIRG,ALAR,ALH,ALNT,ALOT,ALSN |
| one_minute:C001 | fifteen_minute:C004 | 0.1770 | 43 | AAPL,ABX,ACOG,AEC,AGL,AGPU,AIN,ALCO,ALGN,ALMR,ALX,AMCX,AMPH,AMSF,AMTB,AN,ANGO,ANL,ANNX,ANPA |
| five_minute:C004 | fifteen_minute:C002 | 0.1757 | 26 | ALL,ALV,AMG,APP,ASPI,AXIA,BBBY,BCS,BCSF,BEKE,BLK,CCJ,GOOGL,GS,JPM,LMT,META,MRVL,MS,MSFT |
| one_minute:C002 | five_minute:C002 | 0.1674 | 38 | AA,ABBV,ABCL,ABVX,ACDC,ACN,AEHR,AEM,AGI,AGX,ALB,ALM,AMBA,AMPG,APAM,APLE,APTV,ARCC,ARES,ARGX |

## Rotation Read

- The strongest short-horizon radar cluster is the 1m `C005` group, led by `ASX, BE, BABA, APLD, BB` with a broader member list that includes `AMD, AVGO, AMAT, ASML, ARM, ANET, AI, BBAI, ASTS`. This reads like an AI infrastructure / semicap / datacenter-adjacent burst rather than a defensive basket.
- That same growth-tech / infrastructure complex survives into 5m `C002` and 15m `C002`, where `TSLA, ASX, AMD, ASML, AVGO, NVDA, TSM, MRVL, MU, SMR, OKLO, CCJ` coexist. This is the clearest multiscale continuation signal in the scan.
- A second broad mixed-value / cyclicals cluster dominates 5m `C001` and 15m `C004`. It is much larger in member count, but its composition is noisy and cross-sector, so it looks more like broad market co-movement than a clean thematic leadership group.
- The nuclear / power / high-beta hardware sleeve (`SMR, OKLO, CCJ`, sometimes with `NVDA, MU, MRVL`) appears inside the 1m `C008` -> 5m `C004` -> 15m `C002` path, suggesting attention is rotating from pure semis into the power-and-enablement side of the AI trade rather than into classic defensives.
- Relative returns at the close are small, so this is better read as structure/leadership evidence than as a strong close-to-close momentum signal.
