# 20261031-1806-env32

training maps: wide. best checkpoint: `rl_model_9500000_steps.zip` (validation acc 94.93%, 18 misses)

## Test set

| set | model_acc | model_fc | model_miss | autoplay_acc | autoplay_fc |
|---|---|---|---|---|---|
| real (26 maps) | 95.63% | 0/26 | 4.5 | 100.00% | 25/26 |
| narrow (50 maps) | 94.24% | 0/50 | 3.2 | 99.96% | 47/50 |
| wide (50 maps) | 92.24% | 0/50 | 3.6 | 99.96% | 47/50 |

acc = osu!stable-style accuracy over objects (300/100/50/miss). autoplay = scripted perfect player, the ceiling.

## Real maps

| map | objects | sliders | spinners | diff | acc | judgements | combo | lazer_acc | autoplay |
|---|---|---|---|---|---|---|---|---|---|
| 100374 | 389 | 149 | 4 | CS4 AR6 OD5 | 98.29% | 381/4/0/4 | 272/634 | 98.41% | 100.00% |
| 103019 | 281 | 129 | 6 | CS4 AR7 OD5 | 97.63% | 274/1/0/6 | 272/601 | 97.90% | 100.00% |
| 104973 | 428 | 186 | 4 | CS4 AR6 OD6 | 96.50% | 408/15/0/5 | 174/725 | 96.59% | 100.00% |
| 1124896 | 1044 | 347 | 2 | CS3.8 AR8.7 OD7.5 | 99.17% | 1032/10/0/2 | 1342/1428 | 99.21% | 100.00% |
| 112643 | 541 | 156 | 6 | CS5 AR8 OD8 | 97.29% | 522/13/0/6 | 248/785 | 97.36% | 100.00% |
| 1341554 | 541 | 297 | 0 | CS4.3 AR9.3 OD8 | 98.34% | 528/12/0/1 | 775/877 | 98.39% | 100.00% |
| 1431386 | 370 | 292 | 1 | CS3.3 AR5 OD4 | 99.32% | 367/1/1/1 | 636/781 | 99.52% | 100.00% |
| 1450162 | 260 | 155 | 1 | CS4 AR7.5 OD7 | 99.10% | 257/2/0/1 | 290/441 | 99.11% | 100.00% |
| 1597806 | 115 | 62 | 0 | CS7 AR8 OD8 | 96.09% | 109/4/1/1 | 148/181 | 96.28% | 100.00% |
| 20544 | 98 | 23 | 2 | CS5 AR2 OD2 | 95.24% | 92/4/0/2 | 82/166 | 95.39% | 100.00% |
| 2190499 | 801 | 482 | 2 | CS3.7 AR9 OD8.4 | 98.67% | 786/13/0/2 | 1108/1322 | 98.76% | 100.00% |
| 2593923 | 1110 | 302 | 0 | CS4.2 AR9.4 OD8.7 | 94.41% | 1018/90/0/2 | 949/1429 | 94.62% | 100.00% |
| 2768615 | 143 | 60 | 2 | CS9 AR8 OD8 | 50.70% | 61/32/5/45 | 19/215 | 51.77% | 100.00% |
| 3227428 | 98 | 75 | 1 | CS3.5 AR4 OD4 | 98.30% | 96/1/0/1 | 198/199 | 98.49% | 100.00% |
| 3644427 | 1151 | 621 | 2 | CS3 AR9.2 OD8 | 99.20% | 1139/8/1/3 | 1331/2256 | 99.27% | 99.94% |
| 37902 | 192 | 52 | 3 | CS3 AR5 OD5 | 96.61% | 184/4/1/3 | 96/413 | 96.74% | 100.00% |
| 39206 | 475 | 121 | 8 | CS4 AR8 OD8 | 95.12% | 445/20/1/9 | 101/660 | 95.12% | 100.00% |
| 42587 | 429 | 147 | 4 | CS5 AR8 OD8 | 98.14% | 419/6/0/4 | 266/600 | 98.20% | 100.00% |
| 50859 | 229 | 78 | 3 | CS4 AR7 OD7 | 93.16% | 209/12/2/6 | 248/386 | 93.66% | 100.00% |
| 75858 | 338 | 130 | 1 | CS4 AR7 OD7 | 97.49% | 326/10/1/1 | 468/498 | 97.63% | 100.00% |
| 801165 | 1612 | 674 | 0 | CS4 AR9.3 OD8 | 99.14% | 1592/18/1/1 | 1339/2359 | 99.17% | 100.00% |
| 871815 | 134 | 60 | 1 | CS4 AR8.3 OD6 | 99.25% | 133/0/0/1 | 235/236 | 99.32% | 100.00% |
| bakenohana | 569 | 351 | 0 | CS4 AR9.6 OD9.2 | 92.50% | 507/57/2/3 | 408/1003 | 92.98% | 100.00% |
| renatus_hard | 558 | 396 | 2 | CS4 AR7.5 OD6 | 99.64% | 556/0/0/2 | 764/1134 | 99.68% | 100.00% |
| renatus_insane | 897 | 309 | 2 | CS4 AR9 OD8 | 99.26% | 888/7/0/2 | 905/1328 | 99.29% | 100.00% |
| renatus_normal | 278 | 209 | 2 | CS3 AR4 OD3 | 97.84% | 271/1/4/2 | 353/804 | 98.04% | 100.00% |

judgements = 300/100/50/miss. lazer_acc = accuracy osu!lazer shows for the same replay (it also weights slider ticks).

## Simulator vs osu!lazer

27403/27409 judgements identical on the real-map replays.

## Checkpoints (validation)

| ckpt | acc | miss |
|---|---|---|
| rl_model_500000_steps.zip | 13.03% | 299 |
| rl_model_1000000_steps.zip | 48.12% | 172 |
| rl_model_1500000_steps.zip | 70.15% | 105 |
| rl_model_2000000_steps.zip | 89.68% | 26 |
| rl_model_2500000_steps.zip | 89.66% | 24 |
| rl_model_3000000_steps.zip | 93.66% | 19 |
| rl_model_3500000_steps.zip | 92.17% | 20 |
| rl_model_4000000_steps.zip | 93.13% | 22 |
| rl_model_4500000_steps.zip | 90.55% | 24 |
| rl_model_5000000_steps.zip | 93.09% | 21 |
| rl_model_5500000_steps.zip | 93.10% | 22 |
| rl_model_6000000_steps.zip | 93.59% | 18 |
| rl_model_6500000_steps.zip | 93.83% | 18 |
| rl_model_7000000_steps.zip | 93.77% | 19 |
| rl_model_7500000_steps.zip | 94.16% | 18 |
| rl_model_8000000_steps.zip | 93.27% | 19 |
| rl_model_8500000_steps.zip | 92.77% | 19 |
| rl_model_9000000_steps.zip | 94.80% | 18 |
| rl_model_9500000_steps.zip | 94.93% | 18 |
| rl_model_10000000_steps.zip | 94.64% | 19 |
| rl_model_10500000_steps.zip | 94.34% | 19 |
| rl_model_11000000_steps.zip | 94.56% | 19 |
| rl_model_11500000_steps.zip | 94.07% | 21 |
| rl_model_12000000_steps.zip | 93.57% | 20 |
| rl_model_12500000_steps.zip | 92.31% | 19 |
| rl_model_13000000_steps.zip | 94.56% | 18 |
| rl_model_13500000_steps.zip | 94.30% | 18 |
| rl_model_14000000_steps.zip | 93.11% | 18 |
| rl_model_14500000_steps.zip | 93.04% | 21 |
| rl_model_15000000_steps.zip | 92.71% | 23 |
| rl_model_15500000_steps.zip | 92.94% | 19 |
| rl_model_16000000_steps.zip | 92.44% | 24 |
| rl_model_16500000_steps.zip | 92.22% | 23 |
| rl_model_17000000_steps.zip | 93.24% | 19 |
| rl_model_17500000_steps.zip | 93.93% | 18 |
| rl_model_18000000_steps.zip | 93.97% | 19 |
| rl_model_18500000_steps.zip | 92.47% | 20 |
| rl_model_19000000_steps.zip | 93.37% | 19 |
| rl_model_19500000_steps.zip | 94.01% | 19 |
| rl_model_20000000_steps.zip | 94.00% | 19 |
| model.zip | 93.87% | 19 |
