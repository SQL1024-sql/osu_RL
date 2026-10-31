# 20261031-1933-timing120

training maps: wide. best checkpoint: `rl_model_17500000_steps.zip` (validation acc 95.95%, 18 misses)

## Test set

| set | model_acc | model_fc | model_miss | autoplay_acc | autoplay_fc |
|---|---|---|---|---|---|
| real (26 maps) | 97.81% | 3/26 | 2.4 | 100.00% | 25/26 |
| narrow (50 maps) | 95.36% | 0/50 | 3.1 | 99.96% | 47/50 |
| wide (50 maps) | 94.24% | 0/50 | 3.1 | 99.96% | 47/50 |

acc = osu!stable-style accuracy over objects (300/100/50/miss). autoplay = scripted perfect player, the ceiling.

## Real maps

| map | objects | sliders | spinners | diff | acc | judgements | combo | lazer_acc | autoplay |
|---|---|---|---|---|---|---|---|---|---|
| 100374 | 389 | 149 | 4 | CS4 AR6 OD5 | 98.97% | 385/0/0/4 | 272/634 | 99.04% | 100.00% |
| 103019 | 281 | 129 | 6 | CS4 AR7 OD5 | 97.86% | 275/0/0/6 | 272/601 | 98.11% | 100.00% |
| 104973 | 428 | 186 | 4 | CS4 AR6 OD6 | 98.75% | 422/2/0/4 | 296/725 | 98.85% | 100.00% |
| 1124896 | 1044 | 347 | 2 | CS3.8 AR8.7 OD7.5 | 99.74% | 1041/1/0/2 | 1342/1428 | 99.76% | 100.00% |
| 112643 | 541 | 156 | 6 | CS5 AR8 OD8 | 96.89% | 519/15/1/6 | 326/785 | 96.99% | 100.00% |
| 1341554 | 541 | 297 | 0 | CS4.3 AR9.3 OD8 | 99.26% | 535/6/0/0 | 877/877 | 99.30% | 100.00% |
| 1431386 | 370 | 292 | 1 | CS3.3 AR5 OD4 | 99.73% | 369/0/0/1 | 636/781 | 99.76% | 100.00% |
| 1450162 | 260 | 155 | 1 | CS4 AR7.5 OD7 | 99.62% | 259/0/0/1 | 440/441 | 99.65% | 100.00% |
| 1597806 | 115 | 62 | 0 | CS7 AR8 OD8 | 93.62% | 104/11/0/0 | 58/181 | 93.67% | 100.00% |
| 20544 | 98 | 23 | 2 | CS5 AR2 OD2 | 97.11% | 95/0/1/2 | 123/166 | 97.19% | 100.00% |
| 2190499 | 801 | 482 | 2 | CS3.7 AR9 OD8.4 | 99.50% | 796/3/0/2 | 1292/1322 | 99.54% | 100.00% |
| 2593923 | 1110 | 302 | 0 | CS4.2 AR9.4 OD8.7 | 97.30% | 1065/45/0/0 | 1429/1429 | 97.40% | 100.00% |
| 2768615 | 143 | 60 | 2 | CS9 AR8 OD8 | 83.33% | 111/23/3/6 | 31/215 | 83.40% | 100.00% |
| 3227428 | 98 | 75 | 1 | CS3.5 AR4 OD4 | 98.98% | 97/0/0/1 | 198/199 | 99.10% | 100.00% |
| 3644427 | 1151 | 621 | 2 | CS3 AR9.2 OD8 | 99.48% | 1143/6/0/2 | 1641/2256 | 99.53% | 99.94% |
| 37902 | 192 | 52 | 3 | CS3 AR5 OD5 | 98.44% | 189/0/0/3 | 224/413 | 98.61% | 100.00% |
| 39206 | 475 | 121 | 8 | CS4 AR8 OD8 | 96.18% | 452/14/1/8 | 200/660 | 96.32% | 100.00% |
| 42587 | 429 | 147 | 4 | CS5 AR8 OD8 | 97.98% | 418/7/0/4 | 254/600 | 98.01% | 100.00% |
| 50859 | 229 | 78 | 3 | CS4 AR7 OD7 | 98.11% | 224/2/0/3 | 328/386 | 98.25% | 100.00% |
| 75858 | 338 | 130 | 1 | CS4 AR7 OD7 | 99.51% | 336/1/0/1 | 468/498 | 99.53% | 100.00% |
| 801165 | 1612 | 674 | 0 | CS4 AR9.3 OD8 | 99.63% | 1603/9/0/0 | 2359/2359 | 99.64% | 100.00% |
| 871815 | 134 | 60 | 1 | CS4 AR8.3 OD6 | 99.25% | 133/0/0/1 | 235/236 | 99.32% | 100.00% |
| bakenohana | 569 | 351 | 0 | CS4 AR9.6 OD9.2 | 95.20% | 528/41/0/0 | 672/1003 | 95.56% | 100.00% |
| renatus_hard | 558 | 396 | 2 | CS4 AR7.5 OD6 | 99.64% | 556/0/0/2 | 764/1134 | 99.68% | 100.00% |
| renatus_insane | 897 | 309 | 2 | CS4 AR9 OD8 | 99.63% | 893/2/0/2 | 905/1328 | 99.65% | 100.00% |
| renatus_normal | 278 | 209 | 2 | CS3 AR4 OD3 | 99.28% | 276/0/0/2 | 546/804 | 99.41% | 100.00% |

judgements = 300/100/50/miss. lazer_acc = accuracy osu!lazer shows for the same replay (it also weights slider ticks).

## Simulator vs osu!lazer

27407/27409 judgements identical on the real-map replays.

## Checkpoints (validation)

| ckpt | acc | miss |
|---|---|---|
| rl_model_500000_steps.zip | 12.03% | 376 |
| rl_model_1000000_steps.zip | 45.40% | 70 |
| rl_model_1500000_steps.zip | 90.81% | 20 |
| rl_model_2000000_steps.zip | 93.63% | 18 |
| rl_model_2500000_steps.zip | 93.03% | 22 |
| rl_model_3000000_steps.zip | 95.09% | 19 |
| rl_model_3500000_steps.zip | 94.99% | 21 |
| rl_model_4000000_steps.zip | 95.45% | 18 |
| rl_model_4500000_steps.zip | 95.29% | 18 |
| rl_model_5000000_steps.zip | 95.03% | 18 |
| rl_model_5500000_steps.zip | 95.06% | 19 |
| rl_model_6000000_steps.zip | 94.86% | 18 |
| rl_model_6500000_steps.zip | 95.29% | 18 |
| rl_model_7000000_steps.zip | 94.63% | 19 |
| rl_model_7500000_steps.zip | 94.06% | 21 |
| rl_model_8000000_steps.zip | 94.76% | 18 |
| rl_model_8500000_steps.zip | 93.27% | 20 |
| rl_model_9000000_steps.zip | 94.40% | 18 |
| rl_model_9500000_steps.zip | 94.56% | 18 |
| rl_model_10000000_steps.zip | 95.39% | 18 |
| rl_model_10500000_steps.zip | 94.72% | 18 |
| rl_model_11000000_steps.zip | 94.62% | 18 |
| rl_model_11500000_steps.zip | 95.55% | 18 |
| rl_model_12000000_steps.zip | 95.09% | 18 |
| rl_model_12500000_steps.zip | 95.09% | 18 |
| rl_model_13000000_steps.zip | 94.99% | 18 |
| rl_model_13500000_steps.zip | 95.06% | 19 |
| rl_model_14000000_steps.zip | 95.32% | 18 |
| rl_model_14500000_steps.zip | 95.19% | 18 |
| rl_model_15000000_steps.zip | 94.89% | 18 |
| rl_model_15500000_steps.zip | 95.36% | 19 |
| rl_model_16000000_steps.zip | 95.36% | 19 |
| rl_model_16500000_steps.zip | 95.29% | 18 |
| rl_model_17000000_steps.zip | 95.82% | 18 |
| rl_model_17500000_steps.zip | 95.95% | 18 |
| rl_model_18000000_steps.zip | 95.23% | 21 |
| rl_model_18500000_steps.zip | 94.14% | 23 |
| rl_model_19000000_steps.zip | 94.73% | 20 |
| rl_model_19500000_steps.zip | 94.89% | 19 |
| rl_model_20000000_steps.zip | 95.06% | 19 |
| model.zip | 95.06% | 19 |
