# 20261101-0021-pool120

training maps: wide. best checkpoint: `rl_model_11500000_steps.zip` (validation acc 99.39%, 1 misses)

## Test set

| set | model_acc | model_fc | model_miss | autoplay_acc | autoplay_fc |
|---|---|---|---|---|---|
| real (26 maps) | 98.98% | 12/26 | 0.7 | 100.00% | 25/26 |
| narrow (50 maps) | 98.71% | 20/50 | 0.5 | 99.96% | 47/50 |
| wide (50 maps) | 97.28% | 15/50 | 1.1 | 99.96% | 47/50 |

acc = osu!stable-style accuracy over objects (300/100/50/miss). autoplay = scripted perfect player, the ceiling.

## Real maps

| map | objects | sliders | spinners | diff | acc | judgements | combo | lazer_acc | autoplay |
|---|---|---|---|---|---|---|---|---|---|
| 100374 | 389 | 149 | 4 | CS4 AR6 OD5 | 100.00% | 389/0/0/0 | 634/634 | 100.00% | 100.00% |
| 103019 | 281 | 129 | 6 | CS4 AR7 OD5 | 100.00% | 281/0/0/0 | 601/601 | 100.00% | 100.00% |
| 104973 | 428 | 186 | 4 | CS4 AR6 OD6 | 99.65% | 426/1/1/0 | 441/725 | 99.66% | 100.00% |
| 1124896 | 1044 | 347 | 2 | CS3.8 AR8.7 OD7.5 | 99.90% | 1043/0/0/1 | 1427/1428 | 99.91% | 100.00% |
| 112643 | 541 | 156 | 6 | CS5 AR8 OD8 | 99.88% | 540/1/0/0 | 670/785 | 99.87% | 100.00% |
| 1341554 | 541 | 297 | 0 | CS4.3 AR9.3 OD8 | 98.40% | 529/10/0/2 | 540/877 | 98.41% | 100.00% |
| 1431386 | 370 | 292 | 1 | CS3.3 AR5 OD4 | 99.55% | 368/1/0/1 | 460/781 | 99.60% | 100.00% |
| 1450162 | 260 | 155 | 1 | CS4 AR7.5 OD7 | 100.00% | 260/0/0/0 | 441/441 | 100.00% | 100.00% |
| 1597806 | 115 | 62 | 0 | CS7 AR8 OD8 | 100.00% | 115/0/0/0 | 181/181 | 100.00% | 100.00% |
| 20544 | 98 | 23 | 2 | CS5 AR2 OD2 | 99.32% | 97/1/0/0 | 127/166 | 99.27% | 100.00% |
| 2190499 | 801 | 482 | 2 | CS3.7 AR9 OD8.4 | 99.33% | 793/8/0/0 | 1322/1322 | 99.36% | 100.00% |
| 2593923 | 1110 | 302 | 0 | CS4.2 AR9.4 OD8.7 | 98.86% | 1091/19/0/0 | 821/1429 | 98.86% | 100.00% |
| 2768615 | 143 | 60 | 2 | CS9 AR8 OD8 | 88.34% | 122/12/2/7 | 37/215 | 87.60% | 100.00% |
| 3227428 | 98 | 75 | 1 | CS3.5 AR4 OD4 | 98.30% | 96/1/0/1 | 160/199 | 98.40% | 100.00% |
| 3644427 | 1151 | 621 | 2 | CS3 AR9.2 OD8 | 99.33% | 1141/7/0/3 | 1354/2256 | 99.36% | 99.94% |
| 37902 | 192 | 52 | 3 | CS3 AR5 OD5 | 99.48% | 191/0/0/1 | 312/413 | 99.54% | 100.00% |
| 39206 | 475 | 121 | 8 | CS4 AR8 OD8 | 98.91% | 468/5/1/1 | 293/660 | 98.91% | 100.00% |
| 42587 | 429 | 147 | 4 | CS5 AR8 OD8 | 99.69% | 427/2/0/0 | 600/600 | 99.50% | 100.00% |
| 50859 | 229 | 78 | 3 | CS4 AR7 OD7 | 98.98% | 226/2/0/1 | 328/386 | 99.06% | 100.00% |
| 75858 | 338 | 130 | 1 | CS4 AR7 OD7 | 100.00% | 338/0/0/0 | 498/498 | 100.00% | 100.00% |
| 801165 | 1612 | 674 | 0 | CS4 AR9.3 OD8 | 98.92% | 1586/26/0/0 | 2359/2359 | 98.98% | 100.00% |
| 871815 | 134 | 60 | 1 | CS4 AR8.3 OD6 | 99.50% | 133/1/0/0 | 236/236 | 99.52% | 100.00% |
| bakenohana | 569 | 351 | 0 | CS4 AR9.6 OD9.2 | 97.77% | 550/19/0/0 | 1003/1003 | 98.11% | 100.00% |
| renatus_hard | 558 | 396 | 2 | CS4 AR7.5 OD6 | 100.00% | 558/0/0/0 | 1134/1134 | 100.00% | 100.00% |
| renatus_insane | 897 | 309 | 2 | CS4 AR9 OD8 | 99.55% | 891/6/0/0 | 891/1328 | 99.55% | 100.00% |
| renatus_normal | 278 | 209 | 2 | CS3 AR4 OD3 | 99.76% | 277/1/0/0 | 804/804 | 99.80% | 100.00% |

judgements = 300/100/50/miss. lazer_acc = accuracy osu!lazer shows for the same replay (it also weights slider ticks).

## Simulator vs osu!lazer

27394/27409 judgements identical on the real-map replays.

## Checkpoints (validation)

| ckpt | acc | miss |
|---|---|---|
| rl_model_500000_steps.zip | 11.73% | 659 |
| rl_model_1000000_steps.zip | 73.57% | 44 |
| rl_model_1500000_steps.zip | 96.30% | 19 |
| rl_model_2000000_steps.zip | 96.93% | 20 |
| rl_model_2500000_steps.zip | 97.36% | 17 |
| rl_model_3000000_steps.zip | 96.33% | 21 |
| rl_model_3500000_steps.zip | 96.99% | 19 |
| rl_model_4000000_steps.zip | 96.21% | 25 |
| rl_model_4500000_steps.zip | 96.97% | 18 |
| rl_model_5000000_steps.zip | 97.31% | 18 |
| rl_model_5500000_steps.zip | 97.40% | 12 |
| rl_model_6000000_steps.zip | 97.93% | 15 |
| rl_model_6500000_steps.zip | 96.28% | 18 |
| rl_model_7000000_steps.zip | 96.33% | 14 |
| rl_model_7500000_steps.zip | 97.06% | 16 |
| rl_model_8000000_steps.zip | 98.58% | 0 |
| rl_model_8500000_steps.zip | 98.20% | 1 |
| rl_model_9000000_steps.zip | 98.61% | 0 |
| rl_model_9500000_steps.zip | 98.95% | 4 |
| rl_model_10000000_steps.zip | 99.32% | 2 |
| rl_model_10500000_steps.zip | 99.09% | 3 |
| rl_model_11000000_steps.zip | 98.99% | 1 |
| rl_model_11500000_steps.zip | 99.39% | 1 |
| rl_model_12000000_steps.zip | 99.04% | 2 |
| rl_model_12500000_steps.zip | 99.07% | 3 |
| rl_model_13000000_steps.zip | 99.14% | 2 |
| rl_model_13500000_steps.zip | 99.02% | 2 |
| rl_model_14000000_steps.zip | 98.92% | 1 |
| rl_model_14500000_steps.zip | 98.59% | 2 |
| rl_model_15000000_steps.zip | 96.06% | 19 |
| rl_model_15500000_steps.zip | 95.99% | 3 |
| rl_model_16000000_steps.zip | 97.18% | 5 |
| rl_model_16500000_steps.zip | 96.00% | 10 |
| rl_model_17000000_steps.zip | 94.53% | 28 |
| rl_model_17500000_steps.zip | 96.17% | 16 |
| rl_model_18000000_steps.zip | 97.82% | 3 |
| rl_model_18500000_steps.zip | 97.83% | 5 |
| rl_model_19000000_steps.zip | 97.51% | 9 |
| rl_model_19500000_steps.zip | 97.33% | 10 |
| rl_model_20000000_steps.zip | 97.29% | 7 |
| model.zip | 97.46% | 6 |
