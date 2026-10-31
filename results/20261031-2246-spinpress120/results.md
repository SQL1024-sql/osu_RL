# 20261031-2246-spinpress120

training maps: wide. best checkpoint: `rl_model_9500000_steps.zip` (validation acc 98.58%, 0 misses)

## Test set

| set | model_acc | model_fc | model_miss | autoplay_acc | autoplay_fc |
|---|---|---|---|---|---|
| real (26 maps) | 98.27% | 10/26 | 1.6 | 100.00% | 25/26 |
| narrow (50 maps) | 99.16% | 25/50 | 0.2 | 99.96% | 47/50 |
| wide (50 maps) | 98.60% | 19/50 | 0.3 | 99.96% | 47/50 |

acc = osu!stable-style accuracy over objects (300/100/50/miss). autoplay = scripted perfect player, the ceiling.

## Real maps

| map | objects | sliders | spinners | diff | acc | judgements | combo | lazer_acc | autoplay |
|---|---|---|---|---|---|---|---|---|---|
| 100374 | 389 | 149 | 4 | CS4 AR6 OD5 | 97.94% | 377/12/0/0 | 206/634 | 97.80% | 100.00% |
| 103019 | 281 | 129 | 6 | CS4 AR7 OD5 | 99.47% | 279/1/1/0 | 287/601 | 99.45% | 100.00% |
| 104973 | 428 | 186 | 4 | CS4 AR6 OD6 | 99.38% | 425/1/0/2 | 717/725 | 99.57% | 100.00% |
| 1124896 | 1044 | 347 | 2 | CS3.8 AR8.7 OD7.5 | 99.90% | 1043/0/0/1 | 1427/1428 | 99.91% | 100.00% |
| 112643 | 541 | 156 | 6 | CS5 AR8 OD8 | 99.72% | 539/1/1/0 | 526/785 | 99.72% | 100.00% |
| 1341554 | 541 | 297 | 0 | CS4.3 AR9.3 OD8 | 99.51% | 537/4/0/0 | 567/877 | 99.51% | 100.00% |
| 1431386 | 370 | 292 | 1 | CS3.3 AR5 OD4 | 99.28% | 366/4/0/0 | 300/781 | 99.30% | 100.00% |
| 1450162 | 260 | 155 | 1 | CS4 AR7.5 OD7 | 100.00% | 260/0/0/0 | 441/441 | 100.00% | 100.00% |
| 1597806 | 115 | 62 | 0 | CS7 AR8 OD8 | 99.42% | 114/1/0/0 | 181/181 | 99.43% | 100.00% |
| 20544 | 98 | 23 | 2 | CS5 AR2 OD2 | 99.32% | 97/1/0/0 | 166/166 | 99.34% | 100.00% |
| 2190499 | 801 | 482 | 2 | CS3.7 AR9 OD8.4 | 99.63% | 797/2/2/0 | 984/1322 | 99.63% | 100.00% |
| 2593923 | 1110 | 302 | 0 | CS4.2 AR9.4 OD8.7 | 99.88% | 1108/2/0/0 | 1429/1429 | 99.88% | 100.00% |
| 2768615 | 143 | 60 | 2 | CS9 AR8 OD8 | 77.51% | 108/8/1/26 | 22/215 | 77.55% | 100.00% |
| 3227428 | 98 | 75 | 1 | CS3.5 AR4 OD4 | 91.50% | 86/11/0/1 | 39/199 | 91.47% | 100.00% |
| 3644427 | 1151 | 621 | 2 | CS3 AR9.2 OD8 | 99.61% | 1145/4/1/1 | 1025/2256 | 99.63% | 99.94% |
| 37902 | 192 | 52 | 3 | CS3 AR5 OD5 | 99.13% | 190/1/0/1 | 312/413 | 99.18% | 100.00% |
| 39206 | 475 | 121 | 8 | CS4 AR8 OD8 | 97.68% | 463/1/4/7 | 144/660 | 97.37% | 100.00% |
| 42587 | 429 | 147 | 4 | CS5 AR8 OD8 | 98.68% | 422/3/2/2 | 361/600 | 98.35% | 100.00% |
| 50859 | 229 | 78 | 3 | CS4 AR7 OD7 | 98.69% | 225/3/0/1 | 205/386 | 98.71% | 100.00% |
| 75858 | 338 | 130 | 1 | CS4 AR7 OD7 | 99.61% | 336/2/0/0 | 437/498 | 99.60% | 100.00% |
| 801165 | 1612 | 674 | 0 | CS4 AR9.3 OD8 | 99.83% | 1608/4/0/0 | 2359/2359 | 99.88% | 100.00% |
| 871815 | 134 | 60 | 1 | CS4 AR8.3 OD6 | 100.00% | 134/0/0/0 | 236/236 | 100.00% | 100.00% |
| bakenohana | 569 | 351 | 0 | CS4 AR9.6 OD9.2 | 99.41% | 564/5/0/0 | 1003/1003 | 99.44% | 100.00% |
| renatus_hard | 558 | 396 | 2 | CS4 AR7.5 OD6 | 100.00% | 558/0/0/0 | 1134/1134 | 100.00% | 100.00% |
| renatus_insane | 897 | 309 | 2 | CS4 AR9 OD8 | 99.93% | 896/1/0/0 | 1328/1328 | 99.93% | 100.00% |
| renatus_normal | 278 | 209 | 2 | CS3 AR4 OD3 | 100.00% | 278/0/0/0 | 804/804 | 100.00% | 100.00% |

judgements = 300/100/50/miss. lazer_acc = accuracy osu!lazer shows for the same replay (it also weights slider ticks).

## Simulator vs osu!lazer

27397/27409 judgements identical on the real-map replays.

## Checkpoints (validation)

| ckpt | acc | miss |
|---|---|---|
| rl_model_500000_steps.zip | 13.85% | 317 |
| rl_model_1000000_steps.zip | 89.98% | 18 |
| rl_model_1500000_steps.zip | 93.71% | 25 |
| rl_model_2000000_steps.zip | 95.39% | 18 |
| rl_model_2500000_steps.zip | 96.12% | 18 |
| rl_model_3000000_steps.zip | 95.75% | 19 |
| rl_model_3500000_steps.zip | 96.02% | 18 |
| rl_model_4000000_steps.zip | 95.58% | 20 |
| rl_model_4500000_steps.zip | 95.25% | 18 |
| rl_model_5000000_steps.zip | 95.23% | 17 |
| rl_model_5500000_steps.zip | 95.43% | 18 |
| rl_model_6000000_steps.zip | 95.02% | 19 |
| rl_model_6500000_steps.zip | 95.92% | 16 |
| rl_model_7000000_steps.zip | 95.36% | 14 |
| rl_model_7500000_steps.zip | 95.35% | 17 |
| rl_model_8000000_steps.zip | 95.15% | 13 |
| rl_model_8500000_steps.zip | 95.96% | 11 |
| rl_model_9000000_steps.zip | 96.66% | 6 |
| rl_model_9500000_steps.zip | 98.58% | 0 |
| rl_model_10000000_steps.zip | 98.51% | 0 |
| rl_model_10500000_steps.zip | 97.02% | 7 |
| rl_model_11000000_steps.zip | 98.51% | 0 |
| rl_model_11500000_steps.zip | 96.50% | 3 |
| rl_model_12000000_steps.zip | 97.05% | 3 |
| rl_model_12500000_steps.zip | 97.75% | 2 |
| rl_model_13000000_steps.zip | 96.52% | 4 |
| rl_model_13500000_steps.zip | 97.54% | 1 |
| rl_model_14000000_steps.zip | 95.85% | 3 |
| rl_model_14500000_steps.zip | 96.36% | 3 |
| rl_model_15000000_steps.zip | 96.75% | 4 |
| rl_model_15500000_steps.zip | 97.11% | 2 |
| rl_model_16000000_steps.zip | 97.05% | 2 |
| rl_model_16500000_steps.zip | 97.12% | 3 |
| rl_model_17000000_steps.zip | 97.15% | 2 |
| rl_model_17500000_steps.zip | 96.48% | 2 |
| rl_model_18000000_steps.zip | 96.55% | 3 |
| rl_model_18500000_steps.zip | 95.63% | 7 |
| rl_model_19000000_steps.zip | 96.22% | 6 |
| rl_model_19500000_steps.zip | 97.25% | 3 |
| rl_model_20000000_steps.zip | 97.48% | 2 |
| model.zip | 97.48% | 2 |
