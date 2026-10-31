# 20261031-0226-full120

training maps: narrow. best checkpoint: `rl_model_15500000_steps.zip` (validation acc 95.13%, 19 misses)

## Test set

| set | model_acc | model_fc | model_miss | autoplay_acc | autoplay_fc |
|---|---|---|---|---|---|
| real (26 maps) | 93.16% | 0/26 | 13.6 | 100.00% | 25/26 |
| narrow (50 maps) | 94.48% | 0/50 | 3.4 | 99.96% | 47/50 |
| wide (50 maps) | 91.38% | 0/50 | 5.1 | 99.96% | 47/50 |

acc = osu!stable-style accuracy over objects (300/100/50/miss). autoplay = scripted perfect player, the ceiling.

## Real maps

| map | objects | sliders | spinners | diff | acc | judgements | combo | lazer_acc | autoplay |
|---|---|---|---|---|---|---|---|---|---|
| 100374 | 389 | 149 | 4 | CS4 AR6 OD5 | 98.03% | 380/4/0/5 | 160/634 | 98.07% | 100.00% |
| 103019 | 281 | 129 | 6 | CS4 AR7 OD5 | 97.33% | 273/1/1/6 | 196/601 | 97.59% | 100.00% |
| 104973 | 428 | 186 | 4 | CS4 AR6 OD6 | 96.11% | 410/4/0/14 | 173/725 | 96.39% | 100.00% |
| 1124896 | 1044 | 347 | 2 | CS3.8 AR8.7 OD7.5 | 99.65% | 1040/1/0/3 | 1115/1428 | 99.65% | 100.00% |
| 112643 | 541 | 156 | 6 | CS5 AR8 OD8 | 95.47% | 509/19/7/6 | 292/785 | 95.35% | 100.00% |
| 1341554 | 541 | 297 | 0 | CS4.3 AR9.3 OD8 | 92.54% | 491/28/2/20 | 108/877 | 92.22% | 100.00% |
| 1431386 | 370 | 292 | 1 | CS3.3 AR5 OD4 | 97.25% | 357/8/1/4 | 307/781 | 97.41% | 100.00% |
| 1450162 | 260 | 155 | 1 | CS4 AR7.5 OD7 | 98.59% | 255/4/0/1 | 387/441 | 98.63% | 100.00% |
| 1597806 | 115 | 62 | 0 | CS7 AR8 OD8 | 90.72% | 100/13/0/2 | 58/181 | 91.08% | 100.00% |
| 20544 | 98 | 23 | 2 | CS5 AR2 OD2 | 79.25% | 76/4/2/16 | 27/166 | 79.79% | 100.00% |
| 2190499 | 801 | 482 | 2 | CS3.7 AR9 OD8.4 | 98.25% | 781/18/0/2 | 1292/1322 | 98.39% | 100.00% |
| 2593923 | 1110 | 302 | 0 | CS4.2 AR9.4 OD8.7 | 97.24% | 1067/37/0/6 | 611/1429 | 97.27% | 100.00% |
| 2768615 | 143 | 60 | 2 | CS9 AR8 OD8 | 45.22% | 55/27/4/57 | 10/215 | 45.82% | 100.00% |
| 3227428 | 98 | 75 | 1 | CS3.5 AR4 OD4 | 91.16% | 87/7/0/4 | 36/199 | 91.53% | 100.00% |
| 3644427 | 1151 | 621 | 2 | CS3 AR9.2 OD8 | 98.73% | 1132/13/0/6 | 774/2256 | 98.77% | 99.94% |
| 37902 | 192 | 52 | 3 | CS3 AR5 OD5 | 97.40% | 186/3/0/3 | 224/413 | 97.64% | 100.00% |
| 39206 | 475 | 121 | 8 | CS4 AR8 OD8 | 95.16% | 447/15/0/13 | 136/660 | 95.12% | 100.00% |
| 42587 | 429 | 147 | 4 | CS5 AR8 OD8 | 96.89% | 412/11/0/6 | 268/600 | 96.99% | 100.00% |
| 50859 | 229 | 78 | 3 | CS4 AR7 OD7 | 96.65% | 219/7/0/3 | 328/386 | 96.90% | 100.00% |
| 75858 | 338 | 130 | 1 | CS4 AR7 OD7 | 99.21% | 335/1/0/2 | 208/498 | 99.19% | 100.00% |
| 801165 | 1612 | 674 | 0 | CS4 AR9.3 OD8 | 93.05% | 1480/58/4/70 | 174/2359 | 92.59% | 100.00% |
| 871815 | 134 | 60 | 1 | CS4 AR8.3 OD6 | 99.25% | 133/0/0/1 | 235/236 | 99.32% | 100.00% |
| bakenohana | 569 | 351 | 0 | CS4 AR9.6 OD9.2 | 77.91% | 426/51/2/90 | 79/1003 | 77.46% | 100.00% |
| renatus_hard | 558 | 396 | 2 | CS4 AR7.5 OD6 | 99.52% | 555/1/0/2 | 451/1134 | 99.56% | 100.00% |
| renatus_insane | 897 | 309 | 2 | CS4 AR9 OD8 | 98.48% | 880/10/0/7 | 426/1328 | 98.46% | 100.00% |
| renatus_normal | 278 | 209 | 2 | CS3 AR4 OD3 | 92.99% | 252/17/5/4 | 147/804 | 93.28% | 100.00% |

judgements = 300/100/50/miss. lazer_acc = accuracy osu!lazer shows for the same replay (it also weights slider ticks).

## Simulator vs osu!lazer

27399/27409 judgements identical on the real-map replays.

## Checkpoints (validation)

| ckpt | acc | miss |
|---|---|---|
| rl_model_500000_steps.zip | 8.73% | 375 |
| rl_model_1000000_steps.zip | 8.00% | 374 |
| rl_model_1500000_steps.zip | 49.46% | 208 |
| rl_model_2000000_steps.zip | 89.34% | 34 |
| rl_model_2500000_steps.zip | 92.99% | 20 |
| rl_model_3000000_steps.zip | 87.64% | 42 |
| rl_model_3500000_steps.zip | 91.49% | 31 |
| rl_model_4000000_steps.zip | 90.50% | 27 |
| rl_model_4500000_steps.zip | 94.89% | 18 |
| rl_model_5000000_steps.zip | 95.02% | 18 |
| rl_model_5500000_steps.zip | 93.95% | 20 |
| rl_model_6000000_steps.zip | 94.55% | 21 |
| rl_model_6500000_steps.zip | 94.49% | 18 |
| rl_model_7000000_steps.zip | 94.49% | 18 |
| rl_model_7500000_steps.zip | 94.55% | 21 |
| rl_model_8000000_steps.zip | 93.89% | 23 |
| rl_model_8500000_steps.zip | 94.52% | 20 |
| rl_model_9000000_steps.zip | 94.79% | 18 |
| rl_model_9500000_steps.zip | 94.65% | 24 |
| rl_model_10000000_steps.zip | 94.98% | 21 |
| rl_model_10500000_steps.zip | 94.72% | 21 |
| rl_model_11000000_steps.zip | 94.62% | 20 |
| rl_model_11500000_steps.zip | 93.93% | 23 |
| rl_model_12000000_steps.zip | 94.22% | 20 |
| rl_model_12500000_steps.zip | 94.66% | 20 |
| rl_model_13000000_steps.zip | 94.83% | 19 |
| rl_model_13500000_steps.zip | 94.53% | 18 |
| rl_model_14000000_steps.zip | 94.33% | 19 |
| rl_model_14500000_steps.zip | 94.56% | 19 |
| rl_model_15000000_steps.zip | 94.36% | 20 |
| rl_model_15500000_steps.zip | 95.13% | 19 |
| rl_model_16000000_steps.zip | 93.85% | 20 |
| rl_model_16500000_steps.zip | 93.69% | 19 |
| rl_model_17000000_steps.zip | 94.79% | 18 |
| rl_model_17500000_steps.zip | 95.02% | 18 |
| rl_model_18000000_steps.zip | 95.02% | 18 |
| rl_model_18500000_steps.zip | 94.52% | 21 |
| rl_model_19000000_steps.zip | 93.83% | 21 |
| rl_model_19500000_steps.zip | 93.79% | 19 |
| rl_model_20000000_steps.zip | 93.92% | 20 |
| model.zip | 93.92% | 20 |
