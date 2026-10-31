# 20261101-0130-seed1

training maps: wide. best checkpoint: `rl_model_6500000_steps.zip` (validation acc 99.37%, 0 misses)

## Test set

| set | model_acc | model_fc | model_miss | autoplay_acc | autoplay_fc |
|---|---|---|---|---|---|
| real (26 maps) | 94.77% | 2/26 | 1.8 | 100.00% | 25/26 |
| narrow (50 maps) | 97.67% | 13/50 | 0.3 | 99.96% | 47/50 |
| wide (50 maps) | 96.57% | 4/50 | 0.7 | 99.96% | 47/50 |

acc = osu!stable-style accuracy over objects (300/100/50/miss). autoplay = scripted perfect player, the ceiling.

## Real maps

| map | objects | sliders | spinners | diff | acc | judgements | combo | lazer_acc | autoplay |
|---|---|---|---|---|---|---|---|---|---|
| 100374 | 389 | 149 | 4 | CS4 AR6 OD5 | 99.83% | 388/1/0/0 | 514/634 | 99.82% | 100.00% |
| 103019 | 281 | 129 | 6 | CS4 AR7 OD5 | 94.78% | 260/17/4/0 | 117/601 | 94.83% | 100.00% |
| 104973 | 428 | 186 | 4 | CS4 AR6 OD6 | 93.89% | 390/35/1/2 | 89/725 | 93.59% | 100.00% |
| 1124896 | 1044 | 347 | 2 | CS3.8 AR8.7 OD7.5 | 99.36% | 1034/10/0/0 | 1428/1428 | 99.36% | 100.00% |
| 112643 | 541 | 156 | 6 | CS5 AR8 OD8 | 98.64% | 531/7/2/1 | 160/785 | 98.84% | 100.00% |
| 1341554 | 541 | 297 | 0 | CS4.3 AR9.3 OD8 | 93.90% | 493/43/4/1 | 351/877 | 94.04% | 100.00% |
| 1431386 | 370 | 292 | 1 | CS3.3 AR5 OD4 | 94.05% | 337/33/0/0 | 206/781 | 94.19% | 100.00% |
| 1450162 | 260 | 155 | 1 | CS4 AR7.5 OD7 | 95.64% | 243/17/0/0 | 403/441 | 95.78% | 100.00% |
| 1597806 | 115 | 62 | 0 | CS7 AR8 OD8 | 96.67% | 110/3/1/1 | 152/181 | 96.68% | 100.00% |
| 20544 | 98 | 23 | 2 | CS5 AR2 OD2 | 89.12% | 86/4/0/8 | 49/166 | 89.52% | 100.00% |
| 2190499 | 801 | 482 | 2 | CS3.7 AR9 OD8.4 | 96.00% | 753/48/0/0 | 1322/1322 | 96.21% | 100.00% |
| 2593923 | 1110 | 302 | 0 | CS4.2 AR9.4 OD8.7 | 98.66% | 1088/21/1/0 | 1382/1429 | 98.67% | 100.00% |
| 2768615 | 143 | 60 | 2 | CS9 AR8 OD8 | 67.48% | 87/28/1/27 | 16/215 | 68.55% | 100.00% |
| 3227428 | 98 | 75 | 1 | CS3.5 AR4 OD4 | 89.80% | 83/15/0/0 | 39/199 | 89.66% | 100.00% |
| 3644427 | 1151 | 621 | 2 | CS3 AR9.2 OD8 | 92.18% | 1024/95/32/0 | 291/2256 | 92.30% | 99.94% |
| 37902 | 192 | 52 | 3 | CS3 AR5 OD5 | 94.79% | 177/15/0/0 | 159/413 | 94.21% | 100.00% |
| 39206 | 475 | 121 | 8 | CS4 AR8 OD8 | 97.19% | 456/15/4/0 | 162/660 | 96.87% | 100.00% |
| 42587 | 429 | 147 | 4 | CS5 AR8 OD8 | 99.07% | 423/6/0/0 | 195/600 | 99.00% | 100.00% |
| 50859 | 229 | 78 | 3 | CS4 AR7 OD7 | 98.11% | 223/5/0/1 | 141/386 | 98.15% | 100.00% |
| 75858 | 338 | 130 | 1 | CS4 AR7 OD7 | 98.42% | 331/5/0/2 | 251/498 | 98.41% | 100.00% |
| 801165 | 1612 | 674 | 0 | CS4 AR9.3 OD8 | 95.99% | 1516/92/4/0 | 777/2359 | 96.00% | 100.00% |
| 871815 | 134 | 60 | 1 | CS4 AR8.3 OD6 | 98.76% | 132/1/0/1 | 163/236 | 98.79% | 100.00% |
| bakenohana | 569 | 351 | 0 | CS4 AR9.6 OD9.2 | 94.64% | 526/36/3/4 | 547/1003 | 94.95% | 100.00% |
| renatus_hard | 558 | 396 | 2 | CS4 AR7.5 OD6 | 91.79% | 490/65/3/0 | 291/1134 | 92.15% | 100.00% |
| renatus_insane | 897 | 309 | 2 | CS4 AR9 OD8 | 98.33% | 875/20/2/0 | 485/1328 | 98.27% | 100.00% |
| renatus_normal | 278 | 209 | 2 | CS3 AR4 OD3 | 96.82% | 265/12/1/0 | 193/804 | 96.90% | 100.00% |

judgements = 300/100/50/miss. lazer_acc = accuracy osu!lazer shows for the same replay (it also weights slider ticks).

## Simulator vs osu!lazer

27395/27409 judgements identical on the real-map replays.

## Checkpoints (validation)

| ckpt | acc | miss |
|---|---|---|
| rl_model_500000_steps.zip | 14.00% | 325 |
| rl_model_1000000_steps.zip | 47.02% | 109 |
| rl_model_1500000_steps.zip | 79.88% | 22 |
| rl_model_2000000_steps.zip | 88.34% | 22 |
| rl_model_2500000_steps.zip | 91.12% | 22 |
| rl_model_3000000_steps.zip | 94.26% | 18 |
| rl_model_3500000_steps.zip | 95.72% | 9 |
| rl_model_4000000_steps.zip | 96.28% | 9 |
| rl_model_4500000_steps.zip | 98.57% | 0 |
| rl_model_5000000_steps.zip | 98.64% | 1 |
| rl_model_5500000_steps.zip | 98.97% | 1 |
| rl_model_6000000_steps.zip | 98.14% | 1 |
| rl_model_6500000_steps.zip | 99.37% | 0 |
| rl_model_7000000_steps.zip | 98.95% | 3 |
| rl_model_7500000_steps.zip | 99.30% | 0 |
| rl_model_8000000_steps.zip | 98.14% | 2 |
| rl_model_8500000_steps.zip | 95.99% | 9 |
| rl_model_9000000_steps.zip | 96.99% | 5 |
| rl_model_9500000_steps.zip | 96.53% | 7 |
| rl_model_10000000_steps.zip | 97.48% | 5 |
| rl_model_10500000_steps.zip | 97.51% | 4 |
| rl_model_11000000_steps.zip | 98.44% | 2 |
| rl_model_11500000_steps.zip | 99.01% | 0 |
| rl_model_12000000_steps.zip | 99.24% | 1 |
| rl_model_12500000_steps.zip | 98.97% | 3 |
| rl_model_13000000_steps.zip | 98.75% | 2 |
| rl_model_13500000_steps.zip | 99.04% | 2 |
| rl_model_14000000_steps.zip | 97.99% | 3 |
| rl_model_14500000_steps.zip | 94.43% | 4 |
| rl_model_15000000_steps.zip | 95.92% | 4 |
| rl_model_15500000_steps.zip | 94.78% | 9 |
| rl_model_16000000_steps.zip | 92.50% | 16 |
| rl_model_16500000_steps.zip | 93.89% | 8 |
| rl_model_17000000_steps.zip | 96.01% | 2 |
| rl_model_17500000_steps.zip | 98.24% | 3 |
| rl_model_18000000_steps.zip | 97.61% | 3 |
| rl_model_18500000_steps.zip | 97.31% | 3 |
| rl_model_19000000_steps.zip | 97.22% | 4 |
| rl_model_19500000_steps.zip | 96.85% | 6 |
| rl_model_20000000_steps.zip | 96.02% | 8 |
| model.zip | 96.02% | 8 |
