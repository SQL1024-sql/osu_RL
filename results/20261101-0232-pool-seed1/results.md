# 20261101-0232-pool-seed1

training maps: wide. best checkpoint: `rl_model_10500000_steps.zip` (validation acc 99.01%, 1 misses)

## Test set

| set | model_acc | model_fc | model_miss | autoplay_acc | autoplay_fc |
|---|---|---|---|---|---|
| real (26 maps) | 95.79% | 2/26 | 3.4 | 100.00% | 25/26 |
| narrow (50 maps) | 97.99% | 10/50 | 0.4 | 99.96% | 47/50 |
| wide (50 maps) | 96.17% | 8/50 | 1.2 | 99.96% | 47/50 |

acc = osu!stable-style accuracy over objects (300/100/50/miss). autoplay = scripted perfect player, the ceiling.

## Real maps

| map | objects | sliders | spinners | diff | acc | judgements | combo | lazer_acc | autoplay |
|---|---|---|---|---|---|---|---|---|---|
| 100374 | 389 | 149 | 4 | CS4 AR6 OD5 | 99.49% | 386/3/0/0 | 545/634 | 99.23% | 100.00% |
| 103019 | 281 | 129 | 6 | CS4 AR7 OD5 | 98.04% | 275/1/1/4 | 184/601 | 98.03% | 100.00% |
| 104973 | 428 | 186 | 4 | CS4 AR6 OD6 | 99.14% | 423/4/0/1 | 321/725 | 98.86% | 100.00% |
| 1124896 | 1044 | 347 | 2 | CS3.8 AR8.7 OD7.5 | 99.86% | 1042/1/1/0 | 1303/1428 | 99.76% | 100.00% |
| 112643 | 541 | 156 | 6 | CS5 AR8 OD8 | 99.14% | 535/4/0/2 | 377/785 | 99.08% | 100.00% |
| 1341554 | 541 | 297 | 0 | CS4.3 AR9.3 OD8 | 99.88% | 540/1/0/0 | 877/877 | 99.88% | 100.00% |
| 1431386 | 370 | 292 | 1 | CS3.3 AR5 OD4 | 99.64% | 368/2/0/0 | 461/781 | 99.52% | 100.00% |
| 1450162 | 260 | 155 | 1 | CS4 AR7.5 OD7 | 99.49% | 258/2/0/0 | 298/441 | 99.46% | 100.00% |
| 1597806 | 115 | 62 | 0 | CS7 AR8 OD8 | 85.36% | 91/21/1/2 | 27/181 | 85.23% | 100.00% |
| 20544 | 98 | 23 | 2 | CS5 AR2 OD2 | 98.98% | 97/0/0/1 | 93/166 | 99.05% | 100.00% |
| 2190499 | 801 | 482 | 2 | CS3.7 AR9 OD8.4 | 98.83% | 790/5/0/6 | 473/1322 | 98.90% | 100.00% |
| 2593923 | 1110 | 302 | 0 | CS4.2 AR9.4 OD8.7 | 99.19% | 1098/9/0/3 | 651/1429 | 99.27% | 100.00% |
| 2768615 | 143 | 60 | 2 | CS9 AR8 OD8 | 25.99% | 15/62/9/57 | 8/215 | 26.46% | 100.00% |
| 3227428 | 98 | 75 | 1 | CS3.5 AR4 OD4 | 99.32% | 97/1/0/0 | 141/199 | 98.98% | 100.00% |
| 3644427 | 1151 | 621 | 2 | CS3 AR9.2 OD8 | 99.44% | 1142/7/1/1 | 1169/2256 | 99.47% | 99.94% |
| 37902 | 192 | 52 | 3 | CS3 AR5 OD5 | 98.35% | 188/2/1/1 | 312/413 | 98.27% | 100.00% |
| 39206 | 475 | 121 | 8 | CS4 AR8 OD8 | 99.37% | 471/3/0/1 | 448/660 | 99.35% | 100.00% |
| 42587 | 429 | 147 | 4 | CS5 AR8 OD8 | 99.07% | 424/3/0/2 | 280/600 | 99.25% | 100.00% |
| 50859 | 229 | 78 | 3 | CS4 AR7 OD7 | 99.42% | 227/2/0/0 | 189/386 | 99.38% | 100.00% |
| 75858 | 338 | 130 | 1 | CS4 AR7 OD7 | 98.52% | 331/6/0/1 | 116/498 | 97.90% | 100.00% |
| 801165 | 1612 | 674 | 0 | CS4 AR9.3 OD8 | 99.35% | 1598/9/3/2 | 605/2359 | 99.35% | 100.00% |
| 871815 | 134 | 60 | 1 | CS4 AR8.3 OD6 | 99.00% | 132/2/0/0 | 236/236 | 99.09% | 100.00% |
| bakenohana | 569 | 351 | 0 | CS4 AR9.6 OD9.2 | 97.01% | 544/24/0/1 | 594/1003 | 97.17% | 100.00% |
| renatus_hard | 558 | 396 | 2 | CS4 AR7.5 OD6 | 99.28% | 553/3/0/2 | 368/1134 | 99.35% | 100.00% |
| renatus_insane | 897 | 309 | 2 | CS4 AR9 OD8 | 99.74% | 894/2/0/1 | 1198/1328 | 99.75% | 100.00% |
| renatus_normal | 278 | 209 | 2 | CS3 AR4 OD3 | 99.52% | 276/2/0/0 | 482/804 | 99.55% | 100.00% |

judgements = 300/100/50/miss. lazer_acc = accuracy osu!lazer shows for the same replay (it also weights slider ticks).

## Simulator vs osu!lazer

27357/27409 judgements identical on the real-map replays.

## Checkpoints (validation)

| ckpt | acc | miss |
|---|---|---|
| rl_model_500000_steps.zip | 11.47% | 656 |
| rl_model_1000000_steps.zip | 12.74% | 636 |
| rl_model_1500000_steps.zip | 77.04% | 134 |
| rl_model_2000000_steps.zip | 93.44% | 38 |
| rl_model_2500000_steps.zip | 91.19% | 57 |
| rl_model_3000000_steps.zip | 97.15% | 19 |
| rl_model_3500000_steps.zip | 95.39% | 26 |
| rl_model_4000000_steps.zip | 95.56% | 30 |
| rl_model_4500000_steps.zip | 94.96% | 29 |
| rl_model_5000000_steps.zip | 97.13% | 20 |
| rl_model_5500000_steps.zip | 96.89% | 18 |
| rl_model_6000000_steps.zip | 97.36% | 20 |
| rl_model_6500000_steps.zip | 97.93% | 16 |
| rl_model_7000000_steps.zip | 97.38% | 19 |
| rl_model_7500000_steps.zip | 97.70% | 14 |
| rl_model_8000000_steps.zip | 97.90% | 13 |
| rl_model_8500000_steps.zip | 97.95% | 14 |
| rl_model_9000000_steps.zip | 96.19% | 14 |
| rl_model_9500000_steps.zip | 98.46% | 5 |
| rl_model_10000000_steps.zip | 97.76% | 3 |
| rl_model_10500000_steps.zip | 99.01% | 1 |
| rl_model_11000000_steps.zip | 98.91% | 3 |
| rl_model_11500000_steps.zip | 98.39% | 4 |
| rl_model_12000000_steps.zip | 98.26% | 5 |
| rl_model_12500000_steps.zip | 98.61% | 5 |
| rl_model_13000000_steps.zip | 97.83% | 4 |
| rl_model_13500000_steps.zip | 98.19% | 2 |
| rl_model_14000000_steps.zip | 98.38% | 2 |
| rl_model_14500000_steps.zip | 98.31% | 4 |
| rl_model_15000000_steps.zip | 97.83% | 3 |
| rl_model_15500000_steps.zip | 98.33% | 3 |
| rl_model_16000000_steps.zip | 98.29% | 5 |
| rl_model_16500000_steps.zip | 98.13% | 7 |
| rl_model_17000000_steps.zip | 96.99% | 11 |
| rl_model_17500000_steps.zip | 96.83% | 14 |
| rl_model_18000000_steps.zip | 97.20% | 8 |
| rl_model_18500000_steps.zip | 98.11% | 7 |
| rl_model_19000000_steps.zip | 97.83% | 11 |
| rl_model_19500000_steps.zip | 98.81% | 2 |
| rl_model_20000000_steps.zip | 98.56% | 6 |
| model.zip | 98.49% | 6 |
