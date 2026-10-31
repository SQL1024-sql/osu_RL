# 20261031-1522-wide120

training maps: wide. best checkpoint: `rl_model_8000000_steps.zip` (validation acc 96.28%, 18 misses)

## Test set

| set | model_acc | model_fc | model_miss | autoplay_acc | autoplay_fc |
|---|---|---|---|---|---|
| real (26 maps) | 97.20% | 0/26 | 2.9 | 100.00% | 25/26 |
| narrow (50 maps) | 92.85% | 0/50 | 3.2 | 99.96% | 47/50 |
| wide (50 maps) | 92.15% | 0/50 | 3.4 | 99.96% | 47/50 |

acc = osu!stable-style accuracy over objects (300/100/50/miss). autoplay = scripted perfect player, the ceiling.

## Real maps

| map | objects | sliders | spinners | diff | acc | judgements | combo | lazer_acc | autoplay |
|---|---|---|---|---|---|---|---|---|---|
| 100374 | 389 | 149 | 4 | CS4 AR6 OD5 | 98.97% | 385/0/0/4 | 272/634 | 99.04% | 100.00% |
| 103019 | 281 | 129 | 6 | CS4 AR7 OD5 | 97.86% | 275/0/0/6 | 272/601 | 98.11% | 100.00% |
| 104973 | 428 | 186 | 4 | CS4 AR6 OD6 | 98.60% | 421/3/0/4 | 296/725 | 98.71% | 100.00% |
| 1124896 | 1044 | 347 | 2 | CS3.8 AR8.7 OD7.5 | 98.69% | 1025/16/0/3 | 791/1428 | 98.72% | 100.00% |
| 112643 | 541 | 156 | 6 | CS5 AR8 OD8 | 95.04% | 504/30/1/6 | 351/785 | 95.23% | 100.00% |
| 1341554 | 541 | 297 | 0 | CS4.3 AR9.3 OD8 | 97.04% | 518/21/0/2 | 228/877 | 97.03% | 100.00% |
| 1431386 | 370 | 292 | 1 | CS3.3 AR5 OD4 | 99.73% | 369/0/0/1 | 636/781 | 99.76% | 100.00% |
| 1450162 | 260 | 155 | 1 | CS4 AR7.5 OD7 | 99.62% | 259/0/0/1 | 440/441 | 99.65% | 100.00% |
| 1597806 | 115 | 62 | 0 | CS7 AR8 OD8 | 96.23% | 109/5/0/1 | 89/181 | 96.17% | 100.00% |
| 20544 | 98 | 23 | 2 | CS5 AR2 OD2 | 95.92% | 94/0/0/4 | 93/166 | 96.21% | 100.00% |
| 2190499 | 801 | 482 | 2 | CS3.7 AR9 OD8.4 | 96.75% | 763/36/0/2 | 325/1322 | 96.89% | 100.00% |
| 2593923 | 1110 | 302 | 0 | CS4.2 AR9.4 OD8.7 | 94.70% | 1022/87/1/0 | 450/1429 | 94.82% | 100.00% |
| 2768615 | 143 | 60 | 2 | CS9 AR8 OD8 | 89.98% | 124/14/0/5 | 74/215 | 90.04% | 100.00% |
| 3227428 | 98 | 75 | 1 | CS3.5 AR4 OD4 | 98.98% | 97/0/0/1 | 198/199 | 99.10% | 100.00% |
| 3644427 | 1151 | 621 | 2 | CS3 AR9.2 OD8 | 98.00% | 1118/30/0/3 | 1268/2256 | 98.16% | 99.94% |
| 37902 | 192 | 52 | 3 | CS3 AR5 OD5 | 98.44% | 189/0/0/3 | 224/413 | 98.61% | 100.00% |
| 39206 | 475 | 121 | 8 | CS4 AR8 OD8 | 95.79% | 449/18/0/8 | 218/660 | 95.88% | 100.00% |
| 42587 | 429 | 147 | 4 | CS5 AR8 OD8 | 97.94% | 418/6/1/4 | 274/600 | 97.97% | 100.00% |
| 50859 | 229 | 78 | 3 | CS4 AR7 OD7 | 96.36% | 218/8/0/3 | 324/386 | 96.59% | 100.00% |
| 75858 | 338 | 130 | 1 | CS4 AR7 OD7 | 98.52% | 331/6/0/1 | 305/498 | 98.58% | 100.00% |
| 801165 | 1612 | 674 | 0 | CS4 AR9.3 OD8 | 98.33% | 1574/33/0/5 | 540/2359 | 98.31% | 100.00% |
| 871815 | 134 | 60 | 1 | CS4 AR8.3 OD6 | 98.76% | 132/1/0/1 | 163/236 | 98.79% | 100.00% |
| bakenohana | 569 | 351 | 0 | CS4 AR9.6 OD9.2 | 90.01% | 485/80/3/1 | 185/1003 | 90.56% | 100.00% |
| renatus_hard | 558 | 396 | 2 | CS4 AR7.5 OD6 | 99.52% | 555/1/0/2 | 764/1134 | 99.58% | 100.00% |
| renatus_insane | 897 | 309 | 2 | CS4 AR9 OD8 | 98.07% | 872/23/0/2 | 490/1328 | 98.14% | 100.00% |
| renatus_normal | 278 | 209 | 2 | CS3 AR4 OD3 | 99.28% | 276/0/0/2 | 546/804 | 99.41% | 100.00% |

judgements = 300/100/50/miss. lazer_acc = accuracy osu!lazer shows for the same replay (it also weights slider ticks).

## Simulator vs osu!lazer

27407/27409 judgements identical on the real-map replays.

## Checkpoints (validation)

| ckpt | acc | miss |
|---|---|---|
| rl_model_500000_steps.zip | 8.51% | 393 |
| rl_model_1000000_steps.zip | 31.83% | 150 |
| rl_model_1500000_steps.zip | 86.51% | 34 |
| rl_model_2000000_steps.zip | 95.85% | 18 |
| rl_model_2500000_steps.zip | 95.53% | 18 |
| rl_model_3000000_steps.zip | 95.69% | 18 |
| rl_model_3500000_steps.zip | 96.25% | 18 |
| rl_model_4000000_steps.zip | 95.95% | 18 |
| rl_model_4500000_steps.zip | 96.01% | 18 |
| rl_model_5000000_steps.zip | 95.72% | 18 |
| rl_model_5500000_steps.zip | 95.99% | 18 |
| rl_model_6000000_steps.zip | 95.29% | 18 |
| rl_model_6500000_steps.zip | 95.95% | 19 |
| rl_model_7000000_steps.zip | 95.65% | 19 |
| rl_model_7500000_steps.zip | 95.65% | 19 |
| rl_model_8000000_steps.zip | 96.28% | 18 |
| rl_model_8500000_steps.zip | 95.52% | 19 |
| rl_model_9000000_steps.zip | 95.42% | 19 |
| rl_model_9500000_steps.zip | 95.75% | 18 |
| rl_model_10000000_steps.zip | 95.82% | 18 |
| rl_model_10500000_steps.zip | 95.69% | 18 |
| rl_model_11000000_steps.zip | 95.03% | 18 |
| rl_model_11500000_steps.zip | 95.26% | 18 |
| rl_model_12000000_steps.zip | 94.06% | 18 |
| rl_model_12500000_steps.zip | 93.90% | 22 |
| rl_model_13000000_steps.zip | 94.66% | 18 |
| rl_model_13500000_steps.zip | 94.13% | 22 |
| rl_model_14000000_steps.zip | 94.30% | 21 |
| rl_model_14500000_steps.zip | 95.19% | 19 |
| rl_model_15000000_steps.zip | 94.17% | 20 |
| rl_model_15500000_steps.zip | 94.27% | 19 |
| rl_model_16000000_steps.zip | 94.04% | 19 |
| rl_model_16500000_steps.zip | 92.51% | 19 |
| rl_model_17000000_steps.zip | 93.07% | 20 |
| rl_model_17500000_steps.zip | 95.06% | 18 |
| rl_model_18000000_steps.zip | 94.27% | 20 |
| rl_model_18500000_steps.zip | 94.60% | 19 |
| rl_model_19000000_steps.zip | 93.97% | 20 |
| rl_model_19500000_steps.zip | 93.40% | 19 |
| rl_model_20000000_steps.zip | 94.56% | 19 |
| model.zip | 94.56% | 19 |
