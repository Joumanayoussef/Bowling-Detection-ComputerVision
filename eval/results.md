| Video | Setup | True falls | Filter ON: detected / correct / **false** / missed | Filter OFF: detected / correct / **false** / missed | Spatial-filter rejections |
|---|---|---|---|---|---|
| `first_video.mp4` | top-down, RC car, 4 pins | 3 | 4 / 2 / **2** / 1 | 4 / 1 / **3** / 2 | 0 |
| `whatsapp_2026-05-13_2.53.32PM.mp4` | low side view, RC car, 2 pins, hand-held camera drifts | 2 | 2 / 2 / **0** / 0 | 2 / 2 / **0** / 0 | 6 |
| `videobowling.mp4` | side view, reflective floor, thrown ball (no car), 6 pins, 3 rolls | 6 | 5 / 5 / **0** / 1 | 5 / 4 / **1** / 2 | 71 |
| **Total** | | 11 | 11 / 9 / **2** / 2 | 11 / 7 / **4** / 4 | |

Per-event detail (filter ON):

| Video | Pin | Detected (s) | True (s) | Signal | Verdict |
|---|---|---|---|---|---|
| `first_video.mp4` | blue | 1.40 | 3.10 | class_transition | false |
| `first_video.mp4` | green | 1.43 | 2.00 | class_transition | correct |
| `first_video.mp4` | yellow | 1.47 | - | car_contact | false |
| `first_video.mp4` | red | 4.80 | 5.70 | car_contact | correct |
| `whatsapp_2026-05-13_2.53.32PM.mp4` | yellow | 4.26 | 4.55 | car_contact | correct |
| `whatsapp_2026-05-13_2.53.32PM.mp4` | red | 9.97 | 10.15 | proximity_disappearance | correct |
| `videobowling.mp4` | red (front) | 4.43 | 4.60 | class_transition | correct |
| `videobowling.mp4` | right yellow | 12.05 | 12.80 | chain_reaction | correct |
| `videobowling.mp4` | left yellow | 12.35 | 12.70 | chain_reaction | correct |
| `videobowling.mp4` | right blue | 13.58 | 12.60 | class_transition | correct |
| `videobowling.mp4` | left blue | 25.27 | 25.30 | chain_reaction | correct |
