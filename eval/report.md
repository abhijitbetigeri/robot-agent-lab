# Night Shift — with vs without the shared brain

Same seeds, same models, same prompts (MemDelta). 4 no-brain shifts, 4 brain shifts. Gotchas: sticky gripper (every shift), blue rack moves at shift 3, operator camera nudged at shift 5.

| Metric | No brain | Shared brain |
|---|---|---|
| Shifts | 4 | 4 |
| Success rate | 75% | 75% |
| First successful shift | 1 | 2 |
| False 'done' claims | 25% | 25% |
| Verifier accuracy vs ground truth | 75% | 75% |
| Tool calls / shift | 34.2 | 37.8 |
| Prompt tokens / shift | 115,221 | 157,796 |
|   of which cached | 0 | 0 |
| Completion tokens / shift | 7,043 | 8,726 |
| Re-plans (total) | 0 | 2 |

## No brain, shift by shift

| Shift | Success | Claimed done | Verifier | False done | Tool calls | Prompt tok | Decision |
|---|---|---|---|---|---|---|---|
| 1 | ✅ | yes | pass | no | 39 | 138,966 | continue |
| 2 | ❌ | yes | fail | **yes** | 34 | 112,784 | continue |
| 3 | ✅ | yes | fail | no | 34 | 113,979 | continue |
| 4 | ✅ | yes | pass | no | 30 | 95,154 | continue |

## Shared brain, shift by shift

| Shift | Success | Claimed done | Verifier | False done | Tool calls | Prompt tok | Decision |
|---|---|---|---|---|---|---|---|
| 1 | ❌ | yes | fail | **yes** | 37 | 132,093 | continue |
| 2 | ✅ | yes | pass | no | 35 | 134,267 | continue |
| 3 | ✅ | no | pass | no | 34 | 142,020 | stop |
| 4 | ✅ | no | fail | no | 45 | 222,805 | stop |

Stale `bin_blue` positions served after the rack moved: **0**

## What the brain held at the end

| id | tier | subject | relation | object | from shift |
|---|---|---|---|---|---|
| 4 | insight | operator | overclaim | shift 1: claimed done but verifier failed ['C1']; verify holding=true before placing and p | 1 |
| 5 | insight | verifier | criteria_for_red_patch | tube must have x between 0.215 and 0.225 AND y between 0.165 and 0.175 | 1 |
| 6 | insight | gripper | hold_validation | require holding=true before lift or move_above to avoid empty gripper actions | 1 |
| 9 | procedure | gripper | close_behaviour | first close often reports holding=null and does NOT grip; close a second time and check ho | 2 |
| 14 | fact | bin_red | position_xy | [0.219, 0.171] | 4 |
| 15 | fact | bin_blue | position_xy | [0.221, -0.090] | 4 |
| 16 | procedure | park | look_after_park | look for bin states immediately after park and before next look | 4 |