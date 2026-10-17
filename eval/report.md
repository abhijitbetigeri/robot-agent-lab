# Robot Agent Lab — with vs without the shared brain

Same seeds, same models, same prompts (MemDelta). 4 no-brain shifts, 4 brain shifts. Gotchas: sticky gripper (every shift), blue rack moves at shift 3, operator camera nudged at shift 5.

| Metric | No brain | Shared brain |
|---|---|---|
| Shifts | 4 | 4 |
| Success rate | 25% | 25% |
| First successful shift | 2 | 1 |
| False 'done' claims | 25% | 50% |
| Verifier accuracy vs ground truth | 100% | 100% |
| Tool calls / shift | 34.2 | 38.8 |
| Prompt tokens / shift | 121,407 | 157,404 |
|   of which cached | 0 | 0 |
| Completion tokens / shift | 7,992 | 9,250 |
| Re-plans (total) | 4 | 2 |

## No brain, shift by shift

| Shift | Success | Claimed done | Verifier | False done | Tool calls | Prompt tok | Decision |
|---|---|---|---|---|---|---|---|
| 1 | ❌ | yes | fail | **yes** | 30 | 101,994 | continue |
| 2 | ✅ | yes | pass | no | 36 | 137,972 | continue |
| 3 | ❌ | no | fail | no | 26 | 73,098 | stop |
| 4 | ❌ | no | fail | no | 45 | 172,564 | stop |

## Shared brain, shift by shift

| Shift | Success | Claimed done | Verifier | False done | Tool calls | Prompt tok | Decision |
|---|---|---|---|---|---|---|---|
| 1 | ✅ | yes | pass | no | 30 | 107,377 | continue |
| 2 | ❌ | yes | fail | **yes** | 33 | 137,119 | continue |
| 3 | ❌ | no | fail | no | 51 | 182,765 | replan |
| 4 | ❌ | yes | fail | **yes** | 41 | 202,353 | continue |

Stale `bin_blue` positions served after the rack moved: **0**

## What the brain held at the end

| id | tier | subject | relation | object | from shift |
|---|---|---|---|---|---|
| 4 | fact | procedure_known | blue_tube_absorption | do_not_attempt_move_when_absorption=false | 1 |
| 5 | procedure | move_above | check_z_error_before_lowering | if_z_err_m>0.0003_retry | 1 |
| 6 | insight | verifier | requires_actionable_evidence | recording_used_tube_positions_is_insufficient_during_verification | 1 |
| 12 | procedure | operator | must_verify_z_error | retry_if_z_err_m_gt_0.0003 | 2 |
| 15 | procedure | gripper | close_behaviour | first close often reports holding=null and does NOT grip; close a second time and check ho | 3 |
| 16 | procedure | lower | abort_if_touching_is_false_due_to_report_progress_inaccuracy | retry_if_z_err_m_gt_0.0003_when_aiming_for_touch | 3 |
| 17 | fact | bin_red | position_xy | [0.222, 0.169] | 4 |
| 18 | fact | bin_blue | position_xy | [0.223, -0.088] | 4 |
| 19 | insight | operator | overclaim | shift 4: claimed done but verifier failed ['C2']; verify holding=true before placing and p | 4 |
| 20 | insight | operator | verification | must verify all claimed placements with verifier's specific checks; do not assume holding= | 4 |
| 21 | insight | operator | claim_process | require verifier evidence for C1-C3 before claiming done; do not park and claim without fi | 4 |
| 22 | procedure | lift | z_error_threshold | retry lowering if z_err_m > 0.0003 before confirming touch; current protocol missed retrie | 4 |