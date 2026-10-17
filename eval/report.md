# Night Shift — with vs without the shared brain

Same seeds, same models, same prompts (MemDelta). 4 no-brain shifts, 2 brain shifts. Gotchas: sticky gripper (every shift), blue rack moves at shift 3, operator camera nudged at shift 5.

| Metric | No brain | Shared brain |
|---|---|---|
| Shifts | 4 | 2 |
| Success rate | 75% | 50% |
| First successful shift | 1 | 2 |
| False 'done' claims | 25% | 50% |
| Verifier accuracy vs ground truth | 75% | 100% |
| Tool calls / shift | 34.2 | 36.0 |
| Prompt tokens / shift | 115,221 | 133,180 |
|   of which cached | 0 | 0 |
| Completion tokens / shift | 7,043 | 6,172 |
| Re-plans (total) | 0 | 0 |

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
