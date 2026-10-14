# AI review

This project was built by an AI agent (Claude), coordinated and reviewed by Youssef, with a second Claude session as implementer during the last stretch and independent read-only subagents for verification. This file summarizes what the AI got wrong, how each case was caught, and what changed as a result. The full record, with dates and numbers, is `notes/ai_review_log.md` (14 cases).

Written on 2026-10-14 by the implementing agent at Youssef's request. Status: not yet reviewed line by line by Youssef.

## What went wrong, by kind

**Results stated more strongly than the data allowed** (cases 12 and 13). The most consequential kind. A README draft presented the gap between two policies that behave identically as a one-third reduction in wrong writes, and called noise a gain in task success. The cost table mixed two endpoints and presented an extrapolation as a measurement. Both were caught by review against the raw data: a coordinating session that checked the claim against the Phase 4 analysis, and a read-only subagent that recomputed every README number from the rows and manifests.

**Experimental design confounded by an operational change** (case 10). Switching the API endpoint mid-run, after a weak equivalence check, tied endpoint to task order. Fixed by rerunning every configuration in full on one endpoint, and by recording the endpoint on every result row.

**Labeling bugs** (case 1, and the `action_key` bug in `notes/progress.md`, Phase 2). The canonical comparison against gold actions was wrong twice: once on paired item lists, once because a helper dropped the tool name, so exchange actions could never be labeled correct. The second was caught by the plan's mandatory test (the gold list of a task must label all correct). Every label-dependent number in the project rests on this code.

**Infrastructure that failed silently** (cases 4, 5, 7, 8, 11, 14). A cost wrapper installed after modules had bound the original function, so a paid run reported $0. Rate limits with no retry, then retries that backed off too fast. Detached runs that died with the session, then a relaunch without credentials. A balance error retried for 20 minutes as if it were a rate limit. A dependency installed by hand and never declared, which a clean install exposed.

**Smaller slips** (cases 2, 3, 6, 9). A drafting artifact left in code, a test that read stdin, a near-duplicate 460-trajectory run, an overbroad process kill.

## What caught them

- Tests the plan required (the gold-list test caught a bug that would have invalidated the labels).
- Smoke runs that looked at the output instead of the exit code (the $0 cost, the rate-limit losses).
- Review against raw data by a session or subagent that had not written the claim (the README overclaims, the cost table).
- A from-scratch install in a clean copy (the undeclared dependency).

## What changed

- Every claimed difference is now compared against the difference between two equivalent configurations before it is written up.
- Every result row records its endpoint, and resume is per endpoint.
- One model-call wrapper owns caching, accounting and retries, with a regression test per failure mode.
- Implementation and review run in separate sessions: the one that writes a claim does not approve it.
