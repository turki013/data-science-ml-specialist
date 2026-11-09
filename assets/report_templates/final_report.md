# Final Report: {{PROJECT}}
Date: {{DATE}}   Code version: TODO(fill)

## 1. Summary
TODO(fill): what was asked, what was measured, what the evidence supports and what it does not.

## 2. Limitations and open problems (read this first)
1. TODO(fill)

## 3. Worst-performing slices
TODO(fill): table with n and an interval per slice.

## 4. Data and setup
- Problem and success criteria: see reports/project_brief.md
- Row accounting, leakage checks and their outcomes: TODO(fill)
- Split leakage gap (same model, same rows, random vs grouped split): TODO(fill)

## 5. Experiments (all of them, including failed and inconclusive)
Render `reports/experiments.jsonl` with `dstk.explog.render_markdown`. TODO(fill)

## 6. Results on the held-out test set
TODO(fill): baseline next to the model, intervals, n, paired-test verdict ("no clear difference" when that is the finding).

## 7. Model selection
TODO(fill): the pre-declared rule, the rule's choice, the final choice, deviation reason if any.

## 8. Error analysis
TODO(fill): what the worst predictions share; what is tested and what is hypothesis.

## 9. Risks and appropriate use
TODO(fill)

## 10. Reproduction
TODO(fill): commands, versions, seeds, data hash.

## 11. Claims audit
- [ ] Every number comes from reports/results.json or reports/params.json (python scripts/check_deliverables.py passes).
- [ ] results.json was written by the pipeline (integrity stamp valid), never edited by hand.
- [ ] Every comparison states n and an interval or test.
- [ ] Earlier claims were updated after each pipeline fix.
