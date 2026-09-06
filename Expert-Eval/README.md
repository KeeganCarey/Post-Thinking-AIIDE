# Expert-Eval

Blind expert ratings of 10 characters × 3 conditions (no / pre / post). **n = 32** (9 volunteer + 23 Prolific).

- The characters and conversations used in the google form (except with traces attached): `Eval-Source.md`
- A/B/C → condition key: `eval_form_key.json`
- Responses: `FormResponse-Volunteer.csv`, `FormResponse-Prolific.csv` (we did not push them for privacy reasons)
- Analysis: `analysis.py` (unblinds, Wilcoxon+Holm, Friedman, ranks, BEST/WORST, Krippendorff)

```bash
uv run python Expert-Eval/analysis.py
```

See the **Blind Expert Survey** section of the paper for the study design.
