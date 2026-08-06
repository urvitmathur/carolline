# MECH5845M Dissertation — Submission Checklist

## Files

| Item | Path |
|------|------|
| Word report | `dissertation/CAROLLINE_MSc_Report.docx` |
| Chapter sources | `carolline_control/docs/dissertation/ch*.md` |
| Metrics snapshot | `carolline_control/docs/dissertation/report_data.yaml` |
| Figures | `carolline_control/plots/report_figures/` |
| Cover instructions | `carolline_control/docs/dissertation/00_cover_instructions.md` |

## Before submission

- [ ] Replace all `#########` placeholders on cover and declaration pages
- [ ] Paste official cover templates OR merge with module `.docx` templates from Downloads
- [ ] Insert / update **Table of Contents** in Word (References → Table of Contents)
- [ ] Verify page numbers: Roman numerals for front matter, Arabic from Chapter 1
- [ ] Confirm **main body Chapters 1–6 ≤ 35 pages** (references/appendices extra)
- [ ] Font **Arial 11 pt** body, line spacing 1.5; captions/references Arial 10 pt
- [ ] Margins: 25 mm (38 mm left)
- [ ] Every figure/table cited in text with caption
- [ ] Equation variables defined (add MathType if supervisor requires)
- [ ] References complete in Leeds Harvard format
- [ ] Appendix E: insert LinkedIn screenshot
- [ ] Proofread third-person past tense

## Rebuild command

```powershell
.\.venv\Scripts\python.exe carolline_control\docs\dissertation\build_dissertation.py
```

## Estimated body length

Markdown chapters total ~6,500 words ≈ **22–28 pages** at 1.5 spacing with figures — within 35-page limit. Add supervisor-requested depth in Literature Review or Results if under target; trim Monte Carlo discussion if over.

## Examiner quality markers included

- Quantified validation table (Table 5.1)
- Critical discussion of limitations (oracle vs SLAM, Monte Carlo)
- Reproducibility commands in appendices
- Original work separated from paper reproduction
