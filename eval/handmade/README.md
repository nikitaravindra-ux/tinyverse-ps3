# Hand-crafted evaluation documents

Ten short documents with gold labels, used by `eval/run_eval.py`:

- `people`: which mentions belong to the same person (name-variant linking). Every exact
  occurrence of a mention string in the text is a gold mention of that person. Longer mentions take precedence.
- `must_not_leak`: strings that must not appear in the final, audited summary (case-insensitive).
  Digit strings are also checked inside longer digit runs.
- `qi_traps`: phrases that must be flagged as medium or high quasi-identifier risk in the redacted document.

Focus areas: linking traps (01–05: shared surnames, father/son, honorific gender conflicts, all-caps
headers), summary leak traps (06–07: IDs, partial digits, obfuscated emails, UPI handles), and
quasi-identifier traps (08–10: "the only woman director of a Pune firm", age + CFO + company size,
principal + only school + health detail).
