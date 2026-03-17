# Label Schema

## Taxonomy
The project uses a 5-class email risk taxonomy:

1. `advertisement`
- Low-risk commercial spam, mass marketing, lead generation, promotion, or traffic diversion.

2. `phishing`
- Credential theft, fake login, verification, account warning, or deceptive click-through messages.

3. `impersonation`
- Identity spoofing or role abuse, such as fake executive, HR, finance, supplier, or internal instruction messages.

4. `malicious_link_or_attachment`
- Emails centered on suspicious links or risky attachments, including macro documents, archives, or executable-style payloads.

5. `black_industry_or_policy_violation`
- Gray/black-market content, illegal services, invoice abuse, gambling, fraud-oriented traffic, or policy-violating business promotion.

## Boundary Rules
- Prioritize the **main attack intent**, not a single keyword.
- If both identity spoofing and credential theft appear, prefer `phishing` when the core objective is account takeover.
- If the email mainly pushes a risky file or download rather than a fake login page, prefer `malicious_link_or_attachment`.
- Use `black_industry_or_policy_violation` when the content is clearly policy-violating or gray/black-market oriented, even if it looks like generic marketing.
- Use `advertisement` only when the content is promotional but not clearly malicious or policy-violating.

## Commonly Confused Pairs
- `phishing` vs `impersonation`
  - `phishing`: steal credentials or drive fake login actions.
  - `impersonation`: abuse a trusted identity to trigger payment, approval, or instruction execution.

- `advertisement` vs `black_industry_or_policy_violation`
  - `advertisement`: commercial spam with lower direct harm.
  - `black_industry_or_policy_violation`: clearly suspicious, non-compliant, or illegal offer.

- `phishing` vs `malicious_link_or_attachment`
  - `phishing`: login or verification trap.
  - `malicious_link_or_attachment`: payload delivery or risky file execution.

## Annotation Notes
- Read `subject + content + doccontent` first.
- Then inspect `from`, `fromname`, `url`, `attach`, `htmltag`, `ip`, and `rcpt`.
- Keep borderline cases in notes during annotation or error analysis.
