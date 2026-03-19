# Label Schema

## Taxonomy
The project uses a 5-class email risk taxonomy:

1. `advertisement`
- low-risk commercial spam, bulk promotion, lead generation, traffic diversion, or generic marketing

2. `phishing`
- credential theft, fake verification, account recovery traps, fake login flows, or deceptive security notices

3. `impersonation`
- spoofed identity or role abuse, such as fake executive, finance, HR, supplier, or internal instruction messages

4. `malicious_link_or_attachment`
- delivery of suspicious links, compressed payloads, macro documents, or risky executable-style attachments

5. `black_industry_or_policy_violation`
- gray-market or policy-violating content, including invoice abuse, gambling, illicit services, fraud-oriented traffic, and non-compliant offers

## Boundary Rules
- prioritize the **primary attack intent**, not the loudest keyword
- if the core goal is account takeover, prefer `phishing`
- if the core goal is command abuse through fake identity, prefer `impersonation`
- if the email is mainly a payload delivery vehicle, prefer `malicious_link_or_attachment`
- use `black_industry_or_policy_violation` when the content is clearly non-compliant even if it looks promotional
- use `advertisement` only when the content is promotional but not clearly malicious or policy-violating

## Commonly Confused Pairs
### `phishing` vs `impersonation`
- `phishing`: the main objective is stealing credentials or driving a fake login action
- `impersonation`: the main objective is abusing trust to trigger payment, approval, or instruction execution

### `advertisement` vs `black_industry_or_policy_violation`
- `advertisement`: lower-risk commercial spam
- `black_industry_or_policy_violation`: clearly suspicious, illicit, or policy-breaking content

### `phishing` vs `malicious_link_or_attachment`
- `phishing`: login or verification trap
- `malicious_link_or_attachment`: download or execution path is central
