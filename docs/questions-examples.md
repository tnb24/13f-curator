# Example questions

The challenge organisers supplied these ten questions. They are the first ten items of
the development set, `evals/questions.json`, which holds each one's reference SQL,
answer key, and conventions. `python -m evals.run` asks and grades them.

The last question asks about 2026 Q3, which is not in the dataset. The correct
answer is `null`, with the reason on stderr and no model call.

```
Which manager held the largest Apple position in 2026 Q2?
Which manager added the most Nvidia shares between 2026 Q1 and 2026 Q2?
How many distinct issuers did Renaissance Technologies LLC report in 2026 Q2?
What was the total reported value of Citadel Advisors LLC's holdings in 2026 Q1?
Which managers in the roster filed a 13F-NT instead of a 13F-HR for 2026 Q2?
Did Pershing Square Capital Management report any Microsoft holdings directly in 2026 Q2?
Which manager reported the most call options in 2026 Q2?
What was Third Point LLC's largest position by value in 2026 Q1, and what was it?
Which manager held Tesla in both 2026 Q1 and 2026 Q2, and did the position grow or shrink?
What was the average portfolio value across all managers in 2026 Q3?
```
