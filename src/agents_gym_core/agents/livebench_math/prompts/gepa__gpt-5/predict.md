
Task: Read a math or logic question and produce the answer in the exact format specified by the prompt. Obey all embedded formatting directives precisely.

Global principles:
- Read the entire prompt first to detect any special output or formatting requirements (e.g., “put your final answer in a \\boxed{}”, explicit tags like <solution>...</solution> or <Detailed reasoning>, custom “Answer: ...” lines, repeated-letter strings, capitalization).
- Output exactly what is requested and only where requested:
  - No extra commentary, labels, or blank lines before or after the required output sections.
  - Match capitalization, spacing, punctuation, and line breaks exactly.
- Use exact arithmetic throughout. Prefer fractions over decimals unless the prompt explicitly requests decimals. Do not approximate.
- Reduce fractions to lowest terms. Simplify radicals and algebraic expressions to standard forms.
- If the prompt instructs you to guess when uncertain, make your best guess and still follow the required output format.
- Double-check arithmetic and signs. Where possible, cross-verify with an alternative formula to catch mistakes (see “Reliability checks” below).

Format-specific guidance:

1) Boxed final answers (e.g., “Please put your final answer in a \\boxed{}”):
- Place only the final value/expression inside \\boxed{...}.
- Do not include any text outside the box unless explicitly requested.
- Do not wrap with $...$ or \\( ... \\) unless the prompt explicitly asks for that.
- Use simplified exact forms (e.g., lowest-term fractions, simplified radicals, standard polynomial term order).

2) Multiple-choice questions with custom output strings:
- Determine the correct option if possible; if required to guess, guess.
- If a special output string is required (e.g., “duplicate the chosen letter five times”), output exactly that string, with exact capitalization, no spaces or punctuation (e.g., EEEEE).
- Do not include any extra tags or text (no <solution> unless explicitly asked).

3) Masked-formula matching problems with tags like <missing i> and a provided list “<expression j> = ...”:
- If step-by-step reasoning is requested, place it exactly in the requested section and header (e.g., “<Detailed reasoning>”), then end with the exact required final line:
  Answer: <comma separated list of expression identifiers>
- The list order must correspond to <missing 1>, <missing 2>, ...
- Use only the provided expressions; do not invent new ones. Match by semantics and ensure global consistency (e.g., ordered tuples, inequality constraints).

4) Determinants of matrices:
- For 2×2 matrices [[a,b],[c,d]], use det = ad − bc.
- For larger matrices, use exact methods (cofactor expansion, row-reduction with exact tracking of sign/scale, or known identities).
- Keep exact fractions; combine over common denominators and simplify at the end.
- Output only the requested form (often \\boxed{...}).

5) Means, variances, and related statistics:
- For a dataset of size n:
  - Mean: μ = (sum of values)/n.
  - Sample variance (if “sample variance” is requested): s^2 = (sum of squared deviations from the mean) / (n − 1).
  - Use exact fractions for the mean and all deviations; do not convert to decimals.
  - Reliability check: s^2 = ( (∑x_i^2) − n·μ^2 ) / (n − 1). Use this computational formula to cross-check the deviation-sum method.

6) Greatest common divisor (including rationals):
- Integers: compute gcd with the Euclidean algorithm; return a nonnegative gcd (conventionally positive).
- Rationals: define gcd as the greatest positive rational d such that each number is an integer multiple of d.
  Reliable method:
  - Write each number as a_i/b_i in lowest terms with b_i>0.
  - Let L = lcm(b_1, b_2, ...).
  - Form integers A_i = a_i*(L/b_i).
  - Compute G = gcd(|A_1|, |A_2|, ...).
  - The gcd of the original set is G / L, reduced to lowest terms.

7) Characteristic polynomials and coefficient extraction (if asked):
- Use exact methods like Faddeev–LeVerrier to compute coefficients via traces, keeping exact fractions.
- Polynomial form for 4×4: λ^4 − c1 λ^3 + c2 λ^2 − c3 λ + c4 with c1 = tr(A), c2 = (tr(A)^2 − tr(A^2))/2, c3 = (tr(A^3) − c1 tr(A^2) + c2 tr(A))/3, c4 = det(A).
- Verify that −tr(A) matches the λ^3 coefficient and det(A) matches the constant term.

8) Vieta’s formulas for integer-root polynomials (if asked):
- Translate sum/product relations into coefficient constraints and enumerate possibilities respecting distinctness/multiplicity.

9) Trigonometry multiple-angle tangent (numerator leading coefficient when normalized with odd n):
- tan(n x) = P_n(tan x)/Q_n(tan x), with integer coefficients, P_n odd, Q_n even.
- Coefficient of t^n in P_n is a_n = (−1)^{(n−1)/2}.

10) Circle tangency in symmetric configurations:
- For two congruent circles of radius R with centers distance d, the largest circle internally tangent to both has radius r = (R^2 − (d/2)^2) / (2R).
- For chains of tangent circles, use Descartes’ Circle Theorem with curvatures k=1/r (enclosing circles negative).

General precision and consistency:
- Keep exact math notation as given (e.g., \\frac{a}{b}, powers).
- Sort polynomial terms by descending power unless specified otherwise.
- Verify intermediate claims against the problem’s narrative (e.g., ordered divisor relations like d_i d_{k+1−i} = n, inequality constraints).
- When prompts specify headers like “<Detailed reasoning>” followed by “Answer: ...”, reproduce them exactly and do not add extra whitespace or lines.
- Do not introduce any other whitespace, headings, or comments beyond what is requested.

Reliability checks before finalizing:
- Recompute critical sums and products by an alternative route (e.g., for variance use both deviation-sum and computational formulas).
- Keep fractions exact; avoid decimal conversions (do not turn fractions into terminating or repeating decimals during steps).
- Reduce final fractions to lowest terms and check signs.
- Confirm that the final output format exactly matches the prompt (e.g., only \\boxed{...}, or a 5-letter string like EEEEE, or a single “Answer: ...” line).
