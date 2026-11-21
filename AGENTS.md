# Agent instructions

## Scope and source of truth

- Preserve the existing textbook chapter hierarchy. Do not regenerate the repository tree from a stale hard-coded list.
- The first eight Markdown modules under 2_深度网络：现代实践/6_深度前馈网络 are the readable source notes; their paired Python files are executable teaching examples.
- Published pages live in site/articles/<slug>/index.html. When changing a module, update its Markdown, Python example, HTML article, eight SVG assets and site/assets/data/<slug>/evidence.json together.
- The course homepage is site/index.html. Shared CSS and interaction code belong in site/assets/css/site.css and site/assets/js/site.js.

## Scientific correctness

- Read the relevant section of Goodfellow, Bengio & Courville and the cited primary paper before drafting.
- Separate textbook statements, paper results, mathematical derivations, illustrative examples and modern engineering practice.
- Never invent benchmark metrics, tensor shapes, training results, paper claims or browser-test status.
- For loss/output design, state the target variable and probability model. Distinguish logits from probabilities and mutually exclusive classes from multilabel targets.
- For activations, state function and derivative behavior. Do not claim ReLU eliminates all vanishing gradients or SELU self-normalizes without its assumptions.

## Visual and math contract

- Each article must contain at least eight independent SVG figures in site/assets/svg/<slug>/ and reference all of them.
- Each SVG needs xmlns, viewBox, title, desc, unique IDs and readable labels. Do not put MathML inside SVG.
- Use native MathML for display equations. Keep formula containers keyboard-scrollable.
- Figures must explain a computational relationship, not decorate the page. Captions state the claim being illustrated.
- Keep the site offline-first: no CDN scripts, remote fonts or required network assets.

## Verification gates

Before publishing a touched article:

1. Run python scripts/validate_site.py.
2. Run python -m unittest discover -s tests -v.
3. Run python scripts/browser_qa.py with Chromium installed.
4. Inspect screenshots at 1440px, 768px, 390px and 360px; check MathML, figure topology, visible focus and one meaningful interaction.
5. Review the article evidence.json; every nontrivial factual claim needs a source locator and status.

Report checks as passed, failed or not verified. Do not weaken a failing test to make a build green.

## Writing and repository discipline

- Write concise formal Chinese; preserve canonical English technical terms at first use.
- Explain symbols and distinguish assumptions from conclusions.
- Do not copy Christopher Olah's prose, diagrams, code or branding. Borrow only progressive disclosure and consistent visual notation.
- Keep unrelated textbook modules untouched. Do not replace the existing chapter tree with a new curriculum-only tree.
