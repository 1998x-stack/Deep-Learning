# Site architecture

## Information architecture

The textbook-oriented source tree remains intact. The publication layer is additive:

~~~text
Deep-Learning/
├── 1_应用数学与机器学习基础/
├── 2_深度网络：现代实践/
│   └── 6_深度前馈网络/       # Markdown + Python source modules
├── 3_深度学习研究/
├── docs/
├── scripts/
├── tests/
└── site/
    ├── index.html
    ├── articles/<slug>/index.html
    └── assets/
        ├── css/site.css
        ├── js/site.js
        ├── svg/<slug>/fig-01.svg ... fig-08.svg
        └── data/<slug>/evidence.json
~~~

## URL mapping

- Home: /
- Article: /articles/<slug>/
- SVG: /assets/svg/<slug>/fig-XX.svg
- Evidence: /assets/data/<slug>/evidence.json

All links are relative so the same output works under the GitHub Pages project path.

## Rendering

- Semantic HTML5 and native MathML keep formulas offline-first.
- SVG is standalone XML with title, description and stable viewBox.
- Shared CSS provides a narrow reading column, desktop contents navigation, responsive mobile navigation, visible focus, reduced-motion and print rules.
- Shared JavaScript only enhances static demos; each page remains readable without interaction.

## Release

The Pages workflow validates HTML references, SVG XML, evidence records, Python examples and Playwright browser behavior. It uploads only site/. The repository Pages source must be set to GitHub Actions once in Settings.
