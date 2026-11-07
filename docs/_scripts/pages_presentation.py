"""Repair generated Farama theme markup for this project's GitHub Pages site."""

import re
from html import escape
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

REPOSITORY = "env-ViZDoom-turbo"
BASE_URL = f"https://tsilva.github.io/{REPOSITORY}/"
DESCRIPTION = "Learn to build Doom reinforcement learning environments with ViZDoom Turbo, including Gymnasium setup, scenario configuration, and Python and C++ API guides."
MENU_PATH = "_static/versioning/versioning_menu.html"


def polish(path, source):
    """Apply the same presentation fixes to fresh and already published HTML."""
    if path == "_static/pygments.css":
        return re.sub(r"#4e9a06\b", "#376e04", source, flags=re.IGNORECASE)
    if path == MENU_PATH:
        if 'aria-controls="docs-versions"' in source:
            return source
        source = source.replace(
            '<div class="versions_menu__header">',
            '<button class="versions_menu__header" type="button" aria-expanded="false" aria-controls="docs-versions">',
        ).replace(
            '<!-- Current Version -->\n    </div>',
            '<!-- Current Version -->\n    </button>',
        ).replace(
            '<div class="versions_menu__content">',
            '<div class="versions_menu__content" id="docs-versions">',
        ).replace(
            'id="versions_menu-current-version"></span>',
            'id="versions_menu-current-version">Documentation versions</span>',
        ).replace(
            '.versions_menu__header {\n            display: flex;',
            '.versions_menu__header {\n            width: 100%;\n            border: 0;\n            color: inherit;\n            font: inherit;\n            text-align: left;\n            display: flex;',
        ).replace(
            '.versions_menu__header:hover {',
            '.versions_menu__header:hover, .versions_menu__header:focus-visible {',
        ).replace(
            '.versions_menu__content {\n            width: 200px;',
            '.versions_menu__content {\n            visibility: hidden;\n            width: 200px;',
        ).replace(
            '.versions_menu.active .versions_menu__content {',
            '.versions_menu.active .versions_menu__content {\n            visibility: visible;',
            1,
        )
        return (
            source.replace('const basePath = "";', 'const basePath = versioningConfig.basePath;')
            .replace('replace(/^v/i, "")', r'replace(/^.*?v(?=\d)/i, "").replace(/\.post(?=\d)/i, ".")')
            .replace('document.location.href.split(document.location.origin)[1]', 'document.location.pathname')
            .replace(
                '\n                    currentVersionStr = currentVersion || latestVersion;',
                '\n                    let currentVersionStr = currentVersion || latestVersion;',
            )
            .replace(
                'menuElem.classList.add("active");\n            }',
                'menuElem.classList.add("active");\n            }\n'
                '            headerElem.setAttribute("aria-expanded", String(menuElem.classList.contains("active")));',
            )
            .replace(
                'menuElem.classList.remove("active");\n        });',
                'menuElem.classList.remove("active");\n'
                '            headerElem.setAttribute("aria-expanded", "false");\n        });',
            )
            .replace('.innerHTML = currentVersionStr;', '.textContent = currentVersionStr;')
        )

    source = source.replace(
        '<p class="caption" role="heading">',
        '<p class="caption" role="heading" aria-level="2">',
    )
    source = re.sub(
        r'(<img class="farama-black-logo-invert" src="[^"]+")(?=\s*/?>)',
        r'\1 alt=""',
        source,
    )
    source = source.replace('alt="Gymnasium Logo"', 'alt="ViZDoom"')
    if path == "index.html":
        description = escape(DESCRIPTION, quote=True)
        for name in ('name="description"', 'property="og:description"'):
            source = re.sub(
                rf'(<meta {name} content=")[^"]*(")',
                rf"\g<1>{description}\2",
                source,
            )
    canonical = BASE_URL + (path.removesuffix("index.html") if PurePosixPath(path).name == "index.html" else path)
    for attribute in ('rel="canonical" href', 'property="og:url" content'):
        source = re.sub(rf'({attribute}=")[^"]*(")', rf"\g<1>{canonical}\2", source)
    source = re.sub(
        r'(property="og:image" content=")https?://[^"]*/_static/',
        rf"\g<1>{BASE_URL}_static/",
        source,
    )

    source = re.sub(r'(pygments\.css\?v=[0-9a-f]+)(?=")', r'\1-contrast1', source)
    base_path = urlsplit(BASE_URL).path.rstrip("/")
    source = re.sub(
        r"githubUser: '[^']*',\s*githubRepo: '[^']*',(?:\s*basePath: '[^']*',)?",
        f"githubUser: 'tsilva',\n        githubRepo: '{REPOSITORY}',\n        basePath: '{base_path}',",
        source,
    )
    return source.replace(
        "fetch('/main/_static/versioning/versioning_menu.html')",
        f"fetch('{base_path}/{MENU_PATH}')",
    )


def finish_html(app, exception):
    if exception is not None or app.builder.format != "html":
        return
    root = Path(app.outdir)
    for path in (*root.rglob("*.html"), root / "_static/pygments.css"):
        name = path.relative_to(root).as_posix()
        if not path.is_file() or (name.startswith("_static/") and name not in (MENU_PATH, "_static/pygments.css")):
            continue
        source = path.read_text(encoding="utf-8")
        rendered = polish(name, source)
        if rendered != source:
            path.write_text(rendered, encoding="utf-8")
