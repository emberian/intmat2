#!/usr/bin/env python3
"""Stage site/ (curated, tracked) and web/ (viewer) into _site/ for GitHub Pages.

site/collection.json is the ordered list [{"name", "title", "caption"}]; each
site/collection/<name>/ holds photograph.(webp|png), recipe.toml, optics.toml,
manifest.json and optionally volume/ (Intmat2.viewer_volume) and film.mp4.
"""
import html
import json
import pathlib
import shutil

REPO = pathlib.Path(__file__).resolve().parent.parent
SITE, WEB, OUT = REPO / 'site', REPO / 'web', REPO / '_site'
GITHUB = 'https://github.com/emberian/intmat2/blob/dev/'


def trajectory(vol):
    try:
        frames = json.loads((vol / 'index.json').read_text())['frames']
    except (OSError, ValueError, KeyError):
        return ''
    pick = [frames[0], frames[len(frames) // 2], frames[-1]]
    s = ' → '.join(f"{f['summary']['mean_s']:.3f}" for f in pick)
    lb = ' → '.join(f"{f['summary']['light_bottom']:.3f}" for f in pick)
    steps = ' / '.join(str(f['step']) for f in pick)
    return f'steps {steps}: mean s {s}; light at floor {lb}'


def card(item):
    name = item['name']
    d = SITE / 'collection' / name
    photo = next((p.name for p in (d / 'photograph.webp', d / 'photograph.png') if p.exists()), None)
    links = []
    if (d / 'volume' / 'index.json').exists():
        links.append(f'<a href="viewer/viewer.html?v=../collection/{name}/volume&amp;name={html.escape(name)}">volume</a>')
    if (d / 'film.mp4').exists():
        links.append(f'<a href="collection/{name}/film.mp4">film</a>')
    for f in ('recipe.toml', 'optics.toml', 'manifest.json'):
        if (d / f).exists():
            links.append(f'<a href="collection/{name}/{f}">{f}</a>')
    img = f'<a href="collection/{name}/{photo}"><img src="collection/{name}/{photo}" alt="{html.escape(item.get("title", name))}" loading="lazy"></a>' if photo else ''
    return f'''<figure>{img}<figcaption><h2>{html.escape(item.get("title", name))}</h2>
<p>{html.escape(item.get("caption", ""))}</p><p class="m">{html.escape(trajectory(d / "volume"))}</p>
<p class="l">{" · ".join(links)}</p></figcaption></figure>'''


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    shutil.copytree(SITE, OUT, ignore=shutil.ignore_patterns('.DS_Store'))
    shutil.copytree(WEB, OUT / 'viewer')
    items = json.loads((SITE / 'collection.json').read_text()) if (SITE / 'collection.json').exists() else []
    cards = '\n'.join(card(i) for i in items)
    (OUT / 'index.html').write_text(f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Interior 2</title>
<style>
:root {{ color-scheme: dark; }}
body {{ margin: 0; background: #0b0b0c; color: #d8d4cc; font: 15px/1.5 Georgia, serif; }}
main {{ max-width: 1100px; margin: 0 auto; padding: 32px 16px 64px; }}
h1 {{ font-weight: normal; font-size: 26px; margin: 0 0 6px; }}
.lede {{ color: #9a958d; max-width: 46em; }}
figure {{ margin: 48px 0; }} img {{ width: 100%; height: auto; display: block; background: #000; }}
h2 {{ font-weight: normal; font-size: 18px; margin: 12px 0 4px; }}
p {{ margin: 4px 0; max-width: 52em; }} .m, .l {{ font: 12px/1.5 ui-monospace, Menlo, monospace; color: #8a867f; }}
a {{ color: #d8d4cc; }}
</style></head><body><main>
<h1>Interior 2</h1>
<p class="lede">One field: recurrent activity deposits and erodes solid, solid and pigment shade what lies beneath,
light regenerates nutrient. Every photograph is lit only by the light field the simulation itself computed,
so the brightness you see is the light the material lived in. <a href="{GITHUB}docs/rendering.md">optical model</a> ·
<a href="https://github.com/emberian/intmat2">source</a></p>
{cards}
</main></body></html>
''')
    print(f'staged {len(items)} specimens into {OUT}')


if __name__ == '__main__':
    main()
