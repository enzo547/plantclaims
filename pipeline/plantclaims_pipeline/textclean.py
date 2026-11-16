"""Source-text cleaning for extraction packets (entities decoded, markup removed, whitespace collapsed), with a codepoint
offset map back to the raw text."""
import html
import re

DEFAULT_TAGS = ('i', 'b', 'u', 'em', 'strong', 'italic', 'bold', 'sup', 'sub', 'sc', 'scp', 'span', 'font', 'p', 'br', 'div',
                'title', 'sec', 'label', 'a', 'xref', 'ext-link', 'tex', 'tex-math', 'inline-formula', 'list', 'list-item',
                'abstract', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6')
NAMESPACES = ('jats', 'mml')                      # jats:p, mml:math ...: tags dropped, text content kept
BLOCK_TAGS = {'p', 'br', 'div', 'title', 'sec', 'label', 'list', 'list-item', 'abstract', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'}
ENTITY = re.compile(r'&(#[0-9]{1,7}|#[xX][0-9a-fA-F]{1,6}|[A-Za-z][A-Za-z0-9]{1,31});')
# A real tag has only name="value" attributes; `a<b and c>d` must not be read as a <b> tag with attributes "and", "c".
TAG = re.compile(r'''</?([A-Za-z][A-Za-z0-9:_-]*)(\s+[A-Za-z_:][-A-Za-z0-9_:.]*\s*=\s*("[^"]*"|'[^']*'|[^\s"'<>=]+))*\s*/?>''')
TAG_LIKE = re.compile(r'</?[A-Za-z][^<>]{0,60}>')      # reported, never removed
MATH = re.compile(r'\$[^$]{1,300}\$')
# "\n" / "\r" / "\t" left as two characters by upstream double escaping; LaTeX commands that start the same way are kept.
LATEX = re.compile(r'\\(nu|neq|ne|nabla|neg|ni|not|notin|rho|right|rightarrow|rm|tau|theta|times|text|textit|textbf|tilde|to|top|triangle)(?![A-Za-z])')


def _allowed(name, tagset):
    name = name.lower()
    return name in tagset or (':' in name and name.split(':', 1)[0] in NAMESPACES)


def _is_block(name):
    return name.lower().rsplit(':', 1)[-1] in BLOCK_TAGS


def _one_pass(text, starts, ends, tagset, stats):
    """One left-to-right scan. `starts`/`ends` give the raw [start, end) span of every input character."""
    out, ostarts, oends = [], [], []
    math = [(m.start(), m.end()) for m in MATH.finditer(text)]

    def emit(chars, a, b):
        for ch in chars:
            out.append(ch); ostarts.append(starts[a]); oends.append(ends[b - 1])
    i, n, changed = 0, len(text), False
    while i < n:
        ch = text[i]
        if ch == '&':
            m = ENTITY.match(text, i)
            if m and html.unescape(m.group(0)) != m.group(0):
                emit(html.unescape(m.group(0)), i, m.end())
                stats['entities'] += 1; i = m.end(); changed = True; continue
        elif ch == '<':
            m = TAG.match(text, i)
            if m:
                if _allowed(m.group(1), tagset):
                    if _is_block(m.group(1)):
                        emit(' ', i, m.end())
                    stats['tags'] += 1; i = m.end(); changed = True; continue
                stats['unknown_tag_like'].append(m.group(0)[:40])
            else:
                loose = TAG_LIKE.match(text, i)
                if loose:
                    stats['unknown_tag_like'].append(loose.group(0)[:40])
        elif ch == '\\' and i + 1 < n and text[i + 1] in 'nrt' and not LATEX.match(text, i) and not any(a <= i < b for a, b in math):
            emit(' ', i, i + 2)
            stats['literal_escapes'] += 1; i += 2; changed = True; continue
        emit(ch, i, i + 1); i += 1
    return ''.join(out), ostarts, oends, changed


def clean_indexed(text, tags=DEFAULT_TAGS, passes=2):
    """Return (clean, starts, ends, stats): clean[j] came from raw[starts[j]:ends[j]].

    Entities are unescaped, whitelisted tags removed (block-level ones become a space), literal "\\n"-style
    sequences become a space, runs of whitespace collapse to one space and the ends are trimmed. A second pass
    handles entity-encoded markup (`&lt;i&gt;`)."""
    tagset = {t.lower() for t in tags}
    stats = dict(entities=0, tags=0, literal_escapes=0, unknown_tag_like=[], chars_raw=len(text))
    starts, ends = list(range(len(text))), list(range(1, len(text) + 1))
    for n_pass in range(max(1, int(passes))):
        if n_pass >= 1:
            stats['unknown_tag_like'] = []     # report what is still left after the last pass only
        text, starts, ends, changed = _one_pass(text, starts, ends, tagset, stats)
        if not changed:
            break
    out, ostarts, oends = [], [], []
    for ch, a, b in zip(text, starts, ends):
        if ch.isspace():
            if out and out[-1] == ' ':
                oends[-1] = b                  # the single kept space spans the whole raw whitespace run
                continue
            ch = ' '
        out.append(ch); ostarts.append(a); oends.append(b)
    while out and out[0] == ' ':
        out.pop(0); ostarts.pop(0); oends.pop(0)
    while out and out[-1] == ' ':
        out.pop(); ostarts.pop(); oends.pop()
    stats['unknown_tag_like'] = sorted(set(stats['unknown_tag_like']))
    stats['chars_clean'] = len(out)
    return ''.join(out), ostarts, oends, stats


def compress(starts, ends):
    """Run-length form of the map: [clean_pos, raw_start, length] for identity runs (each clean char = one raw char,
    consecutive), [clean_pos, raw_start, 1, raw_end] for a clean char that replaces a longer raw token."""
    segs, j, n = [], 0, len(starts)
    while j < n:
        if ends[j] - starts[j] != 1:
            segs.append([j, starts[j], 1, ends[j]]); j += 1; continue
        k = j
        while k + 1 < n and ends[k + 1] - starts[k + 1] == 1 and starts[k + 1] == starts[k] + 1:
            k += 1
        segs.append([j, starts[j], k - j + 1]); j = k + 1
    return segs


def expand(segs, length):
    starts, ends = [None] * length, [None] * length
    for seg in segs:
        if len(seg) == 4:
            starts[seg[0]], ends[seg[0]] = seg[1], seg[3]
        else:
            for d in range(seg[2]):
                starts[seg[0] + d], ends[seg[0] + d] = seg[1] + d, seg[1] + d + 1
    if any(v is None for v in starts):
        raise ValueError('clean_to_raw map does not cover the cleaned text')
    return starts, ends


def raw_span(starts, ends, start, end):
    """Raw [start, end) covering clean[start:end]."""
    if start is None or end is None or end <= start:
        return None, None
    return starts[start], ends[end - 1]
