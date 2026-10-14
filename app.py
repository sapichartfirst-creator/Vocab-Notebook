"""Vocab Notebook - พิมพ์คำศัพท์ แล้วได้ POS / Synonym / ความหมายไทย / ประโยคตัวอย่างอัตโนมัติ

Run:
    pip install flask
    set ANTHROPIC_API_KEY=sk-ant-...   (แนะนำ: ได้ประโยคคุณภาพดีกว่า; ไม่ตั้งก็ได้ จะใช้ API ฟรีแทน)
    python app.py   ->  http://127.0.0.1:5000
"""
import json
import os
import re
import sqlite3
import urllib.error
import urllib.parse
import urllib.request

from flask import Flask, jsonify, render_template_string, request

app = Flask(__name__)
DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vocab.db")
MODEL = os.environ.get("VOCAB_MODEL", "claude-haiku-5-5")


# ---------- DB ----------
def db():
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    return con


def init_db():
    with db() as con:
        con.execute(
            """CREATE TABLE IF NOT EXISTS words (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                word TEXT UNIQUE NOT NULL,
                pos TEXT, synonyms TEXT, thai TEXT,
                example TEXT, example_th TEXT,
                source TEXT,
                created TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"""
        )


# ---------- Enrichment ----------
def http_json(url, data=None, headers=None, timeout=15):
    headers = {"User-Agent": "Mozilla/5.0 VocabNotebook/1.0", **(headers or {})}
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def enrich_claude(word):
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None
    prompt = (
        f'English word or phrase: "{word}"\n'
        "Return ONLY a JSON object with keys:\n"
        '"pos": parts of speech, comma separated (e.g. "noun, verb"),\n'
        '"synonyms": up to 5 English synonyms, comma separated,\n'
        '"thai": concise Thai meaning(s), mention which POS if several,\n'
        '"example": one natural, short sentence for everyday conversation using the word,\n'
        '"example_th": Thai translation of that sentence.\n'
        'If the word is not valid English, return {"error": "not a word"}.'
    )
    body = json.dumps(
        {"model": MODEL, "max_tokens": 600, "messages": [{"role": "user", "content": prompt}]}
    ).encode()
    res = http_json(
        "https://api.anthropic.com/v1/messages",
        body,
        {"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
        timeout=30,
    )
    text = "".join(b.get("text", "") for b in res["content"])
    m = re.search(r"\{.*\}", text, re.S)
    d = json.loads(m.group(0))
    if d.get("error"):
        raise ValueError(d["error"])
    d["source"] = "claude"
    return d


def translate_th(text):
    url = "https://api.mymemory.translated.net/get?" + urllib.parse.urlencode(
        {"q": text, "langpair": "en|th"}
    )
    return http_json(url)["responseData"]["translatedText"]


def enrich_free(word):
    """Fallback: dictionaryapi.dev + MyMemory (ไม่ต้องใช้ key)"""
    try:
        entries = http_json(
            "https://api.dictionaryapi.dev/api/v2/entries/en/" + urllib.parse.quote(word), timeout=8
        )
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise ValueError("ไม่พบคำนี้ในพจนานุกรม")
        entries = []  # บริการล่ม/บล็อก -> ใช้เฉพาะคำแปล
    except Exception:
        entries = []
    pos, syns, example = [], [], ""
    for e in entries:
        for m in e.get("meanings", []):
            if m["partOfSpeech"] not in pos:
                pos.append(m["partOfSpeech"])
            for s in m.get("synonyms", []):
                if s not in syns:
                    syns.append(s)
            for d in m.get("definitions", []):
                for s in d.get("synonyms", []):
                    if s not in syns:
                        syns.append(s)
                if not example and d.get("example"):
                    example = d["example"]
    if not example:
        example = f'I learned the word "{word}" today.'
    try:
        thai = translate_th(word)
    except Exception:
        thai = ""
    try:
        example_th = translate_th(example)
    except Exception:
        example_th = ""
    return {
        "pos": ", ".join(pos),
        "synonyms": ", ".join(syns[:5]),
        "thai": thai,
        "example": example,
        "example_th": example_th,
        "source": "free",
    }


def enrich(word):
    try:
        d = enrich_claude(word)
        if d:
            return d
    except ValueError:
        raise
    except Exception as e:  # network/API error -> fallback
        app.logger.warning("claude failed: %s", e)
    return enrich_free(word)


# ---------- Routes ----------
@app.get("/api/words")
def list_words():
    with db() as con:
        rows = con.execute("SELECT * FROM words ORDER BY id DESC").fetchall()
    return jsonify([dict(r) for r in rows])


@app.post("/api/words")
def add_word():
    word = (request.json or {}).get("word", "").strip().lower()
    if not word or len(word) > 60 or not re.fullmatch(r"[a-z][a-z\s'\-]*", word):
        return jsonify(error="กรุณาพิมพ์คำศัพท์ภาษาอังกฤษ"), 400
    with db() as con:
        if con.execute("SELECT 1 FROM words WHERE word=?", (word,)).fetchone():
            return jsonify(error=f'"{word}" มีอยู่ในคลังแล้ว'), 409
    try:
        d = enrich(word)
    except ValueError as e:
        return jsonify(error=str(e)), 404
    except Exception as e:
        return jsonify(error=f"ดึงข้อมูลไม่สำเร็จ: {e}"), 502
    with db() as con:
        cur = con.execute(
            "INSERT INTO words (word,pos,synonyms,thai,example,example_th,source) VALUES (?,?,?,?,?,?,?)",
            (word, d.get("pos", ""), d.get("synonyms", ""), d.get("thai", ""),
             d.get("example", ""), d.get("example_th", ""), d.get("source", "")),
        )
        row = con.execute("SELECT * FROM words WHERE id=?", (cur.lastrowid,)).fetchone()
    return jsonify(dict(row)), 201


@app.delete("/api/words/<int:wid>")
def del_word(wid):
    with db() as con:
        con.execute("DELETE FROM words WHERE id=?", (wid,))
    return "", 204


@app.get("/")
def index():
    return render_template_string(PAGE)


PAGE = """<!doctype html>
<html lang="th">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Vocab Notebook</title>
<script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-slate-50 text-slate-800 min-h-screen">
<main class="max-w-3xl mx-auto px-4 py-10">
  <h1 class="text-3xl font-bold text-indigo-700">📒 Vocab Notebook</h1>
  <p class="text-slate-500 mt-1">พิมพ์คำศัพท์ภาษาอังกฤษ แล้วระบบเติม POS, Synonym, ความหมายไทย และประโยคตัวอย่างให้อัตโนมัติ</p>

  <form id="form" class="mt-6 flex gap-2">
    <input id="word" autocomplete="off" autofocus placeholder="เช่น reluctant, grab, schedule"
      class="flex-1 rounded-xl border border-slate-300 px-4 py-3 focus:outline-none focus:ring-2 focus:ring-indigo-400">
    <button id="btn" class="rounded-xl bg-indigo-600 hover:bg-indigo-700 text-white px-6 py-3 font-medium disabled:opacity-50">เพิ่มคำ</button>
  </form>
  <p id="msg" class="mt-2 text-sm h-5"></p>

  <input id="q" placeholder="🔍 ค้นหาในคลัง..." class="mt-4 w-full rounded-xl border border-slate-200 bg-white px-4 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-300">
  <p id="count" class="mt-3 text-sm text-slate-400"></p>
  <div id="list" class="mt-2 space-y-3"></div>
</main>

<script>
const $ = id => document.getElementById(id);
let words = [];
const esc = s => (s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

function render() {
  const q = $('q').value.trim().toLowerCase();
  const shown = words.filter(w => !q || [w.word, w.thai, w.synonyms].join(' ').toLowerCase().includes(q));
  $('count').textContent = `ทั้งหมด ${words.length} คำ` + (q ? ` · พบ ${shown.length}` : '');
  $('list').innerHTML = shown.map(w => `
    <article class="bg-white rounded-2xl shadow-sm border border-slate-100 p-5">
      <div class="flex items-start justify-between gap-3">
        <div>
          <h2 class="text-xl font-semibold text-slate-900">${esc(w.word)}
            <button onclick="speak('${esc(w.word)}')" title="ฟังเสียง" class="ml-1 text-base">🔊</button></h2>
          <div class="mt-1 flex flex-wrap gap-1">
            ${(w.pos||'').split(',').filter(x=>x.trim()).map(p=>`<span class="text-xs px-2 py-0.5 rounded-full bg-indigo-50 text-indigo-700">${esc(p.trim())}</span>`).join('')}
          </div>
        </div>
        <button onclick="del(${w.id})" title="ลบ" class="text-slate-300 hover:text-red-500 text-lg">✕</button>
      </div>
      <p class="mt-3 text-lg text-emerald-700">${esc(w.thai)}</p>
      ${w.synonyms ? `<p class="mt-2 text-sm"><span class="text-slate-400">Synonyms:</span> ${esc(w.synonyms)}</p>` : ''}
      <div class="mt-3 rounded-xl bg-slate-50 p-3">
        <p class="italic">“${esc(w.example)}”</p>
        <p class="text-sm text-slate-500 mt-1">${esc(w.example_th)}</p>
      </div>
    </article>`).join('') || '<p class="text-center text-slate-400 py-10">ยังไม่มีคำศัพท์</p>';
}

function speak(t) {
  const u = new SpeechSynthesisUtterance(t); u.lang = 'en-US'; speechSynthesis.speak(u);
}
async function load() { words = await (await fetch('/api/words')).json(); render(); }
async function del(id) {
  if (!confirm('ลบคำนี้?')) return;
  await fetch('/api/words/' + id, {method: 'DELETE'}); load();
}
$('q').oninput = render;
$('form').onsubmit = async e => {
  e.preventDefault();
  const word = $('word').value.trim(); if (!word) return;
  $('btn').disabled = true; $('btn').textContent = 'กำลังหา...';
  $('msg').className = 'mt-2 text-sm h-5 text-slate-400'; $('msg').textContent = 'กำลังดึงข้อมูล...';
  try {
    const r = await fetch('/api/words', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({word})});
    const d = await r.json();
    if (!r.ok) throw new Error(d.error);
    $('word').value = ''; $('msg').textContent = '';
    words.unshift(d); render();
  } catch (err) {
    $('msg').className = 'mt-2 text-sm h-5 text-red-500'; $('msg').textContent = err.message;
  } finally {
    $('btn').disabled = false; $('btn').textContent = 'เพิ่มคำ'; $('word').focus();
  }
};
load();
</script>
</body>
</html>"""

init_db()

if __name__ == "__main__":
    app.run(debug=True)
