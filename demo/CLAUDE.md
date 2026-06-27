# Layer 5: Demo

Parent context: see `../CLAUDE.md`
Depends on: `../api/` (REST API running at localhost:8000)

## Responsibility

A self-contained HTML/JS demo that lets a non-technical audience:
1. Select 2–3 domain agents from a menu
2. Enter or choose a pre-baked question
3. Watch the panel run (streaming updates as turns come in)
4. Read the synthesis output with cited papers

This is the artefact you take to a university department or R&D team.
It must work with `DEMO_MODE=true` (no real corpus, no real ingestion).

---

## Files to build here

```
demo/
├── CLAUDE.md
├── index.html           ← single-file demo (HTML + CSS + JS, no build step)
├── demo_panels.json     ← 3 pre-baked panel examples (question + expected output)
└── README.md            ← how to run the demo for non-technical audience
```

---

## index.html — UI structure

Single HTML file, no framework, no build step. Vanilla JS + fetch.
Target: looks professional enough for a boardroom, runs off `python -m http.server`.

### Layout (top to bottom)

```
┌─────────────────────────────────────────┐
│  SciAgent                    [API: ●]   │  header
├─────────────────────────────────────────┤
│  Panel configuration                    │
│  ┌──────────┐ ┌──────────┐ [+ Agent]   │  agent selector
│  │ Math     │ │ Ops Mgmt │             │
│  └──────────┘ └──────────┘             │
│                                         │
│  Research question                      │
│  ┌─────────────────────────────────┐   │  question input
│  │ What scheduling approaches...   │   │
│  └─────────────────────────────────┘   │
│  [Try an example ▾]   [Run panel →]    │
├─────────────────────────────────────────┤
│  Panel transcript                       │
│  ┌─ Dr. Virtanen (Mathematics) ───────┐│  agent turn cards
│  │ Turn 1 · 6 papers retrieved       ││
│  │ [response text...]                 ││
│  │ 📄 Smith 2022 · Aho 2021          ││
│  └───────────────────────────────────┘│
│  ─── Moderator ──────────────────────  │  moderator divider
│  "The tension here is between..."      │
│  ┌─ Prof. Koskinen (Ind. Mgmt) ──────┐│
│  │ ...                                ││
│  └───────────────────────────────────┘│
├─────────────────────────────────────────┤
│  Synthesis                              │
│  Agreements · Conflicts · Hypotheses   │  synthesis tabs
│  [content per tab]                      │
│  Bibliography                           │
└─────────────────────────────────────────┘
```

---

## Interaction design

### Agent selector
- Show all available agents from `GET /agents` as pills
- User clicks to select (highlighted = selected)
- Max 4 selected, min 2 to enable Run button
- Deselected agents are dimmed, not removed

### Example questions dropdown
Pre-load from `demo_panels.json`. Three examples minimum:

1. **Manufacturing + Maths** (default):
   "What scheduling approaches minimise makespan in a job shop with stochastic
   processing times, and what are the practical barriers to implementing the
   theoretically optimal solution?"

2. **Finance + CS** (Velvoite angle):
   "What does DORA operational resilience regulation require from financial
   institutions' ICT systems, and where does current enterprise architecture
   practice fall short of these requirements?"

3. **Biology + Engineering**:
   "What can engineering learn from biological systems about fault tolerance
   and self-repair in critical infrastructure?"

### Run button behaviour
1. Disable button, show spinner
2. POST to `/panels/run`
3. While waiting: show skeleton cards with "Agent thinking..." placeholder
4. On response: render turns one by one with 300ms stagger (feels like streaming)
5. Render synthesis at the end
6. Re-enable button

Note: real streaming (SSE) is a v2 feature. For demo, fake it with stagger on
the complete response.

### Turn card design
```
┌────────────────────────────────────────────┐
│ Dr. Virtanen (Mathematics)    Turn 1 of 2 │  header (teal for math, purple for ops)
│ 6 papers retrieved                         │  muted metadata
├────────────────────────────────────────────┤
│ Response text with [Smith, 2022] inline    │  body
│ citations rendered as small badges...      │
│                                             │
│ → Open question to panel: "..."            │  last sentence styled differently
├────────────────────────────────────────────┤
│ Cited: Smith 2022 · Aho 2021 · Lee 2020   │  footer with paper links
└────────────────────────────────────────────┘
```

Make each `[Author, Year]` citation in the response text a clickable span that
highlights the corresponding paper in the footer.

### Moderator divider
Between turns, render a horizontal rule with the moderator text:
```html
<div class="moderator-divider">
  <span class="label">Moderator</span>
  <p class="text">The tension here is between...</p>
  <span class="tension-badge">Conflict detected</span>
</div>
```

### Synthesis tabs
Three tabs: Agreements | Conflicts | Hypotheses
Plus a Bibliography accordion below.

Conflicts and Hypotheses are the high-value tabs — open Conflicts by default
since that's what impresses a domain expert audience.

---

## demo_panels.json

Pre-baked panels for offline demo (DEMO_MODE=true returns fake data, use this
as the fallback if API is not running):

```json
{
  "panels": [
    {
      "id": "scheduling",
      "question": "What scheduling approaches minimise makespan...",
      "domains": ["mathematics", "industrial_management"],
      "result": { ... }   // copy of a real PanelResultResponse
    }
  ]
}
```

Run a real panel once, save the JSON, use it for demo offline fallback.

---

## Colour coding for agent turns

Each domain gets a consistent accent colour across the demo:

| Domain | Colour |
|---|---|
| mathematics | teal (#1D9E75) |
| industrial_management | purple (#534AB7) |
| biology | green (#3B6D11) |
| finance | amber (#854F0B) |
| engineering | coral (#993C1D) |
| computer_science | blue (#185FA5) |

Use as left-border accent on turn cards, and as background on agent pills.

---

## README.md (for demo recipients)

Write a one-page README that:
1. Says what SciAgent is in 3 sentences
2. Lists what they need (Python 3.11, an Anthropic API key)
3. Shows exactly 4 commands to get it running
4. Describes the 3 demo panels and what to look for in the output

Tone: peer-to-peer technical, not sales. The audience is researchers and R&D
leads who will respect directness about what the system does and does not do yet.

---

## Testing this layer

```bash
# Start API
cd /path/to/sciagent
uvicorn api.main:app --port 8000 &

# Serve demo
cd demo && python -m http.server 3000

# Open browser
open http://localhost:3000
```

Manual checks:
- [ ] Agent pills load from /agents endpoint
- [ ] Selecting 2 agents enables Run button
- [ ] Example question populates input
- [ ] Run button POSTs correctly, shows loading state
- [ ] Turns render in sequence with stagger
- [ ] Moderator dividers appear between turns
- [ ] Synthesis tabs switch correctly
- [ ] Bibliography accordion opens
- [ ] Works with DEMO_MODE=true (no real corpus)
