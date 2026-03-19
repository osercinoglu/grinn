# Design Spec: Chatbot Welcome Message & Settings Default State

**Date:** 2026-03-19
**File affected:** `gRINN_Dashboard/grinn_dashboard.py`

---

## Context

The gRINN Chatbot currently shows no message when a user opens it — the chat area is blank and the Settings panel is expanded by default, pushing the chat input down. Users have no onboarding guidance and may not be aware of important data-privacy and capability constraints before they start typing.

This spec covers three small, targeted changes:

1. Collapse the Settings panel by default so the chat interface is immediately visible
2. Show a static welcome message as the first assistant bubble
3. Make the tutorial links in the welcome message open in a new tab

---

## Changes

### 1. Collapse Settings panel by default

**Location:** `dbc.Collapse` with `id='chat-settings-collapse'` (~line 2219)

**Change:** `is_open=True` → `is_open=False`

The existing toggle callback (`_toggle_chat_settings`) and icon logic handle open/closed states correctly — only the default changes. This also fixes a pre-existing visual bug: the toggle button was hardcoded with `'▼'` (closed icon) at ~line 1986 while the panel was open. With `is_open=False`, the `'▼'` icon is now correct.

---

### 2. Welcome message constants

**Location:** Immediately before the `soft_palette` dict (~line 963)

Add these module-level constants:

```python
# --- Chatbot welcome message ------------------------------------------------
_CHATBOT_TUTORIAL_URL = "https://grinn.bio-cloud.site/tutorial"

_CHATBOT_WELCOME = {
    "role": "assistant",
    "content": {
        "type": "text",
        "text": (
            "👋 **Welcome to the gRINN Chatbot!**\n\n"
            "Before you begin, please read these sections of the "
            "[Tutorial](" + _CHATBOT_TUTORIAL_URL + "):\n\n"
            "- ⚠️ [**Data Privacy Warning**]("
            + _CHATBOT_TUTORIAL_URL + "#a2-data-privacy-warning"
            + ") — your simulation data is transmitted to external LLM APIs\n"
            "- 📋 [**Capabilities & Limitations**]("
            + _CHATBOT_TUTORIAL_URL + "#a3-capabilities-and-limitations"
            + ") — what this chatbot can and cannot do\n\n"
            "To get started, expand ⚙️ **Settings** above to configure your data, "
            "then ask me anything about your interaction energies!"
        ),
    },
}
# ---------------------------------------------------------------------------
```

**Content format:** Uses `{"type": "text", "text": "..."}` — consistent with the rest of the codebase.

---

### 3. Initialize ChatComponent with welcome message

**Location:** `ChatComponent(id='chat', messages=[], ...)` (~line 2244)

**Change:** `messages=[]` → `messages=[_CHATBOT_WELCOME]`

The existing `_on_chat_msg` callback does `messages = list(messages or [])` (~line 3973), so the welcome message is naturally preserved in the chat history — no callback changes needed.

---

### 4. New-tab links via clientside callback

**Why not inline HTML:** `dash_chat` uses `react-markdown ^9.0.3` with `remark-gfm` only — no `rehype-raw`. Raw `<a>` tags in message content are silently stripped. Plain markdown links render correctly but open in the same tab.

**Solution:** A clientside callback that patches `<a>` elements inside `#chat-component-wrap` after the welcome message renders, identical in pattern to the existing PMID link injection callback (~lines 3639–3645).

**Step A — Add a dummy store to the layout** (alongside `pmid_link_dummy` at ~line 990):

```python
welcome_link_dummy = dcc.Store(id='_welcome-link-dummy', storage_type='memory')
```

Add `welcome_link_dummy` to the layout list at ~line 1276 (next to `pmid_link_dummy`).

**Step B — Add the clientside callback** (near the PMID callback, ~line 3645):

```python
app.clientside_callback(
    """
    function(messages) {
        if (!messages || messages.length === 0) return window.dash_clientside.no_update;
        setTimeout(function() {
            var container = document.getElementById('chat-component-wrap');
            if (!container) return;
            container.querySelectorAll('a').forEach(function(a) {
                a.setAttribute('target', '_blank');
                a.setAttribute('rel', 'noopener noreferrer');
            });
        }, 500);
        return window.dash_clientside.no_update;
    }
    """,
    Output('_welcome-link-dummy', 'data'),
    Input('chat', 'messages'),
    prevent_initial_call=False,
)
```

Key decisions:
- **Dummy output:** `_welcome-link-dummy` store (not `chat.messages`) to avoid duplicate Output conflicts with `_on_chat_msg`
- **Target:** `#chat-component-wrap` — same stable container the PMID callback uses; no fragile class-name selectors
- **Scope:** patches all `<a>` tags in the container; safe because all tutorial links should open in a new tab anyway
- **Delay:** 500 ms (vs. 300 ms for the PMID callback) to account for React's initial hydration on page load
- **`prevent_initial_call=False`:** needed so the callback fires on page load when `messages=[_CHATBOT_WELCOME]` is the initial state

---

## Sub-agent Division

The changes split cleanly into two non-overlapping groups that can be dispatched in parallel:

| Agent | Responsibility | Lines touched |
|-------|---------------|---------------|
| A | Add constants (`_CHATBOT_TUTORIAL_URL`, `_CHATBOT_WELCOME`) before `soft_palette`; change `messages=[]` → `messages=[_CHATBOT_WELCOME]`; add `welcome_link_dummy` store to layout; add clientside callback near PMID callback | ~963, ~990, ~1276, ~2244, ~3646 |
| B | Change `is_open=True` → `is_open=False` on `chat-settings-collapse` | ~2219 |

Agent B's change is a single-line edit and could be merged into Agent A, but keeping it separate allows independent review.

---

## Verification

1. Run: `python gRINN_Dashboard/grinn_dashboard.py --help` — confirms no import errors
2. Start: `python gRINN_Dashboard/grinn_dashboard.py /path/to/results/`
3. Open the chatbot panel
4. **Settings collapsed:** Panel is hidden by default; ⚙️ toggle shows `▼` icon; clicking it expands the panel and changes icon to `▲`
5. **Welcome message:** First bubble shows the formatted welcome text with emoji, bold headings, and three hyperlinks
6. **New-tab links:** Click each link — Tutorial, Data Privacy Warning, Capabilities & Limitations — and confirm each opens in a new browser tab
7. **History preserved:** Send a user message; the welcome message remains as the first entry in the chat history
