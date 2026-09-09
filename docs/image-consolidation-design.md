# Consolidating the image tools — design

Status: **implemented**. Companion to `docs/image-pipeline.md`, which describes the state this
replaced; quirk numbers below refer to its §9. Decisions in §1 were made by the repo owner.

## 1. Decisions

| Question | Decision |
|---|---|
| Tool surface | **One tool**, `generate_image`, with an optional list of input images: none = generate, one = edit, several = compose |
| Vocabulary | **Abstract ladder** — purpose word for the engine, `low…max` for quality, aspect ratio + size tier for size; a capability table maps each to the provider's native parameters |
| Delivery | **Sent once** by the tool as a document. The model may *additionally* embed it inline in a rich answer **when the user asks to see it there** |
| Engine choice | User sets a preference in `/settings`; the agent knows it and may **override it from the prompt** (a user who set "fast" but asks for "best quality" gets the better engine) |
| DALL-E | **Removed** (already `[OBSOLETE]` in its own description, quirk 16) |
| GPT Image 2 | **Replaced** by GPT-Image-2.5 |
| Gemini | **Kept, not recommended** — reserved for interleaved story/narrative output |
| Default | `gpt-image-2.5-sunburst` at `auto` quality |

## 2. The tool

One tool replaces five. `image_generator`, `image_editing`, `image_composition` and
`generate_image_dall_e` collapse into `generate_image`; the interleaved story case keeps a separate
tool because its *output shape* differs (alternating text and images), not just its parameters.

```
generate_image
  prompt        string, required
  images        string[], optional   0 → generate · 1 → edit · ≥2 → compose
  engine        auto | fast | best | story          default auto
  quality       auto | low | medium | high | xhigh | max   default auto
  aspect_ratio  1:1 | 3:2 | 2:3 | 16:9 | 9:16 | 4:3 | 3:4 | 21:9 | 5:4 | 4:5   default 1:1
  size          1K | 2K | 4K                         default 2K
  variants      integer 1..4                         default 1
  format        png | jpeg | webp                    default png
  caption       string                               default "Here is your image"
```

Every parameter is optional except `prompt`. One schema, so the copy-paste drift between three
near-identical parameter blocks disappears (quirks 5, 6), and the tool schema in the cached prefix
gets smaller.

`generate_story_images` keeps the Gemini interleaved path, rewritten onto the shared delivery and
result code so it stops duplicating `_deliver_parts` through two throwaway classes (quirk 17).

## 3. Backends and the capability table

One table, in `app/models.py`, is the only place that knows a provider's dialect.

| Engine | Model id (env override) | Generate | Edit | Compose | Size control | Quality | Variants |
|---|---|---|---|---|---|---|---|
| `auto`, `best` | `gpt-image-2.5-sunburst` (`GPT_IMAGE_MODEL`) — most capable, generation and editing | yes | yes | yes (`images.edit`, many files) | pixel `WxH` derived from ratio + tier | ladder, passed through | `n=` in one call |
| `fast` | `gpt-image-2.5-flare` (`GPT_IMAGE_MODEL_FAST`) — fast everyday generation, lower quality | yes | yes | yes | same | ladder | `n=` |
| `story` | `gemini-3.1-flash-image-preview` (`GEMINI_IMAGE_MODEL_FLASH`) | yes | yes | yes (≤3) | ratio + `1K/2K/4K` | none | sequential calls |
| `story` (pro) | `gemini-3-pro-image-preview` (`GEMINI_IMAGE_MODEL_PRO`) | yes | yes | yes (≤3) | ratio + tier | none | sequential calls |

Each row is a `Backend` record: model id, which operations it supports, its quality vocabulary, its
size function, its variant strategy, and its per-call image limit. Dispatch reads the record; no
branch anywhere else compares a mode string. That removes the three different spellings of the same
comparison (quirk 3) and the silently-ignored size parameters in Gemini Normal (quirk 4) — an
unsupported knob is now either mapped or reported, never dropped in silence.

**Quality mapping.** `low|medium|high|auto` pass through to OpenAI unchanged. `xhigh` and `max` are
sent as given; if the API rejects the value the call is retried once at `high` and the result says so.
This is deliberate: neither the released SDK (3.9.0) nor its `main` branch lists `xhigh`/`max` for
images — they exist there only as `ReasoningEffort` for text models — and `developers.openai.com` is
blocked from this environment, so the tiers are unverified. The params are `TypedDict`s with no
runtime validation, so unknown strings reach the wire; the retry is what makes being wrong harmless.
Gemini has no quality parameter: a quality above `high` selects the Pro model, anything else Flash,
and the result notes the substitution.

**Size mapping.** The tool speaks aspect ratio + tier. OpenAI backends run the existing
`_resolve_gpt_size` maths (long edge 1024/2048/3840, clamped to the provider's pixel window). Gemini
takes the ratio and the tier string natively. One function per backend, both covered by tests, which
is where the surprising 4K-shrinks and 1K-grows behaviour (quirk 12) becomes visible instead of
folklore.

## 4. Choosing the engine

Resolution order, highest priority first:

1. **The tool call.** If the agent passed `engine`, that wins — it has read the user's message and may
   legitimately override a "fast" preference when the user asks for the best result.
2. **The user's preference** (`/settings` → Images), when the agent left `engine` at `auto`.
3. **The default**, `auto` → `gpt-image-2.5-sunburst` at `auto` quality.

The tool description states this contract plainly, including the user's current preference, so the
agent can reason about it: *"The user prefers `fast`. Use it unless the request implies otherwise —
if they ask for the best possible result, or the image is a final deliverable, choose `best`."*

`quality` resolves the same way. `auto` is a real OpenAI value that lets the provider pick, so the
default costs nothing and adapts to the prompt; the agent raises it explicitly when the user asks for
quality, and `/settings` carries a ceiling that is enforced in code.

New settings category `image`: `engine` (default `auto`), `quality` (default `auto`), `size`
(default `2K`), `max_variants` (default 4, a hard cap — today `variants` is unvalidated and
`variants=0` silently produces nothing, quirk 8).

## 5. Delivery and the result contract

The tool sends each finished image **once**, as a document, so the file arrives at full quality the
moment it exists. The result string then tells the model, in one fixed sentence, that the user
already has the file and that it should embed it inline **only if the user asked to see it in the
answer** — `![caption](images/<name>)`, which `bot/rich.py:extract_media` resolves. That ends the
unconditional double upload (quirk 35) while keeping the inline path available on request.

Results become structured rather than prose, which is what makes partial success expressible
(quirk 22) and stops a delivery failure from reading as a generation failure (quirk 23):

```
Generated 3 of 4 images with gpt-image-2.5-sunburst (quality high, 2048x1152).
Delivered to the user. Saved: images/sunburst_a1b2.png, images/sunburst_c3d4.png, images/sunburst_e5f6.png
Variant 4 failed: <provider error>
To show one inside your answer (only if the user asked): ![caption](images/sunburst_a1b2.png)
```

**Filenames** get a per-engine prefix and the extension is always derived from the bytes actually
returned, never asserted — which fixes PNG-in-a-`.jpg` for DALL-E, the hard-coded `.png` on GPT edit
and the story tool's fixed `.jpg` (quirk 13), and the `image_<uuid>.jpg` collision with user uploads
(quirk 15). A format the rich renderer cannot inline is still delivered, and the hint is omitted
rather than promising an embed that degrades to alt text (quirk 14).

## 6. Failure handling

Every provider call is wrapped once, in the shared layer: the error is logged with the engine, model
id and resolved parameters (nothing is logged today, quirk 24), classified as transient or permanent,
and turned into a result the model can act on. `candidates[0]` and `result.data[0]` stop being indexed
without a guard (quirk 21). `execute_tool` validates and filters arguments instead of splatting them,
so an unexpected key no longer raises `TypeError` before any handler runs (quirk 25). Blocking file
and PIL work moves onto the thread pool with the SDK calls (quirk 26).

## 7. Scope

**In:** `app/agents/image_tools.py` rewritten around the capability table; the table and model ids in
`app/models.py`; the `image` settings category and its `/settings` menu; the tool-schema and
formatting-guide text in `app/agents/main.py`; tests.

**Out, deliberately:** giving the agent vision over generated images (quirk 39) — a separate change
with its own cost story; retention/pruning of `data/<uid>/images` (quirk 42); per-image cost
accounting in `usage_events`; deleting the dead sandbox hook (quirk 36), which is a one-line cleanup
better done on its own.

**Back-compat.** `generate_image_dall_e` is deleted outright. `image_generator`, `image_editing` and
`image_composition` disappear from the schema; transcripts containing calls to them still replay
fine, because old `tool_use`/`tool_result` pairs are text the model reads, not calls it re-issues.

**Tests** (none exist for image tools today, quirk 14 of the open questions): the capability table
resolves every engine/operation pair; quality and size mapping per backend including the clamps;
engine resolution across the three priority levels; `variants` clamping; extension derived from
bytes; partial-success and error result shapes; the inline hint appearing only for inlineable
formats. The provider SDKs are faked, so the module must import without API keys — which requires
making the two module-level clients lazy, as `bot/clients.py` already does.

## 8. Sunburst vs Flare

Confirmed by the repo owner: **Sunburst is the most capable model** for generation and editing;
**Flare is the fast everyday one** with lower quality. **The two cost the same** — Flare buys latency,
not money.

That price parity decides the guidance the agent reads. Because choosing `fast` saves no money, the
default stays on Sunburst and the tool description frames Flare narrowly: *"`fast` (Flare) trades
quality for speed at the same price — choose it only when the user is waiting on a quick draft or
iterating, never for a final image."* Spend is controlled by `variants` and the quality ladder, not
by engine choice, so `max_variants` in settings is the cost lever.
