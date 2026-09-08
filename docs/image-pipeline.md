# Image generation, editing and composition — current state

Status: **historical**. This describes `app/agents/image_tools.py` *before* the consolidation and is
kept as the record of what was replaced and why. The current tool is described in
`docs/image-consolidation-design.md`. No proposals; where the old tool schema and code disagreed,
both are recorded and the disagreement is called out.

## 1. Overview

`ImageTools` (`app/agents/image_tools.py:36`) is a single provider class exposing **five** tools to the
model — `image_generator`, `generate_image_dall_e`, `generate_multimodal_image_and_text`,
`image_editing`, `image_composition` (`image_tools.py:52-271`) — over **three** providers: Google
Gemini (two model ids, selected by the free-text mode strings `Normal`/`Pro`), OpenAI GPT Image 2
(mode string `GPT`), and OpenAI DALL-E 3 (its own legacy tool, marked `[OBSOLETE]` in its own
description at `:115`). Everything runs **in the bot process**: the SDK calls are hopped to the shared
thread pool with `asyncio.to_thread` (`:381-383`, `:441`, `:550`, `:563`, `:611`), the file writes and
the `PIL.Image.open` calls are not (`:412-413`, `:488-489`, `:520`, `:568`, `:619`, `:659`). Nothing
about image tools is sandboxed: the secure-container hook that was written for them targets a method
name that no longer exists (`secure_container/tool_integrator.py:317`), so it is a no-op that still
reports success. Outputs are written under `data/<uid>/images/` and pushed to the chat by the tool
itself as Telegram **documents** (`:378-379`); the tool result then tells the model it may *also*
embed the picture inline with Markdown (`_inline_hint`, `:290-302`), which `bot/rich.py` resolves into
a second upload. The class is constructed fresh per turn as `ImageTools(user_id, sender)`
(`agents/main.py:626`, assigned at `:650`) and mkdirs its two directories on construction
(`image_tools.py:47`, `:50`).

## 2. The five tools

| Tool (`image_tools.py:`) | Providers reachable | What it does | Key parameters |
|---|---|---|---|
| `image_generator` (`:54`, impl `:575-605`) | Gemini Flash / Gemini Pro / GPT Image 2 | Text → image(s); saves to `images/`, sends each as a document | `prompt`*, `style`, `model`, `aspect_ratio`, `resolution`, `gpt_quality`, `gpt_output_format`, `variants`, `caption` |
| `generate_image_dall_e` (`:114`, impl `:437-456`) | DALL-E 3 only | Legacy text → one image; `n=1` hard-coded, `response_format="b64_json"` | `prompt`*, `size`, `quality`, `caption` |
| `generate_multimodal_image_and_text` (`:143`, impl `:458-496`) | Gemini Flash only (hard-coded `:464`) | Wraps the prompt in a fixed story template and interleaves text messages + images | `prompt`*, `style` |
| `image_editing` (`:162`, impl `:498-543`) | Gemini Flash / Gemini Pro / GPT Image 2 | Edits one existing image from the workspace | `prompt`*, `image_path`*, `model`, `aspect_ratio`, `resolution`, `gpt_quality`, `variants`, `caption` |
| `image_composition` (`:215`, impl `:628-683`) | Gemini Flash / Gemini Pro / GPT Image 2 (via the **edit** endpoint) | Composes from ≥2 existing images | `prompt`*, `image_paths`*, `model`, `aspect_ratio`, `resolution`, `gpt_quality`, `variants`, `caption` |

`*` = in the schema's `required` list. Only three of the five carry the `model` tri-state
(`:68`, `:175`, `:232`); the DALL-E tool has its own `size`/`quality` vocabulary and the story tool has
no knobs at all beyond `style`. `execute_tool` (`:421-432`) is a five-branch `if/elif` chain that
forwards `**tool_args` verbatim to the private coroutine and falls through to
`f"Unknown tool: {tool_name}"`; it has no `try`/`except` of its own, so an argument the signature does
not accept raises `TypeError` before any of the tools' own error handling runs.

## 3. Provider matrix

| | Gemini "Normal" | Gemini "Pro" | GPT Image 2 | DALL-E 3 |
|---|---|---|---|---|
| Model id / env var | `gemini-3.1-flash-image-preview` / `GEMINI_IMAGE_MODEL_FLASH` (`models.py:88`) | `gemini-3-pro-image-preview` / `GEMINI_IMAGE_MODEL_PRO` (`models.py:89`) | `gpt-image-2-2026-04-21` / `GPT_IMAGE_MODEL` (`models.py:90`) | `dall-e-3` / `DALLE_MODEL` (`models.py:91`) |
| Selected by | anything that is not `pro`/`gpt` (`:399` catch-all) | `model.lower() == "pro"` for the id (`:399`), `model == "Pro"` exactly for the config (`:388`) | `model.lower() == "gpt"` (`:508`, `:579`, `:644`) | its own tool name |
| Generation | yes (`:588-596`) | yes | yes (`:607-626`) | yes (`:437-456`) |
| Editing | yes (`:519-530`) | yes | yes (`:545-573`) | no |
| Composition | yes, ≤3 images (`:655-670`) | yes, ≤3 images | yes — same `images.edit` call with more files, no upper bound enforced (`:644-653`) | no |
| Size / aspect control | **none reaches the API** — `_gemini_config` returns `None` (generation) or a `response_modalities`-only config (edit/compose) (`:385-395`) | `types.ImageConfig(aspect_ratio=…, image_size=…)` with the raw `"1K"`/`"2K"`/`"4K"` string (`:391`, `:393`) | `_resolve_gpt_size` derives a `WxH` pixel string from `resolution`+`aspect_ratio` (`:341-372`) | `size` enum `1024x1024` / `1024x1792` / `1792x1024` (`:123-127`) |
| Quality control | none | none | `quality` = `gpt_quality`, forwarded unvalidated (`:555`, `:613`) | `quality` enum `standard`/`hd` (`:128-132`) |
| Variants | N sequential `generate_content` calls (`:588`, `:525`, `:665`) | same | generation: one call with `n=variants` (`:581`, `:613`); edit/compose: N sequential calls (`:511-516`, `:647-652`) | none; `n=1` hard-coded (`:448`) |
| Output format | extension derived from `inline_data.mime_type`, `jpeg`→`jpg` (`:408-410`) | same | generation: `output_format` → `ext = "jpg" if output_format == "jpeg" else output_format` (`:615`); edit: **always `.png`** (`:567`), and `output_format` is never sent (`:555`) | PNG bytes written to a `.jpg` name (`:450-452`) |
| Where the call is made | bot process | bot process | bot process | bot process |
| Threading | `asyncio.to_thread(genai_client.models.generate_content, …)` (`:381-383`); no timeout, no retry, no `http_options` (`:24`) | same | `asyncio.to_thread(openai_client.images.generate/…_edit)` (`:611`, `:563`); SDK defaults apply (its own retries and timeout) | `asyncio.to_thread(openai_client.images.generate, …)` (`:441`) |

Both SDK clients are module-level singletons built at import time from `OPENAI_API_KEY` /
`GOOGLE_API_KEY` (`:21-24`), shared process-wide and distinct from the lazy clients in
`bot/clients.py` and from the second `genai.Client` in `agents/video_tools.py:19`.

## 4. Parameter reference

Effective defaults are the **Python signature** defaults — the Anthropic API does not apply JSON
`default` keys, so an omitted argument takes the signature value. Where the two disagree it is
flagged.

### `image_generator` — signature `image_tools.py:575`

| Parameter | Type / enum | Schema default | Signature default | What it actually does |
|---|---|---|---|---|
| `prompt` | string, required (`:59`) | — | — | Sent raw to GPT (`:581`); for Gemini wrapped by `style` (`:583`) |
| `style` | string, free text (`:63`) | `photorealistic` | `photorealistic` | Only `f"Create a {style} style image: {prompt}"` when truthy **and** `!= "photorealistic"` (`:583`). Inert at its own default; dropped entirely in GPT mode (the branch returns at `:581`) |
| `model` | enum `Normal`/`Pro`/`GPT` (`:68-73`) | `Normal` | `Normal` | Routes to `_gpt_image_generate` or Gemini; description interpolates the two Gemini ids but names the GPT one only in prose |
| `aspect_ratio` | enum of 10 (`:74-79`) | `16:9` | `16:9` | GPT: input to `_resolve_gpt_size`. Pro: passed to `ImageConfig`. **Normal: ignored** (`:395`) |
| `resolution` | enum `1K`/`2K`/`4K` (`:80-85`) | `2K` | `2K` | GPT: long-edge target 1024/2048/3840, unknown → 2048 (`:347-348`). Pro: raw string to `image_size`. **Normal: ignored** |
| `gpt_quality` | enum `low`/`medium`/`high`/`auto` (`:86-91`) | `auto` | `auto` | Forwarded verbatim as `quality` to `images.generate` (`:613`); never validated |
| `gpt_output_format` | enum `png`/`jpeg`/`webp` (`:92-97`) | **`jpeg`** (JSON key `:96`), description says `png` (`:94`) | **`png`** | Sent as `output_format` (`:613`) and decides the saved extension (`:615`). Three-way disagreement; the effective value when omitted is `png` |
| `variants` | integer, enum `[1,2,3,4]` (`:98-103`) | 1 | 1 | GPT: `n=variants`, one call. Gemini: `range(variants)` sequential calls. Never clamped in code |
| `caption` | string (`:104-108`) | `Here is your generated image` | same | Telegram caption; `" (variant i/N)"` appended when >1 (`:593`, `:621`) |

### `generate_image_dall_e` — signature `image_tools.py:437`

| Parameter | Type / enum | Schema default | Signature default | Notes |
|---|---|---|---|---|
| `prompt` | string, required (`:119`) | — | — | Passed straight through |
| `size` | enum `1024x1024`/`1024x1792`/`1792x1024` (`:123-127`) | **no JSON `default` key**; prose says "use by default '1024x1024'" | `1024x1024` | |
| `quality` | enum `standard`/`hd` (`:128-132`) | **no JSON `default` key**; prose says `standard` | `standard` | |
| `caption` | string (`:133-137`) | `Here is your image` | same | |

The call is `images.generate(model=DALLE_MODEL, prompt, size, quality, response_format="b64_json", n=1)`
(`:441-449`) — the only place in the file that pins `response_format`.

### `generate_multimodal_image_and_text` — signature `image_tools.py:458`

| Parameter | Type | Schema default | Signature default | Notes |
|---|---|---|---|---|
| `prompt` | string, required (`:148`) | — | — | Rewritten as `f"Generate a story about {prompt} in a {style} style. For each scene, generate an image."` (`:462`) |
| `style` | string, free text (`:152`) | `3d digital art` | `3d digital art` | Interpolated into that same template |

No `model`, `aspect_ratio`, `resolution`, `variants` or `caption` exist for this tool; the model is
hard-coded to Flash and the config to `response_modalities=["Text","Image"]` (`:463-467`). Its schema
sells it as the general "image + text in one request" tool, but the story framing is unconditional.

### `image_editing` — signature `image_tools.py:498`

| Parameter | Type / enum | Schema default | Signature default | Notes |
|---|---|---|---|---|
| `prompt` | string, required (`:167`) | — | — | Gemini: first element of the `(prompt, image)` tuple (`:526`) |
| `image_path` | string, required (`:171`) | — | — | Description says "a full path to an image file in the user's directory", but the resolver prefers workspace-relative and bare names (`:276-288`) |
| `model` | enum `Normal`/`Pro`/`GPT` (`:175-180`) | `Normal` | `Normal` | |
| `aspect_ratio` | enum of 10 (`:181-186`) | `16:9` | `16:9` | Same three-way behaviour as above |
| `resolution` | enum `1K`/`2K`/`4K` (`:187-192`) | `2K` | `2K` | |
| `gpt_quality` | enum, 4 values (`:193-198`) | `auto` | `auto` | Description is shorter than `image_generator`'s: no "(fast drafts)"/"(best quality…)" annotations |
| `variants` | integer, enum `[1,2,3,4]` (`:199-204`) | 1 | 1 | Both modes loop sequentially |
| `caption` | string (`:205-209`) | `Here is your edited image` | same | |

No `style` and **no `gpt_output_format`** — a model that copies that key from `image_generator`
triggers a `TypeError` at the `**tool_args` splat (`:429`).

### `image_composition` — signature `image_tools.py:628`

| Parameter | Type / enum | Schema default | Signature default | Notes |
|---|---|---|---|---|
| `prompt` | string, required (`:220`) | — | — | Gemini: **last** element of `[*images, prompt]` (`:660`) |
| `image_paths` | array of string, `minItems: 2`, no `maxItems` (`:224-231`) | — | — | Description claims "For Normal/Pro: 2-3 images. For GPT: up to 10 images." Runtime checks are `<2` (`:641`) and, for Normal/Pro only, `>3` (`:655`) |
| `model` | enum `Normal`/`Pro`/`GPT` (`:232-237`) | `Normal` | `Normal` | GPT arm restates the 10-image claim |
| `aspect_ratio` / `resolution` / `gpt_quality` | as above (`:238-255`) | `16:9` / `2K` / `auto` | same | |
| `variants` | integer, enum `[1,2,3,4]` (`:256-261`) | 1 | 1 | GPT: N sequential `_gpt_image_edit` calls |
| `caption` | string (`:262-266`) | `Here is your composed image` | same | |

### Internal helper defaults (never exercised — all callers pass explicit values)

`_gpt_image_edit(prompt, image_paths, size="2048x1152", quality="auto", caption="Here is your edited image")`
(`:545`) and
`_gpt_image_generate(prompt, size="2048x1152", quality="auto", output_format="jpeg", caption=…, n=1)`
(`:607`) — note the helper's `output_format` default is `jpeg` while `_image_generator`'s is `png`.

### `_resolve_gpt_size` (`image_tools.py:341-372`)

`target_map = {"1K":1024, "2K":2048, "4K":3840}`, unknown resolution silently → 2048 (`:348`); the
target goes on the long edge and the short edge is `round(target*ratio/16)*16` (`:352-357`); total
pixels are then repaired into `655_360..8_294_400` (up-scale with a 2-decimal `ceil` factor, down-scale
with `int` snapping, `:359-368`) and each edge clamped to `16..3840` (`:370-371`). Computed output for
the 30 in-enum combinations:

```
1K  1:1=1024x1024  2:3=688x1024   3:2=1024x688   3:4=768x1024   4:3=1024x768
    4:5=816x1024   5:4=1024x816   9:16=608x1088↑ 16:9=1088x608↑ 21:9=1248x528↑
2K  1:1=2048x2048  2:3=1360x2048  3:2=2048x1360  3:4=1536x2048  4:3=2048x1536
    4:5=1632x2048  5:4=2048x1632  9:16=1152x2048 16:9=2048x1152 21:9=2048x880
4K  1:1=2880x2880↓ 2:3=2336x3520↓ 3:2=3520x2336↓ 3:4=2480x3312↓ 4:3=3312x2480↓
    4:5=2560x3216↓ 5:4=3216x2560↓ 9:16=2160x3840 16:9=3840x2160 21:9=3840x1648
```

`↑` = the min-pixel repair fired (so "1K" produces a **larger** long edge than 1024 and, for 21:9, a
2.36:1 ratio instead of 2.33:1); `↓` = the max-pixel clamp fired (so "4K" reaches a 3840 edge only for
16:9, 9:16 and 21:9). All 30 satisfy the API's stated constraints; the docstring's "ratio ≤ 3:1"
(`:344-345`) is never checked and holds only because the enum stops at 21:9.

## 5. The pipeline, step by step

**a. Prompt build.** GPT generation sends the raw `prompt` (`:581`). Gemini generation prefixes the
style when it is not the default (`:583`). Gemini edit/compose send the prompt unmodified. The story
tool rewrites it into the fixed story sentence (`:462`). Nothing else touches the text — there is no
sanitisation, translation or length cap.

**b. Model dispatch.** `if model.lower() == "gpt"` returns into the OpenAI helper (`:508`, `:579`,
`:644`). Otherwise `_gemini_model_name` picks Pro for `model.lower() == "pro"` and Flash for everything
else (`:397-399`), and `_gemini_config` builds the config — but on the **exact literal** `"Pro"`
(`:388`), producing four request shapes: Pro+text → `response_modalities` + `image_config`; Pro+no-text
→ `image_config` only (the local `modalities` variable computed at `:387` is unused on this branch);
non-Pro+text → `response_modalities` only; non-Pro+no-text → `None`, and `image_generator` then omits
the `config` kwarg entirely (`:590-591`).

**c. Input resolution.** `_resolve_image_path` (`:276-288`) calls
`agents.paths.resolve_under(base_path, image_path, fallback_dirs=("images","downloads","documents"))`.
`resolve_under` (`paths.py:41-80`) strips `./`, maps the sandbox spelling `/home/runner/workspace/...`
(`paths.py:14`, `:21-25`) and the host spelling `data/<uid>/...` (`:29-37`) onto the base, resolves
symlinks and refuses anything outside `data/<uid>`; with `must_exist=True` it returns the first
*existing* candidate — the primary path, then `base/<dir>/<basename>` per fallback dir. On `None`
`_resolve_image_path` returns `images_path / Path(image_path).name`, a usually non-existent path, so
the caller reports `f"Error: The image at path {image_path} does not exist."` echoing the model's raw
spelling (`:506`, `:638`). `image_editing` checks one path; `image_composition` resolves in a loop and
**returns on the first missing path** (`:635-639`), then checks `<2` (`:641`), then branches to GPT,
then checks `>3` (`:655`).

**d. Input preparation.** `_prepare_image_for_edit` (`:304-331`), run via `asyncio.to_thread`
(`:519`, `:550`, `:658`), opens the file with PIL and rewrites it **only** when
`max(img.size) > MAX_IMAGE_RESOLUTION_EDIT` (default 4096, parsed at import, `:25`) **or** the detected
format is outside `JPEG_FORMATS = {"JPEG","JPG"}` (`image_utils.py:10`). In that case:
`exif_transpose` → `thumbnail(LANCZOS)` (only if resizing) → `convert("RGB")` → save
`images/edit_input_<uuid4>.jpg` at `quality=90, optimize=True`, tracked in the caller's `temp_paths`.
Otherwise the original path is returned untouched — so EXIF rotation is applied only on the temp path.
Any exception is swallowed with a warning and the original file is used (`:328-331`). Gemini then gets
**PIL image objects** (`PIL.Image.open` on the event loop, `:520`, `:659`); GPT gets **open binary file
handles** built inside the nested `_edit()` (`:553`) and closed in its own `finally` (`:559-560`).
`google.genai.types.Image` is never used here.

**e. API call.** Gemini: `asyncio.to_thread(genai_client.models.generate_content, **kwargs)` (`:381-383`)
with `contents` in three different shapes — `[prompt]` for generation (`:589`), the tuple
`(prompt, source_image)` for editing (`:526`), `[*images, prompt]` for composition (`:660`), and a bare
string for the story tool (`:465`). GPT generation:
`images.generate(model, prompt, quality, output_format, size, n)` (`:611-614`). GPT edit/compose:
`images.edit(model, prompt, quality, size, image=<handle or list>)` (`:552-557`) — no `output_format`,
no `n`, no mask. DALL-E: `images.generate(..., response_format="b64_json", n=1)` (`:441-449`).

**f. Response decode.** Gemini goes through `_deliver_parts` (`:401-416`), which walks
`response.candidates[0].content.parts` (indexed with no guard at `:468`, `:526`+`:529`, `:592`+`:595`,
`:666`+`:669`) and tests only `'text' in part.model_fields_set and part.text` (`:404`) and
`'inline_data' in part.model_fields_set` (`:407`); every other part kind — including a `thought` part —
is either dropped or, if it carries text, sent to the chat verbatim. GPT and DALL-E assume base64:
`base64.b64decode(item.b64_json)` (`:620`), `result.data[0].b64_json` (`:569`, only the first item),
`response.data[0].b64_json` (`:452`); there is no `b64_json`-vs-URL branch.

**g. File write.** All under `data/<uid>/images/`, all with a fresh `uuid4`, written with a blocking
`open()/write()` on the event loop:

| Path | Name | `image_tools.py:` |
|---|---|---|
| Gemini generate | `generated_<uuid4>.<ext from mime>` | `:411` with prefix at `:595` |
| Gemini edit | `transformed_<uuid4>.<ext from mime>` | `:411`, `:529` |
| Gemini compose | `composed_<uuid4>.<ext from mime>` | `:411`, `:669` |
| Gemini story | `story_image_<uuid4>.jpg` (mime captured at `:475`, never read) | `:487` |
| GPT generate | `gpt_generated_<uuid4>.<ext from output_format>` | `:615-618` |
| GPT edit / compose | `gpt_edited_<uuid4>.png` (hard-coded) | `:567` |
| DALL-E | `image_<uuid4>.jpg` (PNG bytes) | `:450` |
| transient edit input | `edit_input_<uuid4>.jpg`, deleted in a `finally` | `:324`, `:333-339` |

**h. Delivery.** Every image goes through `_send_image` → `sender.send_document(str(path), caption)`
(`:378-379`), immediately after being written, i.e. before the model's answer text. Gemini text parts
go through `_send_text` → `sender.send_markdown` (`:374-376`, `:405`) as their own chat messages.
Captions are truncated to 1024 with an ellipsis by `ChatSender` (`bot/sender.py:32`, `:67-70`).

**i. The string returned to the model.** Prose, never JSON:

- DALL-E success (`:454`): `"Image generated and sent to user via telegram successfully.\n\nFile also saved to: {path}\n\n"` + inline hint.
- Gemini generate (`:587`, `:598-602`): starts empty, accumulates any text parts, then
  `"Generated {n} image(s) and sent to user.\nSaved to:\n  - <path>…\nImage generation completed successfully.\n"` + hint; empty case
  `"Warning: No image was generated. The model only provided text response.\n"`.
- Gemini edit (`:524`, `:532-536`): header `"Image transformation results:\n\n"`, text label
  `"Text explanation:"`, tail `"Edited {n} image(s) …"` / `"Warning: No transformed image was generated…"`.
- Gemini compose (`:664`, `:672-676`): header `"Image composition results:\n\n"`, label
  `"Composition details:"`, tail `"Composed {n} image(s) …"` / `"Warning: No composed image was generated…"`.
- GPT generate (`:623-624`): `"Generated {n} image(s) with GPT Image 2 and sent to user.\nSaved to:\n…"` + hint.
- GPT edit/compose (`:571`): `"Image edited with GPT Image 2 and saved to: {path} and sent to user.\nImage editing completed successfully.\n"` + hint, once **per variant**, joined with `"\n"` (`:517`, `:653`).
- Story (`:477`, `:483-493`): `"Text and images were generated and successfully sent to user as telegram messages:\n\n"`, then `f"Part {part_count}:…"` and `f"Image for part {image_count}: {path}."`, closing with `"\n\nEnd of messages.\nWrite answer to user using this text, remove under the hood details about styling info, formatting, or something like that, and translate it to user language if needed."` — no inline hint.
- Errors, one broad `except Exception` per tool, no logging, no traceback:
  `"Error generating image: {e}"` (`:456` **and** `:605` — DALL-E and Gemini generate share the literal),
  `"Error generating multimodal story: {e}"` (`:496`), `"Error editing image: {e}"` (`:539`),
  `"Error editing image with GPT Image 2: {e}"` (`:573`), `"Error composing image: {e}"` (`:679`),
  `"Error generating image with GPT Image 2: {e}"` (`:626`).

## 6. Storage and delivery

Directories are created in `__init__`: `base_path = Path("./data") / str(user_id)` and
`images_path = base_path / "images"`, both `mkdir(parents=True, exist_ok=True)` (`:45-50`) — cwd-relative,
so the process must run with `cwd=app/`. Everything the tools produce, plus user uploads and video
frames, shares that one flat directory (see §8). Nothing prunes it: the only `unlink` in the module is
`_cleanup_temp_images` (`:337`).

Delivery is always `send_document` (`:378-379`), never `send_photo` — `ChatSender.send_photo` exists
(`bot/sender.py:281-282`) and is used only by the rich-message media path. `ChatSender._kw()` carries
just `chat_id` and `message_thread_id` (`bot/sender.py:98-102`); `reply_to_message_id` is stored
(`:92`) but consumed only by `react()` (`:294`), so no tool-sent file is a Telegram reply.

Captions: the model's `caption` verbatim for a single image, `f"{caption} (variant {i}/{N})"` when more
than one (`:513-514`, `:527`, `:593`, `:621`, `:650`, `:667`); the story tool ignores `caption`
entirely and uses `f"Image {image_count}."` (`:490`).

`_inline_hint` (`:290-302`) appends
`"The user already received the file(s). To also show an image inline in your answer, write ![caption](images/x.jpg), …"`
listing at most **three** paths, made relative to `base_path.resolve()` (falling back to
`images/<name>` on `ValueError`). It is emitted by DALL-E (`:454`), Gemini generate (`:600`), edit
(`:534`), compose (`:674`), GPT edit (`:571`) and GPT generate (`:624`) — every path except the story
tool.

If the model follows that hint, the answer text is processed by `bot/rich.py:extract_media`
(`:278-316`) against `ChatSender.media_root = data/<uid>` (set once per turn at
`bot/agent_runner.py:123`). `_resolve_local` (`rich.py:261-275`) re-resolves the target through
`resolve_under` with **its own** fallback dirs `("images","videos","downloads","documents")`
(`rich.py:220`), accepts only `PHOTO_EXTENSIONS = {".jpg",".jpeg",".png",".webp"}` /
`VIDEO_EXTENSIONS = {".mp4",".mov",".m4v"}` (`rich.py:218-219`) and enforces `MAX_PHOTO_BYTES = 10 MB`
/ `MAX_VIDEO_BYTES = 50 MB` and `MAX_INLINE_MEDIA = 10` (`rich.py:221-223`). Resolved items become
`![alt](tg://photo?id=mN)` and are uploaded **again** with the rich message
(`rich.py:310-312`, `bot/sender.py:182-186`); unresolvable or over-limit ones are silently replaced by
their alt text (`rich.py:304-308`); in the non-rich tiers the syntax is stripped to the alt text and
the file is sent afterwards as a separate photo (`bot/sender.py:228-241`). So an embedded generated
image reaches the user twice and is uploaded twice — no `file_id` is reused between the document send
and the inline copy. The system prompt hedges the other way: `RICH_FORMATTING_GUIDE`
(`agents/main.py:65`) says "Images a tool already delivered as a file need not be repeated"; the
legacy guide (`:72`) says an `![caption](images/x.jpg)` reference "is sent as a separate photo after
the text".

Tool result strings are also user-visible: `ToolCall.done` keeps the first `RESULT_CHARS = 800`
characters (`agents/trace.py:15`, `:76-81`) and `bot/status.py:186-194` renders them into the
collapsible "Tool calls (N)" block of the end-of-turn rich summary when `trace.keep_summary` is on
(default true, `bot/settings.py`; gated at `bot/agent_runner.py:133`), shrinking to 300 or 160
characters if the summary exceeds `SUMMARY_BYTES` (`bot/status.py:208-216`). The legacy fallback
summary shows no result text. Independently, `agents/main.py:335` prints every complete tool result to
stdout.

## 7. Wiring and configuration

**Constants.** `app/models.py:88-91` holds the four ids, each an `_env()` override of the same name
read once at import after `load_dotenv()` (`models.py:19`); `.env.example:46-49` documents them
commented out. `MAX_IMAGE_RESOLUTION_EDIT` is **not** a `Config` field — `image_tools.py:25` reads it
directly with a bare `int()` at import, while its sibling `MAX_IMAGE_RESOLUTION_VISION` is
`bot/config.py:50`, `:81` and is used only by `bot/media.encode_image` (`media.py:108-129`);
`.env.example:78-80` ships both. `MODEL_PRICES` (`models.py:203-210`) contains only Claude models, so
`estimate_cost` returns `None` for every image model.

**Construction and dispatch.** Four manual points in `agents/main.py`: import (`:11`), the `None` slot
(`:102`), `tools.extend(self.image_tools.tools_schema)` in fixed position 6 of `get_tools_schema`
(`:252-253`, filtered by `allowed_tools` at `:262-263`), and the awaited branch in `execute_tool`
(`:284-285`) that matches by membership in the provider's schema names — the sixth branch, fifth
`elif`. `ChainOfThoughtAgent.__init__` builds `ImageTools(user_id, sender)` (`:626`) and assigns it
onto the `AgentAnthropic` (`:650`), once per turn.

**Sandbox patch (no-op).** `ToolIntegrator.patch_image_tools` (`tool_integrator.py:308-329`) only acts
`if hasattr(image_tools_class, "_generate_image")` (`:317`); the class defines
`_generate_image_dall_e`, `_gpt_image_generate` and `_gpt_image_edit`, never `_generate_image`, so
nothing is wrapped. `patch_all_tools` still sets `patched_tools["image_tools"] = True` (`:757`) and
logs the summary (`:784`), so startup reports the image tools as patched. The would-be wrapper's
signature `(self, prompt, size="1024x1024", quality="standard", caption=…)` (`:320`) matches only the
old DALL-E shape. `SecureToolWrapper` still lists `"image_tools"` in `SECURE_TOOL_TYPES` (`:27`) and
`"_generate_image"` in `SECURE_TOOL_METHODS` (`:53`), and `wrap_image_operation` (`:221-238`) and
`needs_secure_execution` (`:76-95`) are never called from anywhere.

That block is nevertheless the module's **first import**: `import_module(".image_tools", …)` at
`tool_integrator.py:753` runs inside `initialize_secure_containers()` (`secure_container/main.py:59-60`,
called from `main_bot.py:37`), before `import agents.main` at `main_bot.py:42`. So the module-level
clients (`image_tools.py:21-24`) and the `MAX_IMAGE_RESOLUTION_EDIT` parse (`:25`) execute during
sandbox setup. The per-tool guard there is `except (AttributeError, ImportError)` (`:758`), which
catches none of an `openai.OpenAIError`, a genai key error or a `ValueError` from a bad env value;
those fall to the outer `except Exception` at `:788-790`, which logs `"Error in patch_all_tools"` and
returns `False`. `secure_container/main.py:60-63` logs "Failed to patch some tools…" and still returns
`True`, so startup proceeds and dies at `main_bot.py:42` with a raw traceback. The embeddings and
system_tools patch blocks (`tool_integrator.py:761-780`) sit after the image block inside the same
`try`, so they are skipped whenever it raises.

**Subagents.** `image_tools` is one of `PROVIDER_SLOTS` (`agents/subagent.py:22-25`), and the child
receives the parent's **exact instance** — same `ChatSender`, same workspace (`:100-101`). The
`run_subagent` `tools` parameter is a list of tool *names* filtered against `_available_tool_names()`
(`:88-94`, `:128-130`), so it takes `image_generator`, not the group name `image_tools`; an unmatched
list collapses to `None`, i.e. all tools. Depth is 1 (`child.subagents = None`, `:102`). The subagent
system prompt says "Do not send messages to the user yourself unless the task explicitly says so"
(`:34`), which the image tools cannot honour. The child's calls come out of a slice of the parent's
`TurnBudget` and are charged back at `:151-153`.

**Quota and stats.** `execute_tool` writes one `stats_events` row via
`stats_tracker.track_tool_used(user_id, tool_name)` before dispatch (`agents/main.py:269-270`), even
for an unknown name; `get_user_action_count` counts **every** row (`app/stats.py:462-470`), so one
image tool call = one of the default 50 actions per 30 days regardless of `variants`. The quota is
checked once per incoming message in `AuthMiddleware`, never mid-turn. Token/cost accounting
(`bot/agent_runner.py:67-84`) only walks Anthropic `usage` objects, and `estimate_cost` returns `None`
for image models, so `cost_usd` is stored as NULL and image spend is invisible in `/stats`.

**Concurrency.** All `tool_use` blocks of one assistant message run through `asyncio.gather`
(`agents/main.py:343`) with no image-specific cap; each blocking call takes one thread of the shared
executor sized by `THREAD_POOL_SIZE` (32, `bot/app.py:69-76`), which the sandbox-slot wait, Whisper,
ffmpeg and Tavily also use. `MAX_SANDBOX_CONTAINERS` never applies. Each `tool_use` block costs one
unit of the loop counter and of `TurnBudget` (`agents/main.py:463`, `:485-487`), bounded by the
per-user `tools.max_iteration` (default 20, `:384-386`).

**Prompt surface.** Tool schemas are sent unconditionally on every request
(`api_kwargs["tools"] = self.get_tools_schema()`, `agents/main.py:424`) inside the cached prefix
(`cache_control` ephemeral at `:429-430`), even when `tools.enabled` is false. They have a second
consumer: the OpenAI critique prompt interpolates `{self.get_tools_schema()}` as text
(`agents/main.py:148`), so all five image definitions ship to `gpt-5.6-terra` on every critique
iteration. Because three descriptions are f-strings over `GEMINI_IMAGE_MODEL_FLASH/PRO`
(`image_tools.py:70`, `:177`, `:234`), changing either env var invalidates the cached prefix.

**Prompt guidance.** The four system prompts say nothing about image tools; all mode guidance
("ALWAYS use Normal…", six occurrences) lives in the tool descriptions. The only other mentions in
`agents/main.py` are the two formatting guides (`:65`, `:72`), the complexity classifier bullet
"Needs web search, file operations, code execution, image generation" (`:761`) which routes every image
request to the full tool loop, and two explicit quality-gate carve-outs: the critique prompt's
"If user ask to generate image, video, or transform it, always answer need_rewrite_answer = False,
because you must not check the result of this actions." (`:120`) and the judge prompt's
"…always answer \"Yes\" because you must not check the result of this actions." (`:201`).

**Settings and tests.** There is no image category in `bot/settings.py` or `settings_ui.py` — no
default mode, no cap on variants or resolution, no way to disable the expensive paths. No test
constructs `ImageTools`; coverage is `tests/test_models.py:41-44`, `:116-117` (the four ids and that
they get no reasoning options), `tests/test_paths.py` (the resolver) and `tests/test_rich_media.py`
(inline embedding). The module cannot be imported without `OPENAI_API_KEY` and `GOOGLE_API_KEY`
because of the import-time clients.

## 8. Adjacent consumers

**Inbound photos** (`bot/handlers/messages.py:325-396`). Each ref is downloaded to
`temp_photos/photo_<uuid><ext>` (the extension forced to `.jpg` unless it is in
`image_utils.IMAGE_EXTENSIONS = (".jpg",".jpeg",".heic",".heif")`, `:346-348`), normalised by
`media.prepare_downloaded_image_for_vision` (`bot/media.py:132-136` — a byte copy when already JPEG,
otherwise `save_image_as_jpeg` at quality 90 with **no** downscale), and copied to
**`data/<uid>/images/image_<uuid>.jpg`** (`:355`) — the same pattern `_generate_image_dall_e` writes
(`image_tools.py:450`). Each image is then described twice, sequentially: Claude
(`media.describe_image_anthropic`, `media.py:139-153`, `media_type` hard-coded `image/jpeg`) then
OpenAI (`media.describe_image_openai`, `:156-170`, returns `response.choices[0].message.content` with
no `or ""`), each with a `track_describe_used` row (`messages.py:361`, `:364`). Only the base64 payload
is capped, at `MAX_IMAGE_RESOLUTION_VISION` (1024) inside `encode_image` (`media.py:108-129`). On a
per-image failure the entry gets `"path": "error_path"` (`messages.py:370`), which is rendered into the
prompt as a real path (`:372-379`) alongside the sentence "The images are saved in your workspace; to
show one inline in your answer, write `![caption](images/<file name>)`". Albums are buffered per
`(chat_id, media_group_id)` and flushed 10 s after the last item into one job (`:413-447`).

**Inbound video** (`messages.py:188-268`). The file is saved as `data/<uid>/videos/video_<uuid>.mp4`
regardless of container (`:198-199`); four ffmpeg frames at 10/40/60/85 % land in
`temp_audio/screenshot_<uuid>.jpg` (`media.py:385-411`), are copied to
**`data/<uid>/images/video_frame_<uuid>.jpg`** (`messages.py:219`), and are described by
`VIDEO_FRAMES_MODEL` from the temp copies (`:224`, with the Whisper transcript passed only for video
*notes*). The agent prompt lists the saved frame paths (`:242-255`).

**Video tools** (`agents/video_tools.py`). Three tools take image paths and resolve them with
`_resolve_path` → `resolve_under(base, path, fallback_dirs=("images","videos","downloads"))`, falling
back to `videos/<name>` (`:238-243`) — a **different** fallback set from the image tools'. They feed
`google.genai.types.Image.from_file` directly with no format check, resizing or conversion:
`_image_to_video_generator` (`:308-316`, error `"Error: The image at path … does not exist."`),
`_video_from_reference_images` (`:337-346`, `"Error: The reference image at path …"`),
`_video_interpolation_generator` (`:368-381`, first/last frame, `"Error: The first/last frame at
path …"`). Results go to `data/<uid>/videos/veo3_*.mp4` and are delivered with `send_video`
(`:264-273`), with no inline hint in the result string.

**Other readers of `images/`.** `bot/rich.py` (fallback dirs
`("images","videos","downloads","documents")`, `:220`) and
`file_ops.send_file_to_user` (fallback dirs `("downloads","images","videos","documents")`,
`file_ops.py:558`, which does guard `self.sender is None` at `:567-568`). Four resolvers, four
different fallback tuples, no shared constant.

## 9. Quirks, asymmetries and dead code

1. **`gpt_output_format` has three disagreeing defaults.** JSON `"default": "jpeg"`
   (`image_tools.py:96`), description "Default is 'png'" (`:94`), signature `"png"` (`:575`) — and the
   helper's own default is `"jpeg"` (`:607`). The effective value when omitted is `png`, which decides
   the saved extension (`:615`). A consolidation must pick one.
2. **Edit/compose have no output-format knob and always write `.png`.** `images.edit` is called without
   `output_format` (`:555-556`) and the result is saved as `gpt_edited_<uuid>.png` (`:567`) whatever the
   bytes are. The model can request jpeg/webp on the generate path only.
3. **Three different comparisons of the same mode string.** `model.lower() == "gpt"` (`:508`, `:579`,
   `:644`), `model.lower() == "pro"` (`:399`), `model == "Pro"` (`:388`). A lowercase `"pro"` selects the
   Pro model **and silently drops `aspect_ratio`/`resolution`**. Any unrecognised value falls through to
   Flash with no warning and no log of the resolved id.
4. **`aspect_ratio` and `resolution` are inert in the default mode.** `_gemini_config` builds
   `ImageConfig` only for `"Pro"` (`:388-395`), and `image_generator` in Normal mode sends **no config at
   all** (`:584`, `:590-591`). The schema publishes defaults `16:9` / `2K` on a tool whose description
   says "ALWAYS use Normal".
5. **The GPT arm of every model description is a frozen marketing string.** `:70`, `:177`, `:234` name
   "OpenAI GPT Image 2" in prose while the Normal/Pro arms interpolate the live ids from `models.py`;
   overriding `GPT_IMAGE_MODEL` does not change what the model is told.
6. **The four mode blocks are copy-paste with drift.** `:68-108` vs `:175-209` vs `:232-266`:
   `gpt_quality` loses its "(fast drafts)"/"(best quality, slow and expensive)" annotations in
   editing/composition; `gpt_output_format` and `style` exist only on `image_generator`.
7. **`variants` means two different things.** GPT generation → `n=variants`, one call (`:581`, `:613`);
   everything else → N sequential calls (`:588`, `:525`, `:665`, `:511-516`, `:647-652`). Same schema
   text, order-of-magnitude difference in latency and cost. Each GPT edit variant also re-runs
   `_prepare_image_for_edit` and re-uploads the same source (`:550`).
8. **Neither `variants` nor `image_paths` is validated in code.** The JSON enum `[1,2,3,4]` is advisory:
   `variants=0` makes zero calls and returns the misleading "No image was generated" warning (`:602`),
   and a large value loops unbounded. `image_paths` reaches a bare `for` (`:635`), so a string iterates
   characters.
9. **The GPT composition limit is documented but never enforced.** "up to 10 images" at `:216`, `:226`,
   `:234`; the `>3` guard sits **after** the GPT early return (`:644` vs `:655`), so 11+ paths are opened
   and uploaded in one `images.edit` call and fail as a raw provider error.
10. **GPT composition is the edit endpoint.** `_image_composition`'s GPT branch delegates to
    `_gpt_image_edit` (`:648`), inheriting the hard-coded `.png` and the single-result read.
11. **`_resolve_gpt_size` fails asymmetrically.** An unknown resolution degrades silently to 2048
    (`:348`); a malformed `aspect_ratio` raises inside the *caller's* try, so `image_generator` reports
    `"Error generating image: invalid literal for int()…"` (`:605`), never the "with GPT Image 2"
    variant. `"1:0"` gives `ZeroDivisionError`. The docstring's "ratio ≤ 3:1" (`:344-345`) is never
    checked.
12. **"4K" rarely means a 3840 edge and "1K" is not 1024.** See the table in §4: the max-pixel clamp
    shrinks seven of ten 4K ratios, and the min-pixel repair enlarges three 1K ratios and changes 21:9
    to 2.36:1 (`:359-368`).
13. **Three of four save paths assert the extension instead of deriving it.** DALL-E writes PNG bytes to
    `.jpg` (`:450-452`), the story tool writes any mime to `.jpg` while capturing the real one at `:475`
    (`:487-489`), GPT edit hard-codes `.png` (`:567`). Only `_deliver_parts` (`:408-411`) and
    `_gpt_image_generate` (`:615`) derive it.
14. **Derived extensions can be unembeddable.** `_deliver_parts` accepts any mime subtype (`:408`), but
    `bot/rich.py:218` inlines only `.jpg/.jpeg/.png/.webp` — an `image/gif` or `image/heic` part is saved,
    sent as a document, advertised by `_inline_hint`, and then silently degraded to alt text
    (`rich.py:304-308`).
15. **DALL-E output collides with user uploads by name.** `images/image_<uuid4>.jpg` is written both at
    `image_tools.py:450` and at `bot/handlers/messages.py:355`; every other generator uses a
    distinguishing prefix. Provenance is unrecoverable from the path.
16. **`generate_image_dall_e` is `[OBSOLETE]` and unconditionally shipped.** `:113-141` sits in the tool
    schema of every request (`agents/main.py:252-253`, no filter), inside the cached prefix and in the
    critique prompt text, and the model can still pick it. Overriding `DALLE_MODEL` is a trap: the call
    pins `response_format="b64_json"` and the `standard`/`hd` enum (`:447`, `:131`), which a modern image
    model rejects.
17. **`generate_multimodal_image_and_text` duplicates `_deliver_parts` via two throwaway classes.**
    `TextPart`/`InlineDataPart` (`:27-34`) are used only at `:470-491`. That copy: does not test text
    truthiness (`:472` vs `:404`, so a `None` text becomes the string `"None"`), hard-codes `.jpg`, uses a
    fixed caption, emits no inline hint, and mislabels images — `part_count` counts all parts while
    `image_count` counts images, so `"Part 3"` and `"Image for part 2"` can be the same scene (`:481`,
    `:486`, `:491`).
18. **Tool results carry instructions to the calling model.** `_inline_hint`'s "The user already
    received the file(s)…" (`:302`) and the story epilogue's "Write answer to user using this text,
    remove under the hood details…" (`:493`) are two inconsistent conventions; both are also rendered
    into the user-visible turn summary (`agents/trace.py:15`, `bot/status.py:186-194`).
19. **Gemini text is delivered twice.** `_deliver_parts` sends every non-empty text part to the chat
    (`:405`) *and* appends it to the tool result (`:406`), which the model then usually paraphrases. In
    Normal-mode edit/compose the config always requests `["Text","Image"]` (`:521`, `:661`), so an extra
    "Text explanation:" message is posted per variant.
20. **`_deliver_parts` filters nothing but `text`/`inline_data`.** No `thought` / `thought_signature`
    check (`:403-407`), so a thinking part from a Pro-class model is published to the chat verbatim; and
    `part.inline_data` / `.data` are dereferenced without a None check (`:408`, `:413`).
21. **`candidates[0]` is indexed without a guard.** `:468`, `:526`+`:529`, `:592`+`:595`, `:666`+`:669`.
    A safety block or empty response becomes `"Error generating image: list index out of range"`;
    `finish_reason` and `prompt_feedback` are never read, so a refusal is indistinguishable from an
    outage.
22. **Partial success collapses to an error.** The single `try` wraps the whole variant loop and the
    Telegram sends (`:578-605`, `:503-539`, `:633-679`, `:610-626`). A failure on variant *k* discards
    `all_saved_paths` for the images already written **and already delivered**; the model is told the
    tool failed and can never reference or embed those files.
23. **A delivery failure looks like a generation failure.** `_send_image` is inside the same broad
    `except`, and `ImageTools` never guards `self.sender is None` (`:376`, `:379`) — unlike
    `file_ops.send_file_to_user` (`file_ops.py:567-568`). The documented tool contract says providers must
    tolerate a `None` sender.
24. **Nothing is logged on failure.** Every error path returns `str(e)` with no `logger.exception`
    (`:456`, `:496`, `:539`, `:573`, `:605`, `:626`, `:679`); a 429, a content-policy refusal and a
    Telegram upload error are indistinguishable to both the operator and the model. There are no retries
    or timeouts in this file; the Gemini client is built with no `http_options` (`:24`) and, because the
    call runs in `to_thread`, `/cancel` and `TURN_TIMEOUT_SECONDS` cannot reclaim the thread.
25. **`execute_tool` splats with no filtering.** `:421-432`; a hallucinated key (e.g.
    `gpt_output_format` on `image_editing`) raises `TypeError` outside every `try` in this file and
    surfaces as an `is_error` tool_result from `run_tool_batch` (`agents/main.py:329-334`).
26. **Blocking I/O on the event loop.** The SDK calls are threaded but the image writes are not
    (`:412-413`, `:488-489`, `:568`, `:619`), nor are `PIL.Image.open` at `:520` and the list
    comprehension at `:659`. Multi-megabyte 4K writes block every user.
27. **`_prepare_image_for_edit` re-implements `image_utils.save_image_as_jpeg`.** `:304-331` vs
    `image_utils.py:18-37`; the module only imports `JPEG_FORMATS` (`:15`). The copies differ in their
    default cap (4096 vs 0), so the vision path and the edit path can drift. Line `:313` also rebinds
    `img` inside the `with`, leaving the `exif_transpose` copy unclosed, and EXIF rotation is applied
    **only** when a temp file is produced — identical images above and below the cap are oriented
    differently.
28. **Every non-JPEG edit input is flattened.** `needs_conversion` is true for anything outside
    `{"JPEG","JPG"}` (`:310`), so PNG/WebP alpha is composited away by `convert("RGB")` (`:321-322`) and
    the bytes re-encoded at quality 90 — on the same tool whose description sells GPT mode for
    "screenshot/UI mockup editing" (`:177`). Preparation failures are swallowed (`:328-331`) and the
    unprepared original is uploaded, so the 4096 cap is best-effort only.
29. **Temp inputs live in the user-visible gallery.** `edit_input_<uuid>.jpg` is written into
    `images/` (`:324`) — the first fallback dir for every resolver — and removed only in a `finally`
    (`:333-339`, warning-only on failure). A crash leaks them permanently.
30. **Two independent cleanup mechanisms.** `image_editing`/`image_composition` keep an outer
    `temp_paths` cleaned at `:540-543` / `:680-683`; `_gpt_image_edit` keeps its own (`:548`, `:562-565`).
    On the GPT edit branch the outer list is always empty (`:508-517` returns before `:519`), and the
    inner cleanup only guards the `_edit()` call — a cancellation during the preparation comprehension at
    `:550` leaks.
31. **Path resolution has no `is_file` check and no refusal signal.** `resolve_under` only tests
    `.exists()` (`paths.py:72-79`), so a directory passes (`""` resolves to `images/` itself, `"."` to the
    workspace root) and fails later inside PIL as a generic error. And because `_resolve_image_path`
    collapses `None` into a fake `images/<basename>` (`:288`), a containment refusal and a genuinely
    missing file produce the identical message (`:506`, `:638`). The `must_exist` fallback also retries by
    **basename**, so `"../41/cat.jpg"` silently resolves to the user's own `images/cat.jpg` when one
    exists.
32. **Composition aborts on the first bad path, before the count checks.** `:635-639` returns inside the
    resolve loop; `<2` is checked at `:641`, `>3` only at `:655` after the GPT return. Which error the
    model sees depends on whether the paths exist.
33. **`image_path`'s description contradicts the resolver.** "a full path to an image file in the user's
    directory" (`:173`) steers the model toward absolute host paths that `resolve_under` has to
    special-case or refuse.
34. **Fallback dirs differ per consumer.** `image_tools` `("images","downloads","documents")` (`:287`),
    `video_tools` `("images","videos","downloads")` (`video_tools.py:242`), `rich`
    `("images","videos","downloads","documents")` (`rich.py:220`), `file_ops`
    `("downloads","images","videos","documents")` (`file_ops.py:558`). A bare name that resolves for one
    tool can fail for another.
35. **Images are delivered as documents while the inline copy is a photo.** `send_document` (`:378-379`)
    vs `InputMediaPhoto` (`rich.py:236-238`) — two presentations of one asset in one turn, with different
    limits: the document send has no 10 MB cap, the inline copy does (`rich.py:222`, `:272-274`), so a 4K
    generation silently loses its inline rendering.
36. **`patch_image_tools` is dead and reports success.** `tool_integrator.py:317` guards on a method that
    does not exist; `:757` still sets `patched_tools["image_tools"] = True` and `:784` logs it. It is also
    the single point where a bad image env var takes the whole patcher down (§7) and skips the embeddings
    and system_tools blocks at `:761-780`.
37. **`SecureToolWrapper` advertises a boundary that does not exist.** `"image_tools"` in
    `SECURE_TOOL_TYPES` (`:27`), `"_generate_image"` in `SECURE_TOOL_METHODS` (`:53`),
    `wrap_image_operation` (`:221-238`) and `needs_secure_execution` (`:76-95`) all unreferenced.
38. **The sandbox can read generated images but not write next to them.**
    `ContainerManager.ensure_user_directory` chmods only the top-level user dir, non-recursively
    (`container_manager.py:142-159`); `images/` is created by the bot process, which runs as root
    (`app/Dockerfile` has no `USER`), while the sandbox runs as `runner`
    (`secure_container/Dockerfile:130`). The "generate, then post-process with PIL in the sandbox"
    workflow fails on permissions, and `file_ops.delete_file`/`create_directory` — the two sandbox-patched
    file ops, which build `self.base_path / name` with **no** `resolve_under` (`file_ops.py:361-368`,
    `:370-383`) — cannot remove them either.
39. **The agent can never look at an image.** `read_file` maps to `read_text_file`, which opens with
    `encoding='utf-8'` (`file_ops.py:265`, `:352-359`), so reading a JPEG returns a decode error. Vision
    exists only in the inbound handler pipeline (`bot/media.py:139-170`) and is not a tool. No image tool
    can verify its own output, compare variants, or re-describe an older file.
40. **Both quality gates are switched off for image turns, in prompt text.** `agents/main.py:120`
    (critique) and `:201` (judge).
41. **Transcript pruning erases the only record of a filename.** The `Saved to:` block and the inline hint
    are the sole place a path is recorded; `cap_tool_results` (`agents/transcript.py:167`) truncates them
    and `clear_old_tool_results` (`:181-193`) replaces results older than `keep_tool_results_turns`
    (default 3) with `CLEARED_MARKER` (`:36`). After three turns the model must fall back to `list_files`,
    where quirk 15 makes generated images and uploads indistinguishable.
42. **Nothing ever prunes `data/<uid>/images`.** Generated images, uploads, four frames per video and any
    leaked `edit_input_*` accumulate forever in one flat directory inside the sandbox bind mount; no
    retention policy, size cap or per-user disk quota exists anywhere.
43. **The shared sender has no ordering.** Parallel `tool_use` blocks (`agents/main.py:343`) and
    subagents holding the parent's instance (`subagent.py:100-101`) interleave their `send_document` and
    `_send_text` calls, and both label their outputs `"(variant 1/2)"` — the suffix is per call.
44. **The rich formatting guide's example is coupled to one prefix.** `agents/main.py:65` shows
    `![caption](images/generated_x.jpg)` while the real prefix set is
    `{generated, transformed, composed, image, story_image, gpt_generated, gpt_edited}`, and one of them
    is `.png`.
45. **Inbound and outbound format sets disagree.** Users may upload only
    `.jpg/.jpeg/.heic/.heif` as documents (`image_utils.py:9`), but the tools produce `.png`/`.webp` and
    `rich.py:218` embeds them — a generated PNG cannot be re-uploaded to the bot as a document.

## 10. Open questions a consolidation must answer

1. **One tool or several?** Generate / edit / compose share every parameter except `image_path(s)`,
   `style` and `gpt_output_format`. Does one tool with an optional image list replace all three, and what
   happens to `generate_multimodal_image_and_text` (multi-image + interleaved text, no knobs) and to
   `generate_image_dall_e` (removed, or kept as an alias)?
2. **What is the capability model?** Today the mode string is compared in five places with three
   different spellings (quirk 3) and there is no record of which mode supports which knob. A single
   capability table has to decide what happens when a knob is unsupported: silently drop (today's
   Normal-mode behaviour), coerce, or return an error string.
3. **Who owns size?** Gemini Pro takes `aspect_ratio` + `"1K"/"2K"/"4K"`, GPT takes a pixel `WxH` derived
   locally, DALL-E takes three fixed sizes, Gemini Normal takes nothing. Is the tool's vocabulary
   aspect+tier (and the pixel math per provider), or explicit pixels?
4. **What is the output format contract?** Pick one default for `gpt_output_format` (quirk 1), decide
   whether edit/compose get the knob, and decide whether extensions are always derived from the bytes or
   the requested format — including for DALL-E and story images (quirk 13). The answer constrains what
   `bot/rich.py` can inline (quirk 14).
5. **Does `variants` mean images or API calls?** And is it clamped in code, budgeted against
   `TurnBudget`, or charged per produced image in `stats_events` (§7)?
6. **How are images delivered — document, photo, or once?** Today every image is sent as a document and
   the tool then invites a second, contradicting the system prompt (§6). Deciding this also decides
   whether `_inline_hint` survives, and whether the >10 MB case is handled or silently degraded.
7. **What does a partial failure return?** A result shape that can carry "3 of 4 succeeded, here are the
   paths" would end quirk 22; that implies structured results rather than prose, which in turn changes
   what the turn summary renders (`bot/status.py:186-194`) and what pruning destroys (quirk 41).
8. **Filenames and provenance.** Does the consolidated tool keep per-provider prefixes, adopt one prefix,
   or record provenance out of band? The `image_<uuid>.jpg` collision with uploads (quirk 15) and the lack
   of any retention policy (quirk 42) are the practical constraints.
9. **Input normalisation.** Should `_prepare_image_for_edit` stay lossy JPEG-only (quirk 28), become
   format-preserving for PNG inputs, and be unified with `image_utils.save_image_as_jpeg` (quirk 27)?
   What does it do about EXIF, directories, decompression bombs and unreadable files (quirks 27, 31)?
10. **Errors, timeouts and retries.** Where do provider errors get logged and classified (transient vs
    permanent vs safety block), what timeout do the Gemini and OpenAI calls get, and does the sandbox
    thread pool need an image-specific semaphore (quirks 24, 26, §7)?
11. **Can the agent see images?** Without a vision tool no consolidation removes the blindness in quirk 39;
    with one, edit loops and variant selection become possible and the tool contract changes.
12. **What happens to the dead sandbox hook?** Deleting `patch_image_tools`, fixing it, or leaving it
    changes whether the startup log tells the truth and, via quirk 36, whether a bad image env var still
    aborts the patcher and skips the embeddings/system_tools blocks.
13. **Do settings get an image category?** A default mode, a variants cap and a resolution ceiling would
    be the first user-facing controls (`bot/settings.py`), and would decide whether
    `MAX_IMAGE_RESOLUTION_EDIT` becomes a `Config` field.
14. **Can the module be imported without API keys?** The import-time clients (`image_tools.py:21-24`) are
    why test coverage is zero and why the sandbox initializer can crash on an image env var; lazy clients
    are a precondition for testing the consolidated tool.
15. **Cost accounting.** Image spend is invisible today (`MODEL_PRICES` has no image models, `cost_usd` is
    NULL). Does the consolidated tool report per-image cost into `usage_events`, and is the quota charged
    per call or per produced image?
