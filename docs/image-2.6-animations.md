# GPT Image 2.6: animated GIFs, SVG, markup — design

Status: **implemented; GIFs tested live** (a photo animated with `gpt-image-2.6-flare`: 8 frames, 512x704 from a
3:4 photo; GIF preview access is enabled on the account). The other checks in section 4 are still open. Builds on `docs/image-consolidation-design.md`.

GPT Image 2.6 (`gpt-image-2.6-sunburst` / `gpt-image-2.6-flare`, released 2026-11-04) keeps 2.5's per-token
prices and adds four things the bot now uses: animated GIF output, SVG output, a `complexity` knob and
markup-guided edits. The parameter facts below come from the openai-python 3.46.0 parameter docs (generated
from OpenAI's API spec); launch-week blog numbers are marked as such.

## 1. What changed for still images

| | Before | Now |
|---|---|---|
| Default models | `gpt-image-2.5-sunburst` / `-flare` | `gpt-image-2.6-sunburst` / `-flare` (`GPT_IMAGE_MODEL`, `GPT_IMAGE_MODEL_FAST`; set them back to 2.5 to roll back) |
| Capabilities | implicit | `models.IMAGE_MODEL_FEATURES` by model family: 2.6 → gif, svg, complexity, markup; 2.5 → markup, input_fidelity |
| `input_fidelity=high` on edits | always, dropped after a refusal | pre-2.6 models only ("for gpt-image-2, omit this parameter") |
| Size | custom `WxH` always | custom `WxH`; a refusal retries once at the closest standard size and is remembered (the SDK lists arbitrary sizes for models up to 2.5 only) |
| New | | `format: "svg"` (new drawings, n=1), `complexity`, `regions` / `markup`, `safety_identifier` (sha256 of the user id) |
| Request path | inline in `_openai_images` | `ImageTools._openai_request` for every OpenAI images call (stills, GIFs, frame redraws) |
| SDK | `openai>=1.59.0` | `openai>=3.46.0` (the new parameters are keyword arguments there) |

Cost: same rates ($5 text in / $8 image in / $30 image out per 1M tokens), but 2.6 spends ~4x the output tokens
of 2.5 at `low` quality and ~3x at `medium` (OpenAI calculator, launch-week tests); `high` and above are equal.

## 2. Animated GIFs

The model draws all frames on one sheet within one image budget, and the API encodes the GIF. So the frames
are consistent, but small (512 px short edge: 1:1 → 512x512, 16:9 → 896x512) and few (2-16, default 9, 200 ms
each → a 1.8 s loop). Right for loops, stickers, reaction GIFs, logo/icon animations and simple character
motion; scenes, camera moves, realism and sound stay with the Veo video tools.

Tool choice: "an animation", "a GIF", "a sticker", "a loop" and "animate this picture" go to
`generate_animation`; the Veo tools say they are for an explicit request for a video only (before that,
`image_to_video_generator` also claimed to "animate" pictures and was picked first).

**`generate_animation`** — `prompt`, `images` (animate a picture: `images.edit` with `output_format=gif`),
engine `auto`/`best`/`fast` (`models.ANIMATION_BACKENDS`, its own model pair `GPT_ANIMATION_MODEL[_FAST]` so
stills can roll back without losing GIFs), `quality`, `aspect_ratio` (a photo keeps its own shape),
`frames`, `frame_duration_ms`, `complexity`, `caption`.

1. `frame_count` and `frame_duration_ms` are fixed knobs (a refusal is an error, explained: preview access,
   old SDK, or "animating a picture was refused" with the two alternatives).
2. The returned GIF is read back (frames, delays, loop) so the result reports what really came, and a
   numbered **frame sheet** `<name>_frames.png` is saved: Claude sees only the first frame of a GIF, the sheet
   is how the agent checks the motion (`view_image`).
3. Delivered with `ChatSender.send_animation` (plays and loops in the chat). Telegram stores animations as
   MP4, so `send_file_path_to_user_via_telegram` sends a `.gif` with `disable_content_type_detection` when the
   user wants the file itself.

**`edit_animation`** — everything that only changes how a GIF plays is local and free (`agents/animation.py`):
`frames` ("1-6", "1,3,5"), `reverse`, `boomerang`, `speed` / `frame_duration_ms`, `loop`, `output: "mp4"`; a
video from the workspace becomes a GIF first (ffmpeg: 10 s, 10 fps, ≤ 512 px), so a Veo clip of a photo can
end up a GIF too. `edit_frame` + `edit_prompt` is the one model call: that frame as PNG through `images.edit`
with a "keep everything else" instruction, scaled back and put in place. Order: edit_frame, frames, reverse,
boomerang, timing, loop. Always a new file; the original stays.

## 3. Markup-guided edits

The API wants a transparent PNG the size of the photo with the marks on it. Nobody in Telegram makes one, so
the bot does, in one of three ways:

1. **The agent marks it** (the usual case). The user writes "make the lamp on the left red"; the agent, which
   sees the photo, passes `regions` (boxes or circles as fractions of the image), and the bot draws them as red
   outlines on a transparent PNG.
2. **The user draws in Telegram.** Telegram's photo editor (the brush when attaching a photo) burns the drawing
   into the JPEG. With the clean original in the workspace too (sent twice, or earlier), `markup` = the drawn
   copy: the drawing is what differs from the original by more than `MARKUP_DIFF_THRESHOLD` per channel,
   cleaned of JPEG specks, and only those strokes go into the overlay.
3. **A transparent PNG already in the workspace** (one the agent made or downloaded) is used as the overlay
   as it is. One the user uploads in Telegram is not: uploads are converted to JPEG for the model
   (`media.prepare_downloaded_image_for_vision` flattens transparency on white), so it arrives opaque and
   is treated like a drawn copy (way 2).

A mark only points at an area. The prompt says what it means: change the marked area ("replace the circled
cup with a glass") or keep it while changing the rest ("keep the circled face, change the background"); the
tool description tells the agent to say which. The overlay is the size of the *prepared* first input after
EXIF orientation (the API requires it). A refused markup is dropped and the agent is told.

## 4. To verify with the real API (the parts the docs leave open)

| # | Check | Expected | If not |
|---|---|---|---|
| 1 | "make a GIF of a cat waving" | an animation in the chat, caption `gpt-image-2.6-sunburst · … · 9 frames · 200 ms · 512x512` | error says "GIF preview": ask OpenAI to enable it for the org / both models |
| 2 | same with "16:9, 12 frames, fast" | `gpt-image-2.6-flare`, 896x512, 12 frames | "used auto instead" in the log/result: the GIF size table differs, adjust `gif_pixel_size` |
| 3 | send a photo, "animate this photo" | an animation of that photo (**passed**: flare, 8 frames, 512x704) | "Animating an existing picture was refused": the edits endpoint does not take GIFs on this account; the agent offers the Veo route |
| 4 | "twice as fast", "boomerang", "only frames 1-5", "as mp4" | new animation / video, no API call (no cost line for it) | |
| 5 | "redraw frame 3 with …" | one `images.edit` call, the frame replaced | |
| 6 | a still image at default settings (2K) | delivered as before | "Size 2048x2048 is not accepted … used 1024x1024": 2.6 takes standard sizes only (works, but smaller) |
| 7 | quality `max` on 2.6 | no note | note "used 'high' instead": 2.6 has no xhigh/max |
| 8 | "a fox logo as SVG" | a `.svg` file | "cannot make SVG" / an error naming `svg`: SVG model selection or access missing |
| 9 | send a photo, "make the lamp red" (one lamp among others) | the agent passes `regions`, only that lamp changes | |
| 10 | the photo, then the same photo with a circle drawn on it, "remove what I circled" | the circled thing is gone | |
| 11 | a portrait, "keep the face exactly, change the background to a beach" | the agent circles the face, the prompt says keep | |
| 12 | `/usage` after the tests | GIF costs ≈ $0.02-0.06 each | |

Rollback: `GPT_IMAGE_MODEL=gpt-image-2.5-sunburst`, `GPT_IMAGE_MODEL_FAST=gpt-image-2.5-flare` in the PROD
variables (GIFs keep working through `GPT_ANIMATION_MODEL*`).
