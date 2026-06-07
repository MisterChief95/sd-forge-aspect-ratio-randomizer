# Aspect Ratio Randomizer

A Stable Diffusion WebUI **always-on extension** that enables random aspect ratio generation while maintaining consistent image area.

## Features

- Always-on accordion (like ControlNet/FreeU) — coexists with other scripts; no need to occupy the exclusive `Scripts` dropdown slot
- Supports common aspect ratios (21:9, 16:9, 3:2, 4:3, 1:1, etc.)
- Supports custom aspect ratios via Settings menu
- Automatically adjusts dimensions to the WebUI's Resolution Step (default 64px multiples) to maintain extension compatibility
- Maintains consistent image quality by preserving total pixel area
- Renders the **same seed at every aspect ratio** by default, so you can compare ratios directly (optional)
- Records the chosen ratio in each image's metadata (`Aspect ratio` in PNG info)
- Works with Wan-format image models (e.g. Anima, Qwen)

## Settings

Found under **Settings → Aspect Ratio Randomizer**:

- **Custom Aspect Ratios** — comma-separated `width:height` entries to add to the list (e.g. `3:4, 4:3`). Invalid entries are ignored with a console warning.
- **Log each batch's resolution to the console** — per-batch console line (default on); disable to keep the console quiet on large runs. The run summary always prints.

Computed dimensions are rounded to the WebUI's built-in **Resolution Step** (Settings → default 64) so they stay valid for the model and other extensions.

## Usage

1. Open the `Aspect Ratio Randomizer` accordion on the **Txt2Img** tab and tick its **enable** checkbox
2. Set your width to the desired base size.
    - Example: A width of `1024` will produce images within that resolution range. A ratio of 3:4 will produce an `896 x 1152` image.
3. Select your desired aspect ratios from the checkbox group
4. Set **batch count** (number of resolutions), **batch size** (max concurrent / VRAM), and **Variants per resolution** ([See How It Works](#how-it-works))
5. Run your generation as normal

## How It Works

The extension uses the standard Forge batch controls plus its own variants slider:

- **Batch count** = how many different resolutions to produce this run
- **Batch size** = max images generated concurrently (the GPU batch / VRAM control)
- **Variants per resolution** (extension slider) = how many images to render at each resolution; can exceed the batch-size cap of 8
- **Total images = batch count × variants per resolution**

The variants for each resolution are generated in GPU batches of *batch size*, so
you can request many variants without the VRAM cost of generating them all
simultaneously.

Example — batch count `3`, batch size `2`, variants per resolution `4`, with
`[4:3, 3:4, 16:9]` selected:

```
12 total images, generated in 6 GPU batches of 2:

  res 0 (4:3)  -> 1792x1344   batch[v0,v1]  batch[v2,v3]
  res 1 (3:4)  -> 1344x1792   batch[v0,v1]  batch[v2,v3]
  res 2 (16:9) -> 2048x1152   batch[v0,v1]  batch[v2,v3]
```

With batch size `1`, the same run produces 12 images one at a time.

Ratios are shuffled and assigned without repeating until the whole selection has
been used. If batch count exceeds the number of selected ratios, the selection is
reshuffled and reused.

> **Note:** if *batch size* does not evenly divide *variants per resolution*, the
> variant count rounds **up** to the next multiple (e.g. batch size 3 with variants
> 4 yields 6 variants per resolution). The actual total is logged.

### Use the same seed across all resolutions

Enabled by default. Each comparison group's first resolution captures its variant
seeds **and prompts**, then reuses them for the other resolutions in that group,
so each variant renders with an identical seed-and-prompt pairing at every aspect
ratio — a true side-by-side comparison. When batch count exceeds the number of
selected ratios, the next pass starts a fresh seed/prompt group instead of cycling
back to the first one. This also pins **wildcard / Dynamic Prompts** picks (e.g.
`{joe|jeff|john}` or `__characters__`) so the same word is chosen at every
resolution instead of being re-rolled per batch (each batch is a different slice
of the generated prompt list, so without this the seed would match but the chosen
word generally wouldn't). Uncheck this to let both seeds and prompts advance
normally so each resolution gets fresh variants.

At the end of generation, the returned gallery is sorted by seed, then by
resolution (`width x height`) so same-seed / same-prompt comparisons are grouped
together.

### Limit batch count to number of selected resolutions

When this checkbox is enabled, the batch count is reduced so each selected ratio
is used at most once (a message is logged to the console). With 3 ratios selected
and a batch count of 10, the run is capped to 3 batches.

### Implementation notes

The extension hooks into the normal pipeline as an always-on extension instead of
taking over the generation loop:

1. **`before_process`** builds the per-batch ratio plan (and optionally clamps the
   batch count). It runs before the prompt/seed arrays are built, so the clamp
   resizes everything correctly.
2. **`before_process_batch`** runs once per batch iteration. It sets that batch's
   width/height (square-root-of-area scaling keeps the pixel count roughly
   constant, rounded to the built-in Resolution Step), rebuilds the noise RNG so the
   latent matches the new resolution, and recomputes the Hires-fix target.

## Limitations

- **Txt2Img only.** It is disabled on img2img (changing the resolution would distort the init image).
- Resolution is constant within a single GPU batch (it varies across batches), so a given GPU batch can't mix resolutions.
- The Hires-fix `Use old hires fix width/height` option is not specifically supported and may behave unexpectedly with multiple resolutions.
- Relies on the pipeline's internal noise-RNG construction; if Forge changes how `p.rng` is built upstream, this extension may need updating.
- Conflicts with selectable `Scripts` (e.g. Prompt Matrix, X/Y/Z Plot) that take over the generation loop — set the Scripts dropdown to `None` when using this extension.
- If another always-on script changes the batch count *after* this extension has planned it (notably **Dynamic Prompts'** *Combinatorial generation*, which can rewrite `p.n_iter` to fit its own combination count), the resolution plan is cycled to cover the extra batches — logged once to the console — rather than crashing. The extra batches reuse earlier ratios rather than getting a freshly shuffled plan.
