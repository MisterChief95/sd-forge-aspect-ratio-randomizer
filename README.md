# Aspect Ratio Randomizer

A Stable Diffusion WebUI **always-on extension** that enables random aspect ratio generation while maintaining consistent image area.

## Features

- Supports common aspect ratios (21:9, 16:9, 3:2, 4:3, 1:1, etc.)
- Supports custom aspect ratios via Settings menu
- Automatically adjusts dimensions to the WebUI's Resolution Step (default 64px multiples) to maintain extension compatibility
- Maintains consistent image quality by preserving total pixel area
- Renders exact seed/prompt variant counts at multiple aspect ratios, keeping batch size as a GPU parallelism limit
- Records the chosen ratio in each image's metadata (`Aspect ratio` in PNG info)

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
4. Set normal Forge **batch count** and **batch size**, then set **Variants per ratio** and **Ratios per batch** in the extension ([See How It Works](#how-it-works))
5. Run your generation as normal

## How It Works

The extension treats Forge's batch controls plus its own sliders as:

- **Batch count** = number of comparison groups
- **Variants per ratio** (extension slider) = number of seed/prompt variants per group
- **Ratios per batch** (extension slider) = number of aspect ratios to render for each group
- **Batch size** = maximum GPU parallelism / VRAM control
- **Total images = batch count × variants per ratio × ratios per batch**

Each actual GPU batch uses a single resolution. The extension chooses the largest
exact internal chunk size that divides **Variants per ratio** and does not exceed
the requested **Batch size**.

Example — batch count `3`, variants per ratio `12`, batch size `8`, ratios per
batch `3`, with
`[4:3, 3:4, 16:9]` selected:

```
108 total images.

Batch size 8 is treated as a maximum, so the exact internal chunk is 6:

  group 0 seeds 100-105 -> ratio 4:3
  group 0 seeds 106-111 -> ratio 4:3
  group 0 seeds 100-105 -> ratio 3:4
  group 0 seeds 106-111 -> ratio 3:4
  group 0 seeds 100-105 -> ratio 16:9
  group 0 seeds 106-111 -> ratio 16:9

  ...repeated for groups 1 and 2
```

With batch size `1`, the same setup produces the exact same 108 images one at a
time. If the requested batch size divides **Variants per ratio**, it is used
directly; otherwise a smaller exact divisor is used and logged to the console.

### Ratio Selection

The **Ratio selection** dropdown controls how each comparison group chooses its
ratio set:

- **Random, no repeats** — randomly samples selected ratios without duplicates for each batch
- **Selected order** — uses the selected ratios in UI order
- **Sequential rotating** — walks through the selected ratios across batches
- **Random with replacement** — allows the same ratio to appear more than once in a batch

For all modes except **Random with replacement**, the effective ratio count is
capped to the number of selected ratios.

### Dynamic Prompts

The extension is loaded after **sd-dynamic-prompts** via `metadata.ini`. Dynamic
Prompts first creates the seed/prompt variants for each comparison group, then
Aspect Ratio Randomizer duplicates those finished variants across the chosen
ratios. This pins wildcard / Dynamic Prompts choices (e.g. `{joe|jeff|john}` or
`__characters__`) across the ratios in each comparison group.

At the end of generation, the returned gallery can be sorted by seed, then by
resolution (`width x height`) so same-seed / same-prompt comparisons are grouped
together.

### Implementation notes

The extension hooks into the normal pipeline as an always-on extension instead of
taking over the generation loop:

1. **`before_process`** captures the selected ratios and original base width.
2. **`process`** runs after prompt/seed arrays are built. It expands each
   comparison group's variants once per chosen ratio, using an exact internal
   chunk size at or below the requested `batch_size`.
3. **`before_process_batch`** runs once per expanded batch iteration. It sets that batch's
   width/height (square-root-of-area scaling keeps the pixel count roughly
   constant, rounded to the built-in Resolution Step), rebuilds the noise RNG so the
   latent matches the new resolution, and recomputes the Hires-fix target.

## Limitations

- **Txt2Img only.** It is disabled on img2img (changing the resolution would distort the init image).
- Resolution is constant within a single GPU batch (it varies across batches), so a given GPU batch can't mix resolutions.
- Relies on the pipeline's internal noise-RNG construction; if Forge changes how `p.rng` is built upstream, this extension may need updating.
- Conflicts with selectable `Scripts` (e.g. Prompt Matrix, X/Y/Z Plot) that take over the generation loop — set the Scripts dropdown to `None` when using this extension.
- If another always-on script changes the batch count *after* this extension has expanded the plan, the resolution plan is cycled to cover the extra batches — logged once to the console — rather than crashing.
