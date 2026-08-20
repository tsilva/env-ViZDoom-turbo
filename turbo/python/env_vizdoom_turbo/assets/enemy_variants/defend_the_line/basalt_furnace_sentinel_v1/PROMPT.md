# Image-generation record

Mode: built-in OpenAI image generation with a local reference image.

Reference: `original/freedoom-imp-attack-reference.png`, extracted from the BSD-3-Clause Freedoom `freedoom2.wad`. The reference was supplied only for pose order, front-facing viewpoint, and the three-column layout.

Prompt:

> Use case: stylized-concept
>
> Asset type: game character attack-animation spritesheet proof for a Doom-compatible research environment
>
> Input images: Image 1 is a BSD-licensed Freedoom reference used ONLY for the three attack poses, front-facing viewpoint, sequence order, and three-column layout; do not copy its creature identity, anatomy, face, horns, colors, or distinctive design.
>
> Primary request: create a wholly original alternate enemy called a basalt furnace sentinel, shown in exactly three separate front-facing attack-animation frames: frame 1 compact ready stance with arms lowered; frame 2 both heavy arms raised outward while charging; frame 3 arms extended upward/outward releasing an implied fireball attack. Keep one consistent character identity and proportions across all frames.
>
> Subject: stocky humanoid construct made from asymmetrical cracked black basalt plates, compact rectangular furnace torso with a small bright amber core, blunt helmet-like head with a single narrow pale-cyan visor, two heavy segmented arms, short stable lower body. Clearly different silhouette and anatomy from the reference monster. No horns, no tentacles, no exposed ribs, no insect body, no red crest.
>
> Style/medium: deliberately low-resolution 1990s FPS pixel-art sprite, hard pixel clusters, limited high-contrast palette, no antialiasing, no painterly rendering, no 3D render, suitable for later downscaling to roughly 64 pixels tall.
>
> Composition/framing: one horizontal three-cell spritesheet; three equal-width cells; one centered full-body sprite per cell; consistent baseline, scale, head size, torso size, lighting, and camera; generous separation; no overlap; all extremities stay inside each cell.
>
> Scene/backdrop: perfectly flat solid #ff00ff chroma-key background for local background removal. The background must be one uniform color with no shadows, gradients, texture, reflections, checkerboard, floor plane, or lighting variation.
>
> Color palette: charcoal black, dark slate, pale cyan visor, small amber/orange furnace glow; do not use magenta or pink anywhere in the character.
>
> Constraints: original design only; no copyrighted game character resemblance; no logos; no trademarks; no text; no labels; no watermark; no cast shadow; no contact shadow; crisp closed silhouettes; preserve the exact three-pose progression and same character across frames.
>
> Avoid: horns, tentacles, skeletal ribcage, brown organic skin, the reference character's face or silhouette, extra limbs, cropped limbs, inconsistent costume, perspective changes, background artifacts.

Post-processing: chroma-key removal, largest connected-component isolation per cell, nearest-neighbor downscaling, source-PLAYPAL quantization, and binary-alpha normalization. Decorative generated fire particles were removed so the gameplay projectile remains a separate `DoomImpBall` actor.
