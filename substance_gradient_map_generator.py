import numpy as np
from PIL import Image

# ---- Material gradient keyframes ----
# Each entry is a list of (R, G, B) keyframes from deep/base (left) to surface/tip (right).
# Two single-pixel gradient variations are generated per material.
MATERIALS = [
    # 1.  Stratified Sandstone: Dark red base → orange → pale tan → white crust
    [(90, 30, 20), (180, 90, 40), (210, 180, 130), (240, 235, 220)],
    # 2.  Old Growth Tree Bark: Pitch black fissures → rich brown → pale grey/green moss
    [(10, 5, 5), (80, 45, 20), (160, 140, 110), (170, 175, 150)],
    # 3.  Deep Pile Shag Rug: Dark shadow base → rich mid-tone → bright highlighted tips
    [(30, 15, 50), (120, 60, 140), (200, 150, 210), (240, 220, 245)],
    # 4.  Weathered Corrugated Iron: Deep orange/brown rust → faded silver/grey
    [(90, 40, 10), (180, 90, 30), (160, 155, 150), (200, 200, 195)],
    # 5.  Excavated Soil Profile: Dark brown organic topsoil → sandy loam → red/yellow clay
    [(40, 25, 10), (100, 70, 35), (190, 170, 120), (180, 130, 70)],
    # 6.  Tall Meadow Grass: Brown/yellow thatch → pale green stems → vibrant green tips
    [(100, 75, 30), (160, 150, 80), (140, 180, 100), (60, 160, 40)],
    # 7.  Glacial Ice Wall: Dense blue core → fracturing light blue → opaque white
    [(20, 50, 120), (60, 130, 200), (140, 200, 230), (240, 248, 255)],
    # 8.  Oxidized Copper Sheet: Dark brown/black recesses → bright verdigris green/teal
    [(40, 25, 15), (80, 60, 40), (50, 160, 130), (80, 200, 170)],
    # 9.  Exposed Aggregate Concrete: Matte grey cement base → polished river stone peaks
    [(100, 100, 100), (155, 150, 145), (180, 160, 140), (200, 190, 175)],
    # 10. Volcanic Lava Crust: Incandescent yellow/orange fractures → dull red → pitch black crust
    [(250, 220, 50), (220, 80, 20), (120, 20, 10), (15, 10, 10)],
    # 11. Heavy Velvet Drapes: Near-black → rich base colour → highly reflective sheen
    [(10, 5, 30), (60, 20, 100), (100, 40, 160), (200, 150, 230)],
    # 12. Mossy Retaining Wall: Dark wet porous stone → dense green moss → dry sun-bleached stone
    [(30, 35, 30), (50, 100, 50), (80, 140, 60), (200, 190, 160)],
    # 13. Rusted Steel Hull: Black/green marine growth → orange rust → faded peeling paint
    [(10, 30, 10), (160, 80, 20), (200, 120, 50), (180, 160, 130)],
    # 14. Weathered Brick: Dark mortar deep → red brick mid → white efflorescence surface
    [(60, 55, 50), (170, 70, 50), (200, 100, 80), (230, 225, 215)],
    # 15. Asphalt Road: Black tar matrix deep → grey/white aggregate surface
    [(10, 10, 10), (50, 50, 50), (120, 120, 115), (190, 185, 175)],
    # 16. Stucco Wall: Dark shadows in crevices → bright painted peaks
    [(80, 70, 60), (160, 145, 130), (210, 200, 185), (240, 235, 225)],
    # 17. Aged Wooden Shingles: Dark rot underneath → silver-grey sun-bleached top
    [(30, 20, 10), (80, 60, 40), (140, 135, 125), (180, 175, 170)],
    # 18. Terracotta Roof Tiles: Black algae in dips → baked orange on crests
    [(20, 25, 20), (80, 50, 35), (190, 100, 50), (210, 130, 70)],
    # 19. Rammed Earth Wall: Stratified bands of red, brown, and tan
    [(120, 50, 30), (140, 80, 50), (170, 120, 70), (200, 170, 110)],
    # 20. Cobblestone Path: Dark mud/moss in joints → polished grey stone peaks
    [(30, 35, 25), (80, 80, 70), (140, 140, 135), (180, 178, 172)],
    # 21. Rusty Chainlink Fence: Black core → orange rust → silver galvanized flakes
    [(15, 12, 10), (160, 70, 20), (200, 120, 50), (195, 195, 190)],
    # 22. Tarnished Brass Plaque: Black oxidation in engraved letters → polished gold surface
    [(15, 12, 8), (60, 50, 20), (160, 130, 40), (210, 185, 80)],
    # 23. Travertine Pavers: Dark dirt in porous holes → pale cream surface
    [(60, 50, 35), (160, 150, 130), (215, 208, 192), (235, 230, 215)],
    # 24. Poured Asphalt: Deep black binder → oily rainbow sheen surface
    [(8, 8, 8), (30, 30, 30), (60, 55, 65), (100, 110, 120)],
    # 25. Corroded Bronze Statue: Black recesses → bright green patina → polished bronze highlights
    [(15, 15, 10), (40, 90, 60), (70, 160, 100), (180, 140, 60)],
    # 26. Painted Barn Wood: Dark bare wood deep → peeling red paint → white sun-bleached edges
    [(30, 20, 10), (80, 60, 40), (170, 50, 40), (230, 220, 210)],
    # 27. Slate Roofing: Dark grey shadowed overlaps → blue/green/purple exposed faces
    [(30, 30, 35), (80, 90, 100), (110, 120, 130), (130, 140, 155)],
    # 28. Terrazzo Floor: Opaque cement base → translucent marble chip surface
    [(100, 100, 100), (160, 158, 155), (210, 200, 195), (230, 225, 220)],
    # 29. Peeling Plaster: Grey lath base → brown scratch coat → white flaking finish
    [(80, 75, 65), (130, 105, 80), (200, 195, 185), (240, 238, 232)],
    # 30. Cinder Block: Dark grey shadowed pores → light grey flat surface
    [(50, 50, 50), (110, 108, 105), (155, 153, 150), (180, 178, 175)],
    # 31. Geode Slice: Rough brown exterior → blue agate bands → purple amethyst crystal spikes
    [(100, 70, 40), (50, 100, 180), (120, 80, 200), (180, 130, 240)],
    # 32. Malachite Ore: Dark green deep bands → light green shallow bands
    [(20, 70, 30), (40, 120, 60), (80, 170, 90), (140, 210, 130)],
    # 33. Banded Iron Formation: Black magnetite layers → red chert layers
    [(15, 15, 20), (80, 20, 15), (160, 50, 30), (180, 80, 50)],
    # 34. Desert Sand Dune: Dark damp core → vibrant orange/red dry surface sand
    [(120, 90, 50), (200, 140, 60), (220, 160, 70), (230, 180, 100)],
    # 35. Riverbed Pebbles: Dark muddy base → varied brown/grey stones → bright algae-covered tops
    [(40, 35, 25), (110, 100, 85), (150, 148, 140), (100, 140, 80)],
    # 36. Stalactite: Dark mineral core → translucent white/yellow calcium carbonate rings
    [(60, 55, 50), (160, 150, 120), (220, 215, 190), (245, 240, 220)],
    # 37. Peat Bog Core: Black dense carbon → brown fibrous peat → green living sphagnum top
    [(15, 10, 5), (80, 50, 25), (120, 90, 50), (80, 140, 70)],
    # 38. Obsidian Shard: Dense opaque black core → translucent grey/brown sharp edges
    [(10, 8, 10), (30, 25, 30), (80, 70, 75), (150, 140, 130)],
    # 39. Crystalline Bismuth: Grey metal base → iridescent pink/blue/gold stepped crystal faces
    [(80, 80, 85), (180, 120, 150), (120, 160, 210), (210, 190, 80)],
    # 40. Petrified Wood: Dark brown mineralized core → red/yellow silica rings → grey bark exterior
    [(50, 30, 15), (150, 70, 40), (200, 160, 60), (160, 155, 145)],
    # 41. Mud Puddle Crust: Dark wet mud underneath → light brown curling dry flakes on top
    [(50, 40, 25), (120, 100, 70), (180, 160, 120), (215, 200, 170)],
    # 42. Pumice Stone: Dark shadowed deep vesicles → light grey/white frothy surface
    [(70, 68, 65), (140, 138, 135), (195, 193, 190), (225, 222, 218)],
    # 43. Coral Reef Rock: Dark grey dead core → bright pink/purple living coralline algae surface
    [(50, 50, 55), (100, 80, 90), (200, 100, 160), (220, 140, 200)],
    # 44. Layered Shale: Dark grey/black organic-rich layers → lighter grey silt layers
    [(20, 20, 22), (60, 62, 65), (110, 112, 115), (160, 162, 165)],
    # 45. Permafrost Profile: White ice lenses → dark frozen soil → active thawed brown layer top
    [(220, 235, 245), (80, 70, 60), (60, 55, 45), (120, 100, 70)],
    # 46. Desert Varnish Rock: Pale sandstone interior → dark brown/black manganese oxide surface coating
    [(210, 185, 140), (150, 110, 70), (60, 45, 30), (20, 15, 10)],
    # 47. Salt Flat Crust: Wet brown mud base → pinkish halite layer → bright white salt crust
    [(80, 65, 45), (160, 130, 110), (220, 190, 180), (250, 248, 245)],
    # 48. Mica Schist: Dark grey rock matrix → highly reflective silver/gold flaking layers
    [(50, 50, 55), (110, 108, 105), (185, 180, 160), (220, 215, 180)],
    # 49. Volcanic Tuff: Grey ash matrix → embedded black glass shards → white pumice fragments
    [(100, 98, 95), (30, 28, 28), (150, 148, 145), (220, 218, 215)],
    # 50. Limestone Cave Wall: Dark damp rock → bright white flowstone calcite drips
    [(40, 45, 50), (120, 115, 105), (200, 198, 190), (245, 243, 238)],
    # 51. Pine Cone: Dark brown resinous core → lighter brown woody scales → pale grey weathered tips
    [(40, 20, 5), (110, 70, 35), (160, 130, 90), (190, 185, 175)],
    # 52. Mushroom Gills: Dark brown spore-heavy deep gills → pale tan/white shallow edges
    [(60, 40, 25), (130, 105, 80), (200, 185, 160), (235, 228, 215)],
    # 53. Coral Fungus: White dense base → yellow/orange branching stems → bright red/pink tips
    [(245, 242, 238), (230, 200, 100), (220, 140, 60), (210, 60, 80)],
    # 54. Succulent Leaf: Watery translucent green core → opaque green flesh → pink/red sun-stressed tips
    [(180, 220, 170), (80, 170, 90), (60, 140, 70), (210, 80, 80)],
    # 55. Tortoise Shell: Dark brown/black scute margins → amber/yellow raised centers
    [(20, 15, 10), (80, 50, 20), (180, 140, 50), (210, 185, 80)],
    # 56. Seashell (Mother of Pearl): Dull grey/brown exterior → chalky white mid-layer → iridescent nacre
    [(130, 120, 100), (215, 210, 200), (230, 225, 220), (200, 210, 230)],
    # 57. Peacock Feather: Dark brown central rachis → iridescent green/blue barbs → bronze/gold tips
    [(50, 30, 15), (30, 140, 90), (20, 100, 180), (180, 150, 50)],
    # 58. Alligator Skin: Soft pale yellow belly/recesses → dark olive/black armored ridges
    [(200, 190, 150), (120, 130, 80), (50, 60, 30), (20, 20, 15)],
    # 59. Tree Fern Trunk: Dense black fibrous core → brown overlapping frond bases → green mossy exterior
    [(10, 8, 5), (80, 55, 30), (120, 100, 60), (80, 130, 70)],
    # 60. Ripe Peach: Pale yellow/white flesh near pit → vibrant orange/red skin → white fuzzy surface
    [(240, 230, 180), (230, 150, 80), (210, 80, 60), (240, 235, 230)],
    # 61. Artichoke: Pale yellow/green tender base → dark green tough leaves → purple thistle tips
    [(210, 215, 160), (80, 140, 70), (50, 100, 50), (160, 80, 160)],
    # 62. Leopard Print Fur: Pale tan undercoat → black rosette borders → dark brown rosette centers
    [(215, 195, 150), (15, 12, 10), (80, 50, 20), (100, 70, 30)],
    # 63. Moldy Bread: Pale white/tan bread core → blue/green fuzzy mycelium → black spore caps
    [(230, 220, 195), (80, 130, 100), (50, 150, 80), (10, 10, 10)],
    # 64. Fallen Autumn Leaf: Dark brown rotting veins → brittle yellow/orange lamina → pale curled edges
    [(50, 30, 10), (180, 140, 40), (210, 160, 60), (230, 220, 190)],
    # 65. Porcupine Quill: White base → broad black bands → sharp white/yellow tip
    [(240, 240, 235), (15, 12, 10), (235, 230, 200), (240, 238, 200)],
    # 66. Snake Scales: Pale skin between scales → deeply pigmented scale base → iridescent transparent keratin
    [(210, 200, 180), (60, 80, 50), (30, 60, 30), (180, 200, 160)],
    # 67. Beehive Honeycomb: Dark amber brood comb → translucent yellow wax walls → white wax cappings
    [(100, 55, 10), (200, 150, 40), (230, 200, 100), (240, 238, 225)],
    # 68. Coconut Shell: White fleshy interior → dense brown hard shell → light brown fibrous coir exterior
    [(240, 238, 230), (100, 65, 30), (80, 55, 25), (180, 150, 100)],
    # 69. Deer Antler: Porous dark red/brown marrow core → dense white bone → brown stained velvet exterior
    [(100, 40, 30), (210, 205, 195), (235, 230, 220), (130, 100, 65)],
    # 70. Strawberry: White/pink core flesh → bright red surface flesh → yellow embedded seeds
    [(240, 220, 220), (220, 60, 60), (200, 40, 50), (230, 210, 80)],
    # 71. Sunflower Center: Pale green receptacle → dark brown/black tubular flowers → yellow pollen tips
    [(160, 190, 100), (60, 40, 15), (30, 20, 10), (220, 190, 30)],
    # 72. Lichen Crust: Dark grey attachment points → pale green/grey thallus → bright orange/yellow fruiting bodies
    [(50, 50, 50), (140, 150, 130), (180, 185, 165), (210, 150, 30)],
    # 73. Insect Exoskeleton: Soft pale internal tissue → dark brown/black chitin → iridescent structural colour
    [(200, 195, 180), (40, 25, 10), (15, 12, 8), (80, 160, 120)],
    # 74. Elephant Skin: Pale grey/pink in deep skin folds → dark grey/brown tough outer ridges
    [(190, 175, 170), (100, 95, 90), (70, 65, 60), (90, 85, 80)],
    # 75. Kiwifruit: White central columella → black seeds → green flesh → brown fuzzy skin
    [(245, 243, 238), (10, 10, 10), (80, 160, 70), (140, 105, 60)],
    # 76. Corduroy Fabric: Dark shadowed valleys → rich coloured ridges → pale worn highlights
    [(30, 40, 60), (60, 90, 140), (80, 120, 180), (180, 200, 230)],
    # 77. Faux Fur: Dark synthetic base web → mid-tone undercoat fibres → frosted white guard hair tips
    [(30, 30, 30), (110, 90, 80), (170, 150, 140), (235, 232, 228)],
    # 78. Sequined Cloth: Dark thread base → opaque sequin back → highly reflective coloured sequin face
    [(15, 15, 15), (60, 20, 80), (100, 30, 130), (230, 50, 200)],
    # 79. Distressed Denim: Dark indigo deep weave → mid-blue surface weave → white frayed/broken threads
    [(20, 30, 80), (50, 80, 150), (100, 140, 200), (240, 238, 235)],
    # 80. Bouclé Yarn: Tight dark core thread → loose looping brightly coloured outer fibres
    [(40, 35, 30), (160, 80, 60), (200, 120, 80), (240, 180, 140)],
    # 81. Woven Burlap: Dark empty spaces between threads → rough brown jute fibres → pale frayed hairs
    [(30, 25, 15), (130, 100, 60), (175, 145, 95), (220, 210, 185)],
    # 82. Chenille Upholstery: Dark woven backing → dense colourful fuzzy pile → reflective crushed highlights
    [(20, 30, 40), (60, 100, 140), (80, 130, 180), (200, 220, 240)],
    # 83. Terry Cloth Towel: Flat coloured base weave → darker damp loops → bright dry loop tips
    [(150, 60, 60), (100, 30, 30), (200, 100, 100), (240, 200, 200)],
    # 84. Macramé Wall Hanging: Shadowed knot intersections → natural cotton rope → frayed brushed fringes
    [(60, 55, 45), (180, 165, 140), (210, 200, 180), (235, 230, 215)],
    # 85. Flannel Fleece: Dark dyed base → bright brushed surface → white pilling bobbles
    [(40, 20, 40), (150, 60, 150), (200, 100, 200), (240, 235, 240)],
    # 86. Embossed Leather: Dark dyed/stained recessed patterns → mid-tone smooth surface → pale worn areas
    [(40, 25, 15), (100, 65, 40), (150, 105, 70), (200, 175, 145)],
    # 87. Quilted Satin: Dark shadows in stitched seams → smooth coloured fabric → high-gloss reflective highlights
    [(20, 40, 80), (50, 100, 180), (80, 140, 220), (230, 240, 255)],
    # 88. Tufted Carpet: Dark mesh backing → dense coloured yarn → pale crushed/worn tips
    [(20, 20, 20), (80, 50, 80), (140, 80, 140), (200, 180, 200)],
    # 89. Smocked Fabric: Deep shadowed folds → dense gathered fabric colour → highlighted thread stitches
    [(20, 50, 40), (50, 140, 110), (80, 190, 155), (230, 245, 238)],
    # 90. Brocade Silk: Matte dark background weave → raised metallic gold/silver thread patterns
    [(30, 10, 50), (60, 20, 100), (200, 170, 50), (220, 210, 180)],
    # 91. Seared Steak: Red rare core → grey/brown cooked band → dark brown/black charred crust
    [(180, 30, 30), (130, 80, 60), (80, 55, 35), (25, 15, 10)],
    # 92. Baked Bread Crust: White fluffy crumb → light brown transition → dark brown crispy crust
    [(245, 242, 235), (210, 185, 140), (170, 120, 60), (110, 70, 25)],
    # 93. Melted Cheese: Pale yellow soft core → oily orange separated fat layer → dark brown blistered bubbles
    [(240, 235, 180), (220, 170, 80), (200, 130, 50), (80, 50, 20)],
    # 94. Frosted Cake: Dark dense sponge core → soft pale icing layer → brightly coloured hard sprinkles
    [(100, 60, 40), (240, 235, 225), (250, 245, 240), (220, 80, 120)],
    # 95. Corrugated Cardboard: Flat brown liner board → shadowed air gaps → pale brown fluted paper
    [(160, 120, 70), (60, 45, 25), (180, 150, 100), (210, 190, 155)],
    # 96. Acoustic Foam: Dark black/grey sound-absorbing deep wedges → lighter grey/dusty peaks
    [(15, 15, 15), (60, 60, 60), (110, 108, 105), (165, 162, 158)],
    # 97. Welding Slag: Bright silver molten steel core → dark dull grey cooling slag → rust-coloured oxide surface
    [(220, 225, 230), (80, 80, 80), (60, 55, 50), (180, 100, 50)],
    # 98. Fiberglass Insulation: Dark shadowed dense core → bright pink/yellow loose spun glass → silver foil backing
    [(50, 30, 30), (230, 140, 100), (240, 220, 100), (200, 200, 200)],
    # 99. Abrasive Sandpaper: Heavy brown paper backing → dark resin binder → highly reflective sharp grit peaks
    [(160, 120, 70), (60, 45, 30), (80, 70, 60), (210, 205, 195)],
    # 100. Charred Wood: Unburnt pale wood core → black cracked charcoal layer → silver ash surface
    [(200, 170, 120), (20, 15, 10), (10, 10, 10), (180, 178, 175)],
    # 101. Glazed Pottery: Porous terracotta base → opaque coloured glaze → highly reflective clear glass topcoat
    [(180, 100, 60), (80, 120, 180), (100, 150, 220), (230, 240, 250)],
    # 102. Bubble Wrap: Flat clear plastic backing → shadowed air pockets → highly reflective domed plastic surface
    [(200, 215, 230), (120, 140, 160), (160, 180, 200), (230, 240, 250)],
    # 103. Spongy Bone: Dense yellow marrow core → white calcified trabecular web → solid white cortical bone shell
    [(220, 200, 130), (230, 225, 210), (240, 238, 232), (248, 246, 242)],
    # 104. Solder Joint: Dark copper pad base → dull grey tarnished flux → highly reflective silver solder bead
    [(150, 80, 30), (100, 98, 90), (130, 128, 120), (220, 220, 225)],
    # 105. Ash Pile: Black charcoal coals at base → dark grey dense ash → light white fluffy ash on top
    [(15, 10, 8), (60, 58, 55), (140, 138, 134), (220, 218, 214)],
]

VARIATIONS = 2
MIN_KEYS = 8  # minimum colour keyframes per gradient

_rng = np.random.default_rng(42)


def build_keyframes(base_colors):
    """Expand base colour stops to at least MIN_KEYS with perturbed intermediates
    and non-linearly spaced positions along the gradient."""
    colors = np.array(base_colors, dtype=np.float64)
    n = len(colors)
    n_gaps = n - 1
    # How many extra intermediates to insert between each pair of anchors
    n_insert = max(0, int(np.ceil((MIN_KEYS - n) / max(n_gaps, 1))))

    expanded = [colors[0]]
    for i in range(n_gaps):
        c0, c1 = colors[i], colors[i + 1]
        for j in range(n_insert):
            t = (j + 1) / (n_insert + 1)
            mid = c0 + t * (c1 - c0)
            # Per-channel perturbation so intermediates diverge from the straight line
            perturb = _rng.uniform(-40, 40, 3)
            expanded.append(np.clip(mid + perturb, 0, 255))
        expanded.append(c1)

    expanded = np.array(expanded)
    num = len(expanded)

    # Non-linear positions: beta(0.55, 0.55) concentrates stops near 0 and 1,
    # leaving the mid-section sparse for larger colour jumps across the gradient.
    if num > 2:
        inner = np.sort(_rng.beta(0.55, 0.55, num - 2))
        inner = inner * 0.88 + 0.06        # keep away from exact endpoints
        key_x = np.concatenate([[0.0], inner, [1.0]])
    else:
        key_x = np.array([0.0, 1.0])

    return expanded, key_x


num_materials = len(MATERIALS)
height = num_materials * VARIATIONS
width = height  # square image

atlas = np.zeros((height, width, 3), dtype=np.uint8)
x_indices = np.linspace(0, 1, width)

# Harmonic frequencies used for the overlay noise — prime-ish values give
# an irregular, non-repeating texture over the gradient width.
HARMONICS = [5, 11, 23, 47]

for i, keyframes in enumerate(MATERIALS):
    colors, key_x = build_keyframes(keyframes)

    # Build base gradient from expanded, non-linearly spaced keyframes
    base_gradient = np.zeros((width, 3))
    for c in range(3):
        base_gradient[:, c] = np.interp(x_indices, key_x, colors[:, c])

    for v in range(VARIATIONS):
        row_idx = i * VARIATIONS + v

        # Sum multiple sine harmonics with random per-harmonic phase and
        # decreasing amplitude so higher frequencies add fine detail only.
        noise = np.zeros(width)
        base_amp = 18.0 + v * 10.0
        for k, freq in enumerate(HARMONICS):
            phase = _rng.uniform(0, 2 * np.pi)
            amp = base_amp / (1.4 ** k)        # each harmonic ~40 % quieter
            noise += np.sin(x_indices * freq + phase) * amp

        # Slight per-channel tint on the noise keeps variations visually distinct
        noise_3ch = np.column_stack([noise * 0.80, noise * 1.00, noise * 0.88])
        atlas[row_idx, :, :] = np.clip(base_gradient + noise_3ch, 0, 255).astype(np.uint8)

Image.fromarray(atlas).save("material_height_gradient_atlas.png")
print(f"Saved {width}x{height} atlas  ({num_materials} materials x {VARIATIONS} variations, {MIN_KEYS}+ keys each)")