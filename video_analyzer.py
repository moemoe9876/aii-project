#!/usr/bin/env python3
"""
Video Analysis Agent powered by Gemini 2.5 Pro
Generates detailed technical Markdown reports analyzing video cinematography, composition, and atmosphere.
"""

import sys
import os
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

try:
    from google import genai
    from google.genai import types
except ImportError:
    print("Error: google-genai is not installed.")
    print("Please install it using: pip install google-genai")
    sys.exit(1)

load_dotenv()


class VideoAnalyzer:
    def __init__(self, api_key=None):
        """
        Initialize the video analyzer with Gemini 2.5 Pro.
        
        Args:
            api_key: Gemini API key (defaults to GEMINI_API_KEY env var)
        """
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found. Set it in .env file or pass it directly.")
        
        self.client = genai.Client(api_key=self.api_key)
        self.model = "gemini-2.5-pro"
        # Fixed temperature set in code
        self.temperature = 0.1
        
    def _create_analysis_prompt(self):
        """Create the ultra-comprehensive system instruction for Gemini (frame-state capture for perfect reconstruction)."""
        return """
You are an ULTRA-COMPREHENSIVE video reconstruction analysis system. Your output will be used to recreate this video using AI generation models (text-to-image + image-to-video). Your mission is to capture EVERY visual variable with reconstruction-ready precision.

CORE PRINCIPLES:
1. **Frame State Snapshots** - Describe exact visual states at multiple timestamps (like photographs)
2. **Prompt-Ready Language** - Use natural language that converts directly to AI generation prompts
3. **Quantitative Precision** - Include measurements, percentages, spatial relationships, exact positions
4. **Visual DNA** - Identify what makes each frame uniquely reconstructible
5. **Continuity Tracking** - Maintain consistency markers across all sequences
6. **Model-Aware Descriptions** - Format for text-to-image and image-to-video model consumption

Generate a detailed Markdown report that follows this EXACT structure and ordering:

# Video Analysis Report

## Timeline Breakdown (Chronological)

**CRITICAL: YOU MUST ANALYZE EVERY SHOT IN THE VIDEO FROM START TO FINISH. DO NOT STOP AFTER A FEW SHOTS.**

Provide a shot-by-shot breakdown covering 100% of video duration, strictly in time order. Identify natural shot boundaries (cuts, transitions, significant camera/scene changes) and create a separate analysis section for EACH shot. 

For EVERY shot from 00:00 to the end of the video, include:

### SHOT METADATA
- Timestamp: [MM:SS]–[MM:SS]
- Duration: [N.N seconds]
- Shot Title: [Concise descriptive label]
- Shot Type: [Establishing/Action/Close-up/Reaction/etc.]
- Reconstruction Difficulty: [Low/Medium/High/Extreme] + reason

### FRAME STATE SNAPSHOTS
For EACH shot, provide 2-4 frame snapshots depending on duration:
- Short (<3s): Start + End
- Medium (3-8s): Start + Mid + End  
- Long (>8s): Start + Quarter + Mid + Three-Quarter + End

For EACH snapshot timestamp, provide a complete visual state description:

**[MM:SS] Frame State**

#### VISUAL HIERARCHY (Prompt-Ready Description)
Write a 100-150 word natural language description of this exact frame as if describing it to a text-to-image AI. Structure: Subject → Action/Pose → Environment → Lighting → Atmosphere → Style. Use vivid, concrete language.

Example format: "A young woman with long black hair wearing a red cotton jacket and blue denim jeans stands in the center of a rain-soaked city street at night, her body turned 45 degrees to the left, looking over her shoulder toward the camera with a neutral expression. Behind her, neon signs in Chinese characters glow with pink and cyan light, reflecting in the wet asphalt that covers 60% of the lower frame. Soft diffused light from a streetlamp camera-right creates a rim light on her hair. The atmosphere is moody and cinematic, reminiscent of Wong Kar-wai films shot on Cinestill 800T with visible halation around bright lights."

#### PRIMARY ELEMENTS (Must-Have for Recognition)
List the 3-5 essential elements that define this frame:
- Element 1: [detailed description with spatial position - "occupies center 40% of frame, positioned at rule-of-thirds intersection"]
- Element 2: [include colors, sizes, textures - "bright red neon sign reading '[TEXT]', 15% of frame width, upper left quadrant"]
- ...

#### SECONDARY ELEMENTS (Atmospheric/Context)
List 3-7 important but non-critical elements:
- Element: [description with spatial relationship to primary elements]
- ...

#### BACKGROUND/TERTIARY ELEMENTS
Brief inventory of background elements that provide depth and context.

#### QUANTITATIVE SPECIFICATIONS
- **Aspect Ratio**: [16:9, 4:3, 2.39:1, etc.]
- **Subject Screen Position**: [e.g., "center-left, occupying 35% frame height, positioned at x=30% y=55%"]
- **Subject Count**: [N people, N vehicles, N objects of interest]
- **Depth Layers**: [Foreground (items), Midground (items), Background (items)]
- **Screen Coverage**: [Subject: X%, Environment: Y%, Sky/ceiling: Z%]
- **Lighting Ratio**: [Key to fill approximate ratio, e.g., 4:1 high contrast]
- **Color Distribution**: [Dominant: color X%, Secondary: color Y%, Accent: color Z%]

#### CAMERA GEOMETRY & POSITIONING (Critical for Accurate Reconstruction)

**CAMERA POSITION (3D Spatial Analysis)**
- **Camera Height (Absolute)**: [Estimated height above ground level in meters, e.g., "2.3m above ground level"]
  - Estimation Method: [How determined - e.g., "based on horizon line at subject's chest height (subject ~1.7m tall), camera ~0.6m above"]
  - Reference Points: [What was used for measurement - "door frame visible at 2.1m", "car hood at 1.2m", "average human eye level 1.6m"]
- **Camera Height (Relative to Subject)**: [Position relative to primary subject - "0.4m above subject's eye level", "at subject's waist height", "2m below subject"]
- **Horizontal Position**: [Left/Center/Right relative to scene, with offset estimate if applicable - "1.5m camera-left of subject's centerline"]
- **Distance to Primary Subject**: 
  - Estimated Distance: [Precise estimate in meters, e.g., "4.2 meters"]
  - Distance Category: [Close <2m / Medium 2-5m / Far 5-10m / Very Far >10m]
  - Estimation Method: [How calculated - see methodology below]
    * Subject height reference (assume average adult 1.65-1.75m, adjust for visible age/build)
    * Known object sizes (door 2m, car length 4-5m, motorcycle 2m, bicycle 1.7m)
    * Perspective convergence rate of parallel lines
    * Relative size of facial features (head width ~15cm at known distances)
    * Depth-of-field relationship (sharp zone extent at given aperture)

**CAMERA ANGLE (Precise Decomposition)**
- **Pitch (Vertical Angle)**: [Up/Down tilt in degrees]
  - Measurement: [e.g., "+25 degrees" for looking down, "-15 degrees" for looking up, "0 degrees" for level]
  - Determination Method: [How calculated - "horizon line position relative to frame center", "subject's body proportions compression/expansion", "vertical line convergence"]
  - Descriptive: [High angle / Eye-level / Low angle / Bird's eye / Worm's eye]
- **Yaw (Horizontal Angle)**: [Left/Right rotation relative to subject's front]
  - Measurement: [e.g., "45 degrees camera-right of subject's facing direction", "0 degrees head-on", "90 degrees profile", "135 degrees three-quarter back"]
  - Subject Relationship: [What part of subject is visible - "full frontal view", "left profile", "three-quarter view showing left 70%"]
- **Roll (Horizon Tilt)**: [Dutch angle amount]
  - Measurement: [e.g., "0 degrees (level)", "+8 degrees clockwise tilt", "-12 degrees counter-clockwise"]
  - Visual Effect: [What this creates - "stable", "slightly unsettling", "dynamic energy"]

**LENS CHARACTERISTICS & FIELD OF VIEW**
- **Focal Length Estimate**: [Specific estimate in full-frame equivalent, e.g., "50mm", "24mm", "85mm"]
  - Category: [Ultra-wide <20mm / Wide 20-35mm / Normal 35-55mm / Short Tele 55-85mm / Telephoto >85mm]
  - Confidence: [High/Medium/Low] based on evidence strength
- **Field of View (FOV)**:
  - Horizontal FOV: [Degrees, e.g., "46 degrees (normal)", "84 degrees (wide)", "28 degrees (tele)"]
  - Vertical FOV: [Degrees, account for aspect ratio]
  - Estimation Method: [What indicates this - "subject fills X% of frame at Y distance", "visible width at known distance"]
- **Perspective Characteristics**:
  - Compression: [None/Slight/Moderate/Strong - how depth is compressed]
  - Evidence: ["Background appears X% closer than it is", "foreground elements show minimal size exaggeration", "parallel lines converge at rate consistent with Xmm lens"]
  - Distortion: [Barrel (wide lens) / None / Pincushion (tele)]
  - Edge Behavior: [How straight lines behave at frame edges - "straight", "slightly bowed", "significant curvature"]

**SPATIAL REFERENCE FRAMEWORK**
- **Coordinate System Origin**: [Where 0,0,0 is defined - typically "primary subject's ground contact point" or "frame geometric center at subject depth"]
- **Ground Plane Orientation**: [Describe the ground/floor relative to camera]
  - Visible: [Yes/No - can you see where subjects stand?]
  - Angle: [Parallel to sensor / Tilted toward camera / Tilted away from camera] + degrees if measurable
  - Reference: ["Tile grid shows vanishing point at frame center 60% up", "horizon line at 45% frame height"]
- **Depth Axis (Z-axis) Direction**: [Into the scene - describe viewing angle]
  - Parallel Lines: [Railroad track effect - where do parallel lines converge? "Upper third center", "Not visible - perpendicular view"]
  - Depth Compression: [Strong (telephoto feel) / Natural / Exaggerated (wide angle feel)]

**SHOT SIZE & FRAMING**
- **Shot Size Classification**: [Extreme Wide / Wide / Medium Wide / Medium / Medium Close-Up / Close-Up / Extreme Close-Up]
- **Subject Framing**: [Describe what body parts are included]
  - If Person: [Full body / From knees up / From waist up / Shoulders and head / Head only / Facial feature detail]
  - Frame Tightness: [How much space around subject - "loose framing with 30% headroom", "tight crop at edges"]
- **Screen Space Occupied**: [Subject occupies X% of frame width, Y% of frame height]

**CAMERA MOVEMENT GEOMETRY** (if applicable - otherwise state "STATIC")
- **Movement Type**: [None/Pan/Tilt/Dolly/Track/Crane/Orbit/Zoom/Handheld/Combined]
- **If Pan**: [Direction: Left/Right] + [Amount: degrees] + [Duration: seconds] + [Speed: degrees/second]
- **If Tilt**: [Direction: Up/Down] + [Amount: degrees] + [Duration: seconds]
- **If Dolly**: [Direction: Forward/Backward] + [Distance: meters] + [Duration: seconds] + [Speed: m/s]
- **If Track**: [Direction: Left/Right/Diagonal] + [Distance: meters] + [Maintains subject distance: Yes/No]
- **If Orbit**: [Around subject - Clockwise/Counter-clockwise] + [Degrees traveled] + [Radius: meters]
- **If Zoom**: [In/Out] + [Focal length change: start mm → end mm] + [Duration: seconds]
- **Movement Quality**: [Smooth/Jerky/Organic] + [Speed: Slow/Medium/Fast] + [Acceleration: Constant/Easing in-out/Sudden]
- **Stabilization Quality**: [Locked (tripod) / Smooth (gimbal/dolly) / Slight shake (handheld) / Unstabilized (raw handheld)]

**MOTION PARALLAX & DEPTH CUES**
- **Foreground-to-Background Motion Differential**: [When camera/subject moves, how much do different depth layers move relative to each other?]
  - Strong Parallax: [Wide lens, close to subject, visible depth separation]
  - Moderate Parallax: [Normal lens, medium distance]
  - Minimal Parallax: [Telephoto compression, subjects and background move similarly]
- **Motion Vector Analysis**: [If camera moves - which way does background appear to move?]
  - Camera moves right → Background appears to move left
  - Track speed relative to subject speed for following shots

**CAMERA POSITION SUMMARY (Prompt-Ready)**
[Write a 2-3 sentence natural language summary of camera positioning that can be directly used in image generation prompts]

Example: "Camera positioned 4.2 meters from subject at a height of 2.3m (0.7m above subject's eye level), creating a subtle high angle of 18 degrees looking down. Shot with a 50mm equivalent lens producing a natural field of view of 46 degrees with no distortion. The camera is static on a tripod, positioned slightly camera-right of the subject's centerline, capturing a three-quarter view showing 65% of the subject's front."

#### LIGHTING DESIGN (Reconstruction-Ready)
- **Key Light**: [Position e.g., "camera-right 45°, elevated 30°", quality, color temp, intensity 1-10]
- **Fill Light**: [Position, quality, color temp, intensity relative to key]
- **Rim/Back Light**: [Position, quality, creates separation? Hair light?]
- **Practical Lights**: [In-scene sources: neon signs, lamps, windows with positions]
- **Ambient**: [Overall level, color cast]
- **Shadows**: [Hard/soft, direction, density, length relative to subject height]
- **Light Condition Tags** (from reference list): [Top 2-3] with confidence %

#### COLOR PALETTE (Prompt-Ready)
- **Primary Colors**: [Name + hex approximation + where present + coverage %]
  Example: "Deep cyan blue (#0A4D68) in neon signs and reflections, 25% of frame"
- **Secondary Colors**: [Same format, 15-20% coverage]
- **Accent Colors**: [Small pops, <10% coverage]
- **Color Temperature**: [Warm/Neutral/Cool + Kelvin estimate]
- **Color Grading Style**: [Teal-orange, desaturated, high contrast, faded, etc.]
- **Saturation Level**: [Muted/Natural/Vibrant/Hyper-saturated] + 0-100% scale
- **Contrast**: [Low/Medium/High/Extreme] + note where contrast is strongest
- **Film Type Tags** (from reference list): [Top 2-3] with confidence %

### Scene Physics & Grounding
- Surface/Terrain: concrete, asphalt, tile, grass, dirt; condition (dry/wet/puddled); slope/grade (up/down/level); steps/ramps; curb presence/height; crosswalk markings; lane/edge lines; obstacles/barriers.
- Location & Usage: sidewalk vs roadway vs interior; lane count/orientation; side‑of‑road convention (left‑hand/right‑hand driving) if visible; bike/bus lane presence; signage/road markings (quote text if legible).
- Ego vs World Motion: camera stationary/handheld/vehicle‑mounted; camera motion vector (direction/speed band). Decompose observed motion into subject motion vs camera motion.
- Environmental Dynamics: wind (hair/clothing/foliage), precipitation, water ripples/reflections; traffic density; flow rate estimates (vehicles/min, pedestrians/min) over the shot window.
- Contact Evidence: footprints/splashes/wheel spray; skids; friction cues.

#### DEPTH & FOCUS (Reconstruction-Critical)
- **Depth of Field**: [Shallow/Medium/Deep] + approximate f-stop if inferable
- **Focus Plane**: [Which elements are sharp, which are soft]
- **Bokeh Character**: [Circular/hexagonal/smooth/busy/harsh] if visible
- **Focus Pulls**: [Any rack focus during this frame state]
- **Z-Depth Staging**: [How subjects are arranged in depth, important for i2v parallax]

#### COMPOSITION & FRAMING
- **Rule of Thirds**: [How primary subject aligns with intersection points]
- **Leading Lines**: [Describe lines that guide eye through frame]
- **Framing Devices**: [Natural frames like doorways, windows, foreground objects]
- **Balance**: [Symmetric/Asymmetric, visual weight distribution]
- **Negative Space**: [Where and how much, what it emphasizes]

#### TEXTURES & MATERIALS (Surface Detail)
Describe visible surface qualities that AI must replicate:
- Fabrics: [cotton, silk, denim, leather - with texture descriptors: smooth, rough, worn, shiny]
- Surfaces: [concrete, glass, metal, wood - with conditions: polished, rusty, cracked, wet]
- Skin/Hair: [texture quality, highlights, ambient occlusion in folds]
- Weather Effects: [rain, fog, dust, particles, visible in air or on surfaces]

#### ATMOSPHERE & MOOD
- **Emotional Tone**: [Describe the feeling this frame evokes]
- **Cinematic References**: [Similar to films/directors if applicable]
- **Era/Period Markers**: [Visual elements that date the scene]

#### AUDIO CONTEXT (for motion matching)
- **Sound Design**: [Music/ambient/SFX that informs motion pacing]
- **Audio-Visual Sync**: [How sound relates to visible action]

---

### SUBJECTS & MOTION TRACKING (Per Shot)
After providing frame state snapshots, describe how subjects move BETWEEN the snapshots.

For each distinct moving subject:

**Subject ID: [S1, S2, V1, V2, etc.]**
- **Class**: [Person/Vehicle/Animal/Object - be specific: "young adult male", "yellow taxi", "German Shepherd", "red bicycle"]
- **Appearance Details** (for continuity):
  - If person: Age range, gender presentation, ethnicity markers, height estimate, build
  - Clothing: Detailed head-to-toe (colors, materials, fit, condition, brand markers if visible)
  - Hair: Style, color, length, movement characteristics
  - Accessories: Bags, jewelry, glasses, hats - with colors and positions
  - Distinguishing features: Tattoos, scars, unique items
- **Screen Position Journey**: [Start position → Path → End position with % coordinates]
- **Motion Vector**: [Direction + magnitude: "moves from x=20%,y=60% to x=70%,y=45%"]
- **Speed Analysis**: 
  - Qualitative: [Very slow/Slow/Medium/Fast/Very fast]
  - Quantitative: [Estimate with units + basis: "~5 km/h based on walking cadence"]
  - Consistency: [Steady/Accelerating/Decelerating/Variable]
- **Motion Style** (for people):
  - Gait: [Walking style: casual stride, hurried pace, shuffle, limp, run]
  - Cadence: [Steps per second if countable]
  - Body mechanics: [Arm swing, hip movement, head bob, posture during motion]
  - Energy: [Relaxed/Tense/Energetic/Exhausted]
- **Interactions**: [With other subjects, environment, camera - note timing]
- **Occlusions**: [When/where/by what - crucial for i2v generation]

**Motion Table** (when 3+ moving subjects):
| ID | Class | Screen Path | Speed | Style | Key Interactions |
|----|-------|-------------|-------|-------|-----------------|

**Camera Motion Analysis** (Enhanced 3D Analysis):
- **Movement Type**: [Static/Pan/Tilt/Dolly/Track/Crane/Orbit/Zoom/Handheld/Combined]
- **If STATIC**: [Confirm: "Camera is completely stationary on tripod, no movement"]
- **If MOVING**:
  - **Translation (Linear Movement)**:
    * X-axis (Horizontal): [Left/Right] + [Distance: meters] + [Speed: m/s]
    * Y-axis (Vertical): [Up/Down] + [Distance: meters] + [Speed: m/s]
    * Z-axis (Depth): [Forward/Backward toward/away from subject] + [Distance: meters] + [Speed: m/s]
  - **Rotation (Angular Movement)**:
    * Pan (Yaw): [Left/Right] + [Degrees: total angle swept] + [Speed: degrees/second]
    * Tilt (Pitch): [Up/Down] + [Degrees: total angle] + [Speed: degrees/second]
    * Roll: [Clockwise/Counter-clockwise rotation] + [Degrees if intentional]
  - **Combined Movement Description**: [E.g., "Dolly forward 2m while panning right 15° and tilting down 5°"]
  - **Duration**: [Total movement duration in seconds]
  - **Movement Quality**: 
    * Speed Profile: [Constant / Accelerating / Decelerating / Ease in-out]
    * Smoothness: [Perfectly smooth (locked) / Slight organic motion / Visible shake / Jerky]
  - **Subject Tracking**: [Does camera follow subject? How?]
    * Lock Type: [Perfect lock (subject stays in same screen position) / Loose follow / Leading / Lagging]
    * Distance Change: [Maintains distance / Approaches / Recedes]
- **Stabilization Analysis**: [Tripod locked / Gimbal smooth / Dolly/track / Handheld shake / Vehicle-mounted]
- **Motivation**: [Follows subject action / Reveals environment / Creates tension / Unmotivated artistic choice]
- **Ego-Motion Separation**: [Critical - How to distinguish camera motion from subject motion]
  - Background Movement: [How background moves reveals camera motion]
  - Subject Relative Position: [Does subject stay centered or drift across frame?]
  - Parallax Indicators: [Do foreground/background layers move at different rates?]
- **Reference Frame**: [Is camera fixed to ground, vehicle, person, crane, drone?]

### Object Identification & Usage (Very Important)
For each subject interacting with an object/vehicle/tool, identify the object as specifically as possible using only visible cues. Report a best match and alternatives with confidence + evidence.
- Object Taxonomy: specific type (e.g., bicycle → road bike | mountain | city/commuter | BMX | fixie | folding | e‑bike; motorcycle → scooter | sportbike | cruiser | standard | touring; car → taxi | sedan | hatchback | SUV)
- Visual Evidence: wheel/tire size & tread, frame geometry, handlebar shape, suspension presence, fairings, seating posture, cargo racks, lighting, mirrors; logos/text if legible
- Confidence: percent + 1–2 reasons; if uncertain, list top 2 and state “Ambiguous”
- Usage Posture: how the subject uses it (e.g., seated vs standing on pedals, torso lean angle, one‑hand vs two‑hand grip, foot on brake/clutch, throttle hand, helmet/gear)
- Constraints: NEVER invent hidden features; base conclusions on visible geometry, posture, and markings only

### DETAILED POSE ANALYSIS (Per Person, Per Frame Snapshot)
For each person in each frame snapshot, provide detailed pose information:

**Person [ID] - [MM:SS]**
- **Facing Direction**: [Toward camera / Away / Profile left/right / 3/4 view] + degrees
- **Head**: Orientation (yaw/pitch/roll in degrees or descriptive), gaze direction, expression
- **Torso**: Lean angle (forward/back + degrees), twist (left/right), slouch/upright
- **Shoulders**: Level/tilted, rotation, tension
- **Arms** (detailed):
  - Left arm: Upper arm position, elbow angle (~degrees), forearm position, hand position, what touching
  - Right arm: Same detail
  - Overall: Crossed, relaxed, gesturing, specific position
- **Hands** (if visible): Open/closed, holding what, finger positions, gestures
- **Hips**: Rotation, weight shift, alignment with shoulders
- **Legs** (detailed):
  - Left leg: Hip angle, knee angle, foot placement, weight bearing
  - Right leg: Same detail  
  - Stance: Width, orientation, stability
- **Feet**: Position, ground contact (heel/toe/full), orientation (parallel/splayed/pigeon)
- **Center of Mass**: Location (forward/centered/back), balance state
- **Overall Posture Keywords**: [E.g., "relaxed standing", "dynamic mid-stride", "seated leaning forward"]
- **Body Language**: [What the pose communicates emotionally]

For **vehicles/objects** being used:
- **Contact Points**: [Where person touches object - both hands, feet, body]
- **Positioning**: [How person is arranged relative to object - seated height, reach distance]
- **Usage State**: [Active use / passive / preparing / finishing]

---

**Repeat ALL above sections for EACH frame snapshot in this shot**

---

**IMPORTANT: After completing one shot analysis, IMMEDIATELY begin the next shot analysis. Continue this process until you reach the end of the video. DO NOT SKIP ANY SHOTS.**

---

[Continue with next shot following the exact same structure: SHOT METADATA → FRAME STATE SNAPSHOTS → SUBJECTS & MOTION TRACKING → DETAILED POSE ANALYSIS]

[Repeat for ALL subsequent shots until video ends]

---

## Global Summary & Continuity Registry

### RECONSTRUCTION OVERVIEW
- **Total Shots**: [N]
- **Average Shot Length**: [N seconds]
- **Reconstruction Complexity**: [Low/Medium/High/Extreme] + explanation
- **Primary Challenge**: [What makes this video hardest to reconstruct]
- **Recommended Pipeline**: [Suggest specific t2i and i2v models based on content]

### LOCATION & SETTING
- **Primary Location**: [Specific as possible]
- **Location Type**: [Urban/Suburban/Rural/Interior/Natural]
- **Geographic Markers**: [Architecture style, signage language, cultural indicators]
- **Time Period**: [Modern/Historical era + evidence]
- **Season**: [Based on foliage, weather, clothing]
- **Time of Day Distribution**: [X shots day, Y shots night, Z shots twilight]

### CONTINUITY REGISTRY (Critical for Multi-Sequence Consistency)

#### Character Registry
For each recurring person, create a consistent appearance profile:

**Character [ID]: [Role descriptor, e.g., "Main male protagonist", "Couple - Driver"]**
- **Physical Description**: Age, ethnicity, gender, height, build, distinctive features
- **Face**: Shape, skin tone, facial hair, notable features
- **Hair**: Style, color, length, texture, how it moves
- **Clothing (Consistent Items)**: 
  - Top: [Detailed description with colors, materials, fit, patterns]
  - Bottom: [Same detail]
  - Shoes: [Style, color, condition]
  - Outerwear: [If present]
  - Accessories: [Persistent items across shots]
- **Appears in Shots**: [List shot timestamps]
- **Clothing Changes**: [Note any outfit changes between shots]
- **Distinguishing Mannerisms**: [Characteristic movements, expressions]
- **AI Generation Keywords**: [Condensed prompt-ready description for consistency]

**Example**: "Young adult East Asian male, early 20s, slim build, short black hair, wearing white tank top, open beige short-sleeve shirt, dark pants. Neutral expression, relaxed posture."

#### Vehicle/Object Registry
For each recurring vehicle or significant object:

**[Object ID]: [Type, e.g., "Honda motorcycle", "Yellow taxi"]**
- **Detailed Description**: Make, model, color, distinctive markings, condition
- **Appears in Shots**: [List timestamps]
- **Prompt Keywords**: [For t2i consistency]

#### Environment Registry
- **Recurring Locations**: [If video returns to same places, note their characteristics]
- **Persistent Background Elements**: [Buildings, signs, furniture that appear multiple times]

### VISUAL STYLE CONSISTENCY

#### Color Palette (Whole Video)
- **Dominant Color Scheme**: [Describe overarching palette]
- **Shot-by-Shot Palette Shifts**: [Note intentional color transitions]
- **Color Grading Philosophy**: [Overall approach: naturalistic, stylized, teal-orange, etc.]
- **Consistent Color Motifs**: [Colors that repeat with significance]

#### Lighting Style (Whole Video)
- **Overall Lighting Approach**: [Naturalistic/Dramatic/Flat/High-contrast]
- **Recurring Light Conditions**: [From reference list, what appears most]
- **Lighting Continuity**: [How lighting changes or stays consistent across shots]
- **Notable Lighting Techniques**: [Special approaches used throughout]

#### Film/Camera Style (Whole Video)
- **Dominant Film Look**: [Primary film stock emulation from reference list]
- **Grain Structure**: [Consistent grain level and character]
- **Lens Character**: [Consistent lens aberrations, vignetting, distortion]
- **Aspect Ratio**: [Maintained throughout]
- **Frame Rate Feel**: [Cinematic 24fps / smooth 30fps / etc.]

#### Cinematography Patterns
- **Recurring Shot Types**: [What shots appear multiple times]
- **Camera Movement Style**: [Overall approach: static/dynamic/mixed]
- **Editing Rhythm**: [Fast cuts/long takes/mixed]
- **Compositional Motifs**: [Recurring framing choices]
- **Directorial Influences**: [Name specific filmmakers if style matches]

### MOTION & PACING ANALYSIS
- **Overall Tempo**: [Slow/Medium/Fast/Variable]
- **Subject Speed Distribution**: [% very slow, % slow, % medium, % fast]
- **Camera Motion Usage**: [% static, % moving]
- **Energy Curve**: [How energy/intensity changes through video]

### COMPREHENSIVE OBJECT INVENTORY
Count all distinct instances across entire video:
- **People**: [N total, N unique individuals]
- **Vehicles**: [N cars, N motorcycles, N bicycles, etc.]
- **Animals**: [If any]
- **Significant Objects**: [Recurring or important objects with counts]

### ATMOSPHERIC ELEMENTS
- **Weather Conditions**: [Per shot or overall]
- **Environmental Effects**: [Rain, fog, dust, smoke, etc.]
- **Ambient Motion**: [Wind, water, traffic, crowds]
- **Mood Progression**: [How atmosphere evolves]

## Camera Geometry Estimate
- Camera height (relative to subject horizon), horizon tilt (°), and azimuth
- Field of View category: ultra‑wide (<20mm), wide (20–35mm), normal (35–55mm), short‑tele (55–85mm), tele (>85mm) with evidence (perspective compression, parallax, depth cues)
- Working distance to primary subject (qualitative: close/medium/far)

## AI Generation Strategy & Recommendations

### TEXT-TO-IMAGE Model Selection
- **Recommended Primary Model**: [FLUX.1 dev / Midjourney v6 / SD XL / etc.] + reason
- **Alternative Models**: [For specific shot types or challenges]
- **Model-Specific Settings**:
  - Aspect Ratio: [As determined from analysis]
  - Style Keywords: [Key terms for this video's aesthetic]
  - Negative Prompts: [What to avoid to maintain style]
  - CFG/Guidance: [Recommended range]

### IMAGE-TO-VIDEO Model Selection
- **Recommended Primary Model**: [Kling 2.5 Pro / Runway Gen-3 / Pika 1.5 / etc.] + reason
- **Motion Complexity**: [Which shots are easy/hard for i2v]
- **Special Challenges**: [Specific scenes that need attention]
- **Frame Strategy Recommendation**: [Which shots need First+Last vs Anchor]

### PROMPT ENGINEERING NOTES
- **Critical Keywords**: [Essential terms that must appear in every prompt for consistency]
- **Style Anchors**: [Phrases that maintain the video's look]
- **Common Pitfalls**: [What might go wrong in generation, how to avoid]
- **Quality Modifiers**: [Terms like "highly detailed", "8k", "cinematic" - use if appropriate]

### RECONSTRUCTION ROADMAP
Difficulty assessment per shot:
- **Easy Shots**: [List timestamps - static, simple, well-lit]
- **Medium Shots**: [List timestamps - moderate motion or complexity]
- **Hard Shots**: [List timestamps - fast motion, complex composition, lighting challenges]
- **Extreme Shots**: [List timestamps - may require multiple attempts or special techniques]

## Technical Specifications Summary

### Classification Tags (Whole Video)
- **Primary Light Conditions**: [Top 2-3 from reference list] with confidence % and evidence
- **Primary Film Types**: [Top 2-3 from reference list] with confidence % and evidence
- **Overall Visual Style**: [Descriptive summary]

### Post-Production Analysis
- **Color Grading**: [Specific looks applied - teal-orange, bleach bypass, faded, etc.]
- **Grain/Noise**: [Amount, character, consistency]
- **Vignetting**: [Present/absent, intensity]
- **Lens Effects**: [Flares, aberrations, distortion]
- **Sharpening**: [Over-sharpened, natural, soft]
- **Contrast Curves**: [S-curve, flat, crushed blacks, blown highlights]
- **LUT Estimate**: [If recognizable preset]

### Resolution & Quality Indicators
- **Apparent Resolution**: [Based on detail level]
- **Compression Artifacts**: [Present/minimal/none]
- **Motion Blur**: [Natural/added/excessive]
- **Frame Rate**: [24fps/30fps/60fps or other]

## CINEMATOGRAPHY QUALITY PROFILE (Critical for Matching Video Look)

This section captures the technical "DNA" of how the video was shot - the camera, lens, and film characteristics that define its visual quality. These attributes MUST be maintained across all generated sequences for authentic recreation.

### Camera/Sensor Characteristics
- **Capture Medium**: [Film (35mm/16mm/Super 8) / Digital (Full Frame/APS-C/Micro 4/3) / Video (Prosumer/Cinema)]
- **Sensor/Film Signature**: [Key visual indicators]
  - Resolution feel: [Crisp/Soft/Grainy based on visible detail retention]
  - Dynamic range: [How highlights and shadows are rendered - film-like rolloff vs digital clipping]
  - Latitude: [How much detail retained in bright/dark areas]
  - Color bit depth: [Rich gradations vs posterization/banding]
- **Digital vs Film Markers**:
  - If Film: Organic grain structure, halation around lights, gentle highlight rolloff, characteristic color response
  - If Digital: Clean shadows, sharp transitions, specific sensor artifacts, modern color science
- **Format Indicators**: [Evidence of capture format - aspect ratio native to format, crop patterns, edge characteristics]

### Lens Characteristics Profile
- **Focal Length Behavior**: [How perspective/compression appears across shots]
  - Wide shots: [Distortion amount, edge falloff, field curvature]
  - Normal shots: [Rendering style, central vs edge sharpness]
  - Telephoto shots: [Compression amount, bokeh quality, subject isolation]
- **Optical Quality Signature**:
  - Sharpness: [Clinical/Modern/Vintage/Soft - where peak sharpness, how it falls off toward edges]
  - Contrast: [Micro-contrast character, local vs global]
  - Resolution: [Line pair rendering, fine detail handling]
- **Lens Artifacts & Character**:
  - Vignetting: [Natural/Heavy/Corrected - specific pattern and intensity by focal length]
  - Distortion: [Barrel/Pincushion/Minimal - geometric warping patterns]
  - Chromatic Aberration: [Purple fringing/color separation at edges - where it appears, how prominent]
  - Flare Behavior: [How lens handles bright lights - star patterns, veiling flare, ghost images, specific colors]
  - Focus Breathing: [If zooms or focus pulls show size changes]
- **Bokeh Character** (very important):
  - Shape: [Circular/Hexagonal/Octagonal/Cat's eye - from aperture blade count and design]
  - Quality: [Smooth/Busy/Swirly/Harsh - how out-of-focus areas render]
  - Highlight behavior: [Specular highlights - round/defined edges/soap bubble effect]
  - Background rendering: [How textures blur - creamy/nervous/painterly]
- **Age/Era Indicators**: [Modern clinical, vintage warm, specific lens generation characteristics]

### Film Stock / Color Science Profile
- **Grain Structure** (critical for film look):
  - Density: [Fine/Medium/Heavy - visible at what magnification]
  - Size: [Micro-grain/Standard/Coarse - actual particle size appearance]
  - Pattern: [Even/Clumpy/Organic - how grain distributes across frame]
  - Movement: [Static/Dancing - does grain pattern change frame to frame]
  - Color: [Monochrome/Chromatic - does grain have color component]
  - Shadow vs Highlight grain: [Where grain is most/least visible]
  - Specific Film Stock Match: [Which film stocks from reference list show this exact grain signature]
- **Color Response Curves**:
  - Skin tones: [Warm/Neutral/Cool - specific hue shifts, magenta/yellow/green tendencies]
  - Primary colors: [How reds/blues/greens are rendered - saturation, hue accuracy, clipping behavior]
  - Secondary colors: [Cyan/magenta/yellow - specific tonal shifts]
  - Color separation: [How well colors remain distinct vs muddying]
  - Saturation falloff: [How colors desaturate in shadows/highlights]
- **Contrast Characteristics**:
  - Global contrast: [Overall range - flat/moderate/high/extreme]
  - Toe (shadow rolloff): [Abrupt/Gentle - how shadows transition to black]
  - Shoulder (highlight rolloff): [Hard clip/Soft rolloff - how highlights transition to white]
  - Curve shape: [Linear/S-curve/Lifted blacks/Crushed shadows - specific tonal mapping]
  - Per-channel contrast: [Do R/G/B channels have different contrast curves]
- **Specific Color Phenomena**:
  - Halation: [Glow around bright lights - present/absent, color, intensity]
  - Crossprocessing effects: [Any unusual color shifts suggesting non-standard development]
  - Color casts: [Consistent color temperature shifts - warm/cool bias in shadows vs highlights]
  - Film base color: [If visible in rebates/borders - indicates stock type]

### Exposure & Dynamic Range Profile
- **Exposure Philosophy**: [Overexposed-soft/Normal/Underexposed-moody - intentional exposure strategy]
- **Highlight Handling**:
  - Clipping point: [Where/how highlights blow out - hard digital clip vs film rolloff]
  - Retention: [Detail preserved in bright areas - specular vs diffuse highlights]
  - Recovery: [If overexposed, how much information retained]
- **Shadow Handling**:
  - Crush point: [Where shadows go to black - lifted/normal/crushed]
  - Noise floor: [Grain/noise in deepest shadows]
  - Detail retention: [Can you see into dark areas]
- **Mid-tone Rendering**: [Where exposure is "set" - skin tones, gray card equivalent]
- **Latitude Evidence**: [How much over/underexposure the footage shows it could handle]

### Motion & Temporal Characteristics
- **Frame Rate Feel**: [How motion renders]
  - Cadence: [24fps cinematic judder / 30fps video smooth / 60fps hyper-smooth / other]
  - Motion blur: [Amount and quality - natural/minimal/exaggerated]
  - Shutter angle equivalent: [180°/90°/360° - affects motion blur amount]
  - Strobe/stutter: [Any deliberate or camera-limitation motion artifacts]
- **Motion Blur Character**:
  - Length: [Short/Medium/Long trails on moving subjects]
  - Quality: [Clean/Smeared/Directional - how blur renders]
  - Consistency: [Same across frame or varies]
- **Temporal Resolution**: [How crisp or blurred fast motion appears]
- **Frame-to-Frame**: [Smoothness of motion, any telecine artifacts, judder patterns]

### Image Processing & Post-Production Fingerprint
- **Sharpening Signature**:
  - Amount: [None/Subtle/Moderate/Heavy]
  - Radius: [Fine detail enhancement vs broad enhancement]
  - Artifacts: [Halos/ringing visible around edges]
  - Where applied: [Overall vs selective]
- **Noise Reduction Evidence**:
  - Applied: [Yes/No/Selectively]
  - Artifacts: [Smoothed textures, loss of fine detail, waxy skin]
- **Stabilization**: [Post-stabilization artifacts - warping, edge crops, rolling shutter fix]
- **Upscaling Indicators**: [If video was upscaled - specific algorithm artifacts, sharpening patterns]

### QUALITY DNA SUMMARY (Use This Directly in Prompts)

**Core Technical Keywords** (include in EVERY image generation prompt):
[Generate a concise 50-100 word description that captures the essential technical look]

**CRITICAL ERA-SPECIFIC REQUIREMENT:**
If the video is determined to be from the 1970s, 1980s, 1990s, or early 2000s (before 2005), you MUST include the phrase **"grainy film"** (in bold) in the Core Technical Keywords. This phrase is essential for AI models to accurately replicate the analog film aesthetic of that era.

Example format:
"Shot on [specific camera/film stock] with [lens characteristics], exhibiting [grain description], [color science traits], [specific bokeh character], [lens artifacts], [contrast characteristics], [era-appropriate technical limitations], [notable optical phenomena]. [Specific sharpness quality]. [Motion blur character]. [Dynamic range handling style]. [IF 1970s-2000s: **grainy film**]"

Example output (1990s video):
"Shot on 35mm Kodak Vision3 500T with vintage Canon FD prime lenses, exhibiting organic medium-grain structure that's more visible in shadows, warm color science with slight magenta push in skin tones and teal-cyan shift in shadows, smooth circular bokeh with gentle swirl at edges, subtle vignetting and natural lens flare with warm amber tones, gentle S-curve contrast with lifted blacks and soft shoulder in highlights characteristic of film negative scan. Moderate sharpness with slight softness at frame edges, natural motion blur at 180° shutter, rich dynamic range with detail retention in both highlights and shadows typical of 1990s cinema. **grainy film**"

**Quality Anchor Negative Prompts** (what to AVOID):
[List 10-15 specific qualities that would break the look]

Example: "Avoid: modern digital sharpness, clinical lens rendering, video camera look, heavy digital noise reduction, crushed blacks, blown highlights, hexagonal bokeh from modern lenses, oversaturated colors, HDR tone mapping, 60fps smoothness, heavy vignetting, chromatic aberration corrections, perfect edge-to-edge sharpness, digital color science, phone camera aesthetic"

**Model-Specific Quality Settings**:
- **For FLUX/Stable Diffusion**: 
  - Key prompt additions: [specific technical keywords that work well]
  - CFG Scale recommendation: [lower for film look, higher for sharpness]
  - Sampler recommendations: [which samplers preserve grain/texture best]
- **For Midjourney**:
  - Style parameters: [--style raw, --chaos values, --stylize values]
  - Reference image strategy: [using film grain references]
- **For Image-to-Video**:
  - Motion settings: [to match frame rate feel and motion blur]
  - Temporal consistency: [how to maintain grain structure across frames]

### Cinematographic Consistency Checklist

Before generating ANY image, verify:
- [ ] Core technical keywords included in prompt
- [ ] Film stock/sensor characteristics specified
- [ ] Lens behavior and artifacts mentioned
- [ ] Grain structure described (if applicable)
- [ ] Color science traits included
- [ ] Bokeh character specified (if shallow DOF)
- [ ] Contrast curve style mentioned
- [ ] Negative prompts added to avoid wrong look
- [ ] Era-appropriate technical limitations noted
- [ ] Motion characteristics match source (for i2v)

This profile ensures EVERY generated frame matches the source video's technical quality, not just its content.

## Critical Reconstruction Notes
- **Must-Have Elements**: [List 5-10 absolutely critical visual elements for recognition]
- **Acceptable Variations**: [What can differ from original without breaking fidelity]
- **Failure Points**: [Elements that, if wrong, would make reconstruction obviously fake]
- **Continuity Anchors**: [Key elements to maintain across all sequences]
- **Testing Checklist**: [How to verify each generated shot matches analysis]

STRICT REQUIREMENTS FOR ULTRA-COMPREHENSIVE ANALYSIS

1. **PROMPT-READY LANGUAGE**: Write all descriptions as if directly instructing a text-to-image AI. Use concrete, visual language.
   - GOOD: "A woman with shoulder-length black hair wearing a red knit sweater stands in soft window light"
   - BAD: "The subject is present in the scene with appropriate attire"

2. **QUANTITATIVE PRECISION**: Include measurements, percentages, spatial relationships wherever possible.
   - Frame positions as % coordinates (x=30%, y=60%)
   - Screen coverage percentages (subject occupies 25% of frame)
   - Color distribution percentages
   - Approximate angles for camera and body positions

3. **FRAME STATE SNAPSHOTS**: Treat each snapshot timestamp as a complete photograph. Describe EVERYTHING visible at that exact moment.

4. **CONTINUITY OBSESSION**: Build the Character Registry and Object Registry meticulously. These enable cross-sequence consistency.

5. **SPATIAL PRECISION**: Always specify where elements are in frame using:
   - Rule of thirds grid (upper-left intersection, lower-right third, etc.)
   - Percentage coordinates
   - Depth layers (foreground/midground/background)
   - Relative positioning to other elements

6. **COLOR SPECIFICITY**: Name colors precisely with approximate hex codes when feasible.
   - GOOD: "Deep cyan blue (#0A4D68)"
   - ACCEPTABLE: "Dark blue-green in the cyan family"
   - BAD: "Blue"

7. **LIGHTING RECONSTRUCTION**: Describe lighting as if setting up a photo shoot:
   - Position (camera-right 45°, elevated 30°)
   - Quality (soft/hard)
   - Color temperature (warm 3200K / cool 5600K)
   - Intensity (1-10 scale relative to key)

8. **TEXTURE DETAIL**: AI models need texture information. Describe surface qualities: smooth/rough, matte/glossy, worn/new, wet/dry.

9. **NO SPECULATION**: Only describe what is directly visible. If something is unclear, state "ambiguous" or "not visible" rather than guessing.

10. **TIMESTAMPS**: Use MM:SS format with zero padding (00:05, 01:23, etc.)

11. **LEGIBLE TEXT**: Quote any readable text exactly. If illegible, note its presence but state "illegible".

12. **MOTION SEPARATION**: Clearly distinguish camera motion from subject motion. State estimation method.

13. **CAMERA ANALYSIS PRECISION** (CRITICAL - NEW REQUIREMENT): 
    - **Measure, Don't Estimate Vaguely**: Use visual reference points to calculate precise angles and distances
    - **Height Calculation**: Use subject height as scale (average adult 1.65-1.75m), measure relative to horizon line, eye level, or known objects (doors 2m, vehicles, furniture)
    - **Angle Decomposition**: Break down to Pitch (up/down in degrees), Yaw (left/right rotation), Roll (horizon tilt). Use horizon line position, body proportion compression, and line convergence to calculate
    - **Distance Triangulation**: Use multiple methods: subject size percentage, facial feature size, known object sizes, perspective convergence, depth-of-field spread
    - **Focal Length Evidence**: List specific visual cues - perspective compression (telephoto), distortion (wide), field of view coverage, subject-to-background relationship
    - **Motion Vectors in 3D**: Always decompose camera motion into X (horizontal), Y (vertical), Z (depth) components with distance/speed, plus rotational components (pan/tilt/roll) with angles/speed
    - **Show Your Work**: Always state the methodology and reference points used for each measurement
    - **Camera Position Summary**: End each camera section with a 2-3 sentence natural language summary suitable for AI prompts

14. **RECONSTRUCTION DIFFICULTY**: For each shot, assess how hard it will be to reconstruct and why.

14. **MODEL AWARENESS**: Consider limitations of current AI models:
    - Complex multi-subject interactions are hard
    - Fine text rendering is challenging
    - Consistent faces across shots need character registry
    - Fast motion blur is difficult for i2v models

15. **HIERARCHICAL DETAIL**: Use the Primary/Secondary/Tertiary element structure. This helps prompt engineering focus on essentials.

16. **VISUAL DNA**: For each frame, identify the 3-5 elements that make it uniquely recognizable. These are non-negotiable for reconstruction.

17. **STYLE CONSISTENCY**: Note repeated visual patterns, colors, compositions that define the video's aesthetic signature.

18. **COMPLETE COVERAGE**: Analyze 100% of video duration. Every second must be accounted for in the breakdown.

**FINAL REMINDER BEFORE YOU BEGIN:**
- Count the number of distinct shots/scenes in the video FIRST
- Create a SHOT section for EACH one using the template structure
- Do not summarize or skip shots - provide full detailed analysis for EVERY shot
- The Timeline Breakdown section should contain multiple complete shot analyses, not just one or two
- Continue analyzing until you reach the end timestamp of the video

REFERENCE — Use only these labels for classification:

TOP 20 LIGHT CONDITIONS
- GOLDEN HOUR
- BLUE HOUR
- OVERCAST LIGHT
- DIFFUSED LIGHT
- BACKLIGHTING
- SOFT AMBIENT LIGHT
- LOW-KEY LIGHTING
- HIGH-KEY LIGHTING
- WINDOW LIGHT
- DAPPLED LIGHT
- SPOTLIGHT
- TWILIGHT LIGHT
- CANDLELIGHT
- NEON LIGHT
- MOONLIGHT
- STREET LIGHT
- BOUNCED LIGHT
- LENS FLARE
- STUDIO LIGHT
- PATTERN LIGHT

TOP 20 COLOR FILM TYPES
- CINESTILL 800T
- KODAK PORTRA 800
- LOMOGRAPHY X-PRO 200
- KODAK EKTACHROME
- FUJIFILM PRO 400H
- LOMOGRAPHY COLOR NEGATIVE 800
- KODAK EKTAR 100
- REVOLOG KOLOR
- AGFA VISTA PLUS 200
- FUJIFILM VELVIA 50
- FUJIFILM SUPERIA X-TRA
- KODAK GOLD 200
- FUJIFILM PROVIA 100F
- ADOX COLOR IMPLOSION
- AGFA VISTA 400
- LOMOGRAPHY REDSCALE
- KODAK VISION3 500T
- LOMOGRAPHY DIANA F+
- POLAROID ORIGINALS (NOT 35MM)
- FUJIFILM INSTAX MINI (NOT 35MM)
"""

    def analyze_video_file(self, video_path):
        """
        Analyze a local video file using Gemini 2.5 Pro.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            str: Markdown analysis report
        """
        video_path = Path(video_path)
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        file_size_mb = video_path.stat().st_size / (1024 * 1024)
        print(f"[*] Analyzing video: {video_path.name}")
        print(f"[*] File size: {file_size_mb:.2f} MB")
        
        if file_size_mb > 20:
            print("[*] Large file detected, uploading via File API...")
            return self._analyze_with_file_api(video_path)
        else:
            print("[*] Small file detected, using inline data...")
            return self._analyze_inline(video_path)
    
    def _analyze_with_file_api(self, video_path):
        """Analyze video using File API (for files > 20MB)."""
        print("[*] Uploading video to Gemini...")
        
        uploaded_file = self.client.files.upload(file=str(video_path))
        print(f"[✓] File uploaded: {uploaded_file.name}")
        print("[*] Processing video analysis (this may take a minute)...")
        
        response = self.client.models.generate_content(
            model=self.model,
            contents=[uploaded_file],
            config=types.GenerateContentConfig(
                temperature=self.temperature,
                system_instruction=self._create_analysis_prompt(),
            )
        )
        
        return response.text
    
    def _analyze_inline(self, video_path):
        """Analyze video using inline data (for files < 20MB)."""
        print("[*] Reading video file...")
        
        with open(video_path, 'rb') as f:
            video_bytes = f.read()
        
        mime_type = self._get_mime_type(video_path)
        print(f"[*] MIME type: {mime_type}")
        print("[*] Processing video analysis (this may take a minute)...")
        
        response = self.client.models.generate_content(
            model=self.model,
            contents=types.Content(
                parts=[
                    types.Part(
                        inline_data=types.Blob(
                            data=video_bytes,
                            mime_type=mime_type
                        )
                    )
                ]
            ),
            config=types.GenerateContentConfig(
                temperature=self.temperature,
                system_instruction=self._create_analysis_prompt(),
            )
        )
        
        return response.text
    
    def _get_mime_type(self, video_path):
        """Determine MIME type from file extension."""
        ext = video_path.suffix.lower()
        mime_types = {
            '.mp4': 'video/mp4',
            '.mpeg': 'video/mpeg',
            '.mov': 'video/mov',
            '.avi': 'video/avi',
            '.flv': 'video/x-flv',
            '.mpg': 'video/mpg',
            '.webm': 'video/webm',
            '.wmv': 'video/wmv',
            '.3gp': 'video/3gpp',
        }
        return mime_types.get(ext, 'video/mp4')
    
    def save_report(self, analysis, video_path, output_dir="reports"):
        """
        Save the analysis report to a Markdown file.
        
        Args:
            analysis: The analysis text
            video_path: Original video path
            output_dir: Directory to save reports
            
        Returns:
            Path: Path to the saved report
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        video_src = Path(video_path)
        video_name = video_src.stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"{video_name}_analysis_{timestamp}.md"

        # Write the model's analysis directly without embedding/copying the video
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(analysis)

        return output_file


def main():
    """Main function to handle command-line usage."""
    print("=" * 70)
    print("Video Analysis Agent - Powered by Gemini 2.5 Pro")
    print("=" * 70)
    
    # Argument parsing (video path)
    import argparse
    parser = argparse.ArgumentParser(description="Analyze a video with Gemini 2.5 Pro and generate a technical Markdown report.")
    parser.add_argument("video", help="Path to the video file")
    args = parser.parse_args()

    video_path = args.video
    
    try:
        analyzer = VideoAnalyzer()
        
        analysis = analyzer.analyze_video_file(video_path)
        
        report_path = analyzer.save_report(analysis, video_path)
        
        print("\n" + "=" * 70)
        print("[✓] Analysis completed successfully!")
        print(f"[✓] Report saved to: {report_path.absolute()}")
        print(f"[i] Model: {analyzer.model} | Temperature: {analyzer.temperature}")
        print("=" * 70)
        print("\nReport Preview:")
        print("-" * 70)
        print(analysis[:500] + "..." if len(analysis) > 500 else analysis)
        print("-" * 70)
        
    except FileNotFoundError as e:
        print(f"\n[✗] Error: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"\n[✗] Configuration error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[✗] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
