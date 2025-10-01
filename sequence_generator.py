
#!/usr/bin/env python3
"""
Video Sequence Generator
Generates first frame, last frame, and video motion prompts for each sequence
based on video analysis reports.
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


class SequenceGenerator:
    def __init__(self, api_key=None):
        """
        Initialize the sequence generator with Gemini 2.5 Pro.
        
        Args:
            api_key: Gemini API key (defaults to GEMINI_API_KEY env var)
        """
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found. Set it in .env file or pass it directly.")
        
        self.client = genai.Client(api_key=self.api_key)
        self.model = "gemini-2.5-pro"
        # Fixed temperature set in code (no env/CLI override)
        self.temperature = 0.1
        
    def _create_sequence_prompt(self):
        """Create the ultra-optimized prompt for AI model generation (t2i + i2v reconstruction)."""
        return """
You are an ULTRA-OPTIMIZED video reconstruction prompt generator. Transform the provided ultra-comprehensive analysis into MODEL-SPECIFIC, FIDELITY-MAXIMIZING prompts for text-to-image and image-to-video AI systems.

MISSION
Generate prompts that recreate the original video with MAXIMUM fidelity by:
1. **Extracting Visual DNA** from frame state snapshots in the analysis
2. **Applying Advanced Prompt Engineering** techniques (attention weighting, negative prompts, style anchoring)
3. **Maintaining Cross-Sequence Continuity** using the Character/Object Registry from the analysis
4. **Optimizing for Specific AI Models** (FLUX, Midjourney, Kling, Runway)
5. **Ensuring Physical Plausibility** in all motion descriptions
6. **Maximizing Prompt Quality** with model-aware syntax and quality keywords

CORE PRINCIPLES
- **Frame State Extraction**: Pull exact visual descriptions directly from "VISUAL HIERARCHY" sections of frame snapshots
- **Continuity First**: Reference the Character Registry to maintain consistent character/object appearance across ALL sequences
- **Model-Aware Syntax**: Adapt prompt structure to target model capabilities and best practices
- **Physics-Based Motion**: Describe motion with realistic timing, acceleration, momentum, and physical interactions
- **Smart Frame Strategy**: Choose Anchor vs First+Last based on visual change magnitude from analysis
- **Quality Maximization**: Include model-specific quality keywords, negative prompts, and attention weighting

SMART FRAME STRATEGY (VERY IMPORTANT)
- Short sequence threshold: Duration < 4 seconds.
- For short sequences, avoid generating both First and Last frames if they would be too similar → prefer a Single Anchor Frame.
- Use First+Last frames ONLY if there is meaningful visual change (composition, subject positions, lighting, camera angle) within the short duration.
- For durations ≥ 4 seconds, prefer First+Last frames unless the shot is static with negligible change.
- When using First+Last frames, keep both frames stylistically consistent (subjects, grading, aspect) for smooth i2v interpolation while reflecting the intended change.
- Always state the chosen strategy and include a 1–2 sentence rationale.

DECISION HEURISTICS (apply consistently)
- Similarity: If composition and subject arrangement change < ~20% or no major pose/position/light shift → Single Anchor Frame.
- Motion type: Micro‑movements only (breathing, hair sway, subtle handheld) → Single Anchor Frame.
- Camera: Brief but clear pan/tilt/dolly that changes composition meaningfully → First+Last.
- Content change: Subject/object enters/exits frame or framing coverage shifts → First+Last.

ADVANCED PROMPT ENGINEERING TECHNIQUES

### For Text-to-Image Models

**FLUX/SD XL Format:**
- Natural language paragraphs, detailed, specific
- Write as flowing prose: Subject description → Pose/action → Environment → Lighting details → Color palette → Camera specs → Style/era
- Use parentheses sparingly for emphasis on critical quality terms: (highly detailed), (sharp focus)
- Integrate quality keywords naturally at the end: "professional photography", "8k resolution", "sharp focus", "cinematic composition"
- Weave in style anchors: "shot on [film stock]", "aesthetic of [era] cinema", "in the style of [cinematographer]"
- Example: "A young woman with long black hair wearing a red knit sweater stands in soft window light, her body turned to look over her shoulder with a neutral expression. The window to her right creates gentle, diffused illumination that wraps around her form, casting subtle shadows on the left side. The background fades into soft bokeh, keeping all attention on her figure. Shot from eye level with a medium telephoto lens creating shallow depth of field, this portrait evokes the aesthetic of Kodak Portra 800 film with its warm, natural skin tones and gentle color palette. Professional photography, cinematic composition, highly detailed, 8k resolution, sharp focus."

**Negative Prompts** (critical for quality):
- Common negatives: "blurry, out of focus, low quality, jpeg artifacts, watermark, text, distorted, deformed, bad anatomy, cartoon, 3d render, oversaturated"
- Add specific negatives based on content: "multiple heads, extra limbs" for people; "modern elements" for period pieces

### For Image-to-Video Models

**Kling 2.5 Pro / Runway Gen-3 Format:**
- Write as ONE natural language paragraph describing all motion
- Be physics-based and specific about direction, speed, timing
- Flow: Primary subject motion → Secondary motions → Camera behavior → Environmental responses → What stays still → Timing
- Specify motion vectors naturally: "moves from left to right", "walks toward camera", "rotates clockwise"
- Include speed qualifiers: "slowly", "quickly", "smoothly", "gradually"
- Reference physics: momentum, acceleration, natural arcs, gravity where relevant
- Explicitly state what remains stationary to avoid unwanted AI motion
- Add clear timing cues: "over 3 seconds", "duration of 2 seconds", "gradually over the shot"
- Example: "The woman slowly turns her head to look over her left shoulder while her long hair gently sways with the movement, responding to the momentum of her turning motion. The camera remains completely static throughout, maintaining its framing. Soft window light creates subtle shadows that gradually shift across her face as she rotates. The background and all environmental elements remain perfectly still, with no parallax or movement. The entire motion unfolds smoothly over 3 seconds with natural, fluid body mechanics and realistic physics. The shot concludes with her gaze directed over her shoulder, ready to transition to the next sequence."

**Motion Clarity Principles:**
- ONE primary action per prompt (complex actions confuse models)
- State what stays still explicitly
- Describe motion in viewer's perspective (left/right/toward/away)
- Include timing: "over 2 seconds", "quickly", "gradually"
- Reference physics: gravity, momentum, natural arcs

PRACTICAL TIPS (from video_prompt.md)
- Keep wording simple and direct; avoid overly abstract phrasing.
- Use explicit transition keywords when helpful, e.g., "switch to [next shot]".
- Motions should follow physics; avoid implausible behavior.
- Ensure the prompt matches the analysis/image content; don’t invent elements.
- Avoid exact numbers unless necessary; prefer qualitative terms or ranges.
- For Start/End Frames, choosing visually compatible images improves smoothness.
- You may use adverbs of degree (quickly, intensely, gradually) to convey dynamics.

REFERENCE — Use these labels when classifying or tagging lighting/film look:

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

CONTINUITY EXTRACTION (CRITICAL)
Before generating any prompts, extract from the analysis:
1. **Character Consistency Keywords**: From Character Registry, create a single consistent description for each recurring person
   Example: "young East Asian male, early 20s, slim build, short black hair, white tank top, beige open shirt, dark pants"
2. **Object Consistency Keywords**: From Vehicle/Object Registry, note identifying details
3. **Style Anchor Keywords**: From Visual Style Consistency section, identify the core aesthetic terms that must appear in every prompt

These extracted keywords MUST appear in every relevant sequence prompt to maintain continuity.

OUTPUT FORMAT (STRICT)
Generate Markdown with this structure:

```markdown
# Video Reconstruction Guide

**Source**: [Video name from analysis]
**Total Duration**: [MM:SS]
**Total Sequences**: [N]
**Generation Pipeline**: Text-to-Image (Model-Optimized Prompts) → Image-to-Video (Physics-Based Motion)
**Reconstruction Complexity**: [From analysis Reconstruction Overview]

## Continuity Anchors (Use in ALL Prompts)

### Character Consistency
[For each recurring character from Character Registry, provide a condensed description that must appear in their prompts]
- **Character [ID]**: "[concise, consistent description]"

### Style Anchors
[Core aesthetic keywords that must appear in every prompt for visual consistency]
- Film Look: [from analysis]
- Lighting Style: [from analysis]
- Color Palette: [from analysis]
- Era Markers: [from analysis]

---

## Sequence 1: [Descriptive Title from Analysis]
**Timestamp**: [MM:SS] - [MM:SS]
**Duration**: [N.N seconds]
**Scene Type**: [From analysis Shot Type]
**Reconstruction Difficulty**: [From analysis]
**Frame Strategy**: [Single Anchor Frame | First+Last Frames]
**Strategy Rationale**: [Based on analysis frame state changes - assess if visual change warrants First+Last]

### First Frame Image Prompt

[Write ONE flowing natural language paragraph of 250-350 words that synthesizes ALL visual information from the frame snapshot. Extract details from the analysis and weave them together in readable prose. DO NOT use brackets or structured labels - write it as if verbally describing the frame to an artist who will paint it.]

[Flow: Begin with the subject/character (using continuity keywords if recurring character), describe their appearance and clothing in detail, mention their pose and body positioning, note their screen position. Then describe the environment and spatial arrangement of elements. Continue with lighting details (sources, directions, qualities, color temperature, shadows/highlights). Include the color palette with specific color names. Integrate camera specifications naturally (angle, lens character, distance, depth of field). Weave in the style elements (film stock look, era markers, atmosphere, mood). Conclude with quality descriptors. Make it read as one cohesive, vivid description.]

**Example tone**: "A young East Asian male in his early 20s with a slim build and short black hair stands center-left in the frame, wearing a white tank top under an open beige short-sleeve shirt and dark pants. His body is turned slightly to the left at a 30-degree angle, with his weight evenly distributed and arms relaxed at his sides. He occupies approximately 40% of the frame height and is positioned at the upper-left rule-of-thirds intersection. Behind him stretches a wet asphalt street at night, reflecting the cool cyan and blue light from bright fluorescent storefronts in the background. The lighting comes from these commercial sources camera-right, creating a diffused, even illumination with soft shadows and a color temperature around 5600K. The color palette is dominated by deep cyan blues and cool whites, with small accents of warm yellow from distant signs. Shot from a slightly elevated angle about 2 meters above ground level with a normal 50mm lens equivalent, producing a medium depth of field that keeps the subject sharp while softly blurring the background. The aesthetic evokes Cinestill 800T film stock with visible halation around bright lights, characteristic of 1990s Taiwanese New Wave cinema. The overall mood is contemplative and urban. Professional photography, highly detailed, sharp focus, cinematic composition, 8k resolution."

**Negative Prompt**: "blurry, out of focus, low quality, jpeg artifacts, watermark, text, distorted, deformed, bad anatomy, [add scene-specific negatives based on content]"

---

### Last Frame Image Prompt

[If Single Anchor Frame: write "**Not required** - Using Single Anchor Frame strategy. The anchor frame above will drive motion."]

[If First+Last strategy: Write ONE flowing natural language paragraph of 250-350 words describing the END STATE after motion. Use the same prose style as First Frame but focus on how things have CHANGED - different subject positions, altered composition, new camera position, lighting shifts, etc. Maintain character consistency keywords. Begin with what moved/changed, then provide the complete scene description in its final state. Make it vivid and cohesive.]

**Negative Prompt**: [Same as First Frame]

---

### Video Motion Prompt

[Write ONE flowing natural language paragraph of 200-300 words that describes ALL motion in the shot. Extract from SUBJECTS & MOTION TRACKING and Camera Motion Analysis sections. Synthesize into readable prose that describes what moves, how it moves, and what stays still. Use physics-based language with natural timing cues. DO NOT use brackets or structured labels.]

[Flow: Begin with the primary subject's motion (what they do, direction, speed, style/gait). Then describe any secondary moving subjects or objects. Mention camera behavior explicitly (static/moving, type of movement, smooth/jerky). Include environmental motion (wind effects, water, traffic, background elements). Add timing cues naturally ("over 3 seconds", "gradually", "suddenly"). Reference physics where relevant (momentum, natural movement arcs, gravity). Explicitly state what remains stationary. If applicable, note how the shot transitions to the next sequence. Make it read as one cohesive motion description.]

**Example tone**: "The man walks slowly from left to right across the wet tiled sidewalk, moving at a relaxed pace of approximately 1 meter per second with a casual, slightly swaggering gait. His arms swing naturally in opposition to his legs, and he briefly glances down at his watch on his left wrist as he passes behind a parked silver van. The camera remains completely static throughout, maintaining its elevated viewpoint from across the street. In the background, the bright fluorescent lights from the storefronts stay constant, creating steady reflections on the wet pavement. The dark foreground silhouettes remain motionless, framing the scene. The overall pacing is slow and contemplative, with the motion lasting approximately 2 seconds as the subject traverses from the left third to the right third of the frame. The shot concludes with the subject partially occluded by the van, establishing a sense of urban anonymity before transitioning to the next sequence. All movement follows natural physics with realistic weight distribution and momentum."

**Motion Complexity**: [Easy/Medium/Hard from analysis]
**Recommended Model**: [Kling 2.5 Pro or Runway Gen-3 based on complexity]

---

[Repeat above structure for each sequence, maintaining character/style continuity...]

---

## Technical Specifications & Generation Strategy

### Model Selection Rationale
**Text-to-Image**: [From analysis AI Generation Strategy section]
- Primary: [Recommended model + why]
- Alternative: [Alternative model for specific challenges]

**Image-to-Video**: [From analysis recommendations]
- Primary: [Recommended model + why]
- Motion Complexity Notes: [Which sequences are easy/hard]

### Consistent Style Elements (Enforce Across ALL Prompts)
[Pull from analysis Visual Style Consistency and Technical Specifications Summary]
- **Film Stock**: [Specific film from reference list]
- **Light Conditions**: [Dominant conditions from reference list]
- **Color Grading**: [Specific style]
- **Grain Level**: [Amount and character]
- **Aspect Ratio**: [From analysis]
- **Era/Period**: [With specific markers]
- **Directorial Style**: [Named influences]

### Continuity Maintenance Checklist
For each prompt, verify:
- [ ] Character appearance matches Character Registry
- [ ] Style anchors present in description
- [ ] Lighting consistent with analysis
- [ ] Color palette consistent
- [ ] Film look keywords included
- [ ] Quality keywords added
- [ ] Negative prompts included

### Generation Workflow
1. **Prepare**: Review Continuity Anchors section
2. **Generate Images**:
   - If Anchor strategy: Generate 1 image per sequence
   - If First+Last: Generate 2 images per sequence
   - Use same seed for First+Last to maintain consistency
   - Include negative prompts
3. **Generate Videos**:
   - Feed images + motion prompts to i2v model
   - Monitor for physics violations or unwanted motion
   - Regenerate if motion doesn't match analysis
4. **Concatenate**: Join all sequences in temporal order
5. **Post-Process**: Color grade if needed to match analysis Film Look

### Critical Success Factors
[From analysis Critical Reconstruction Notes]
- **Must-Have Elements**: [List from analysis]
- **Failure Points**: [What would break fidelity]
- **Acceptable Variations**: [What can differ]
```

WRITING GUIDELINES
1. Use clear, simple sentences; avoid unnecessary jargon.
2. Prefer qualitative ranges over exact numbers unless required.
3. Maintain consistency of characters, lighting, and style across sequences.
4. Motions should obey physical plausibility.
5. Do not invent elements beyond the analysis report.

Now analyze the video analysis report and generate the complete sequence breakdown using the Smart Frame Strategy and the prompt formulas above.
"""

    def generate_sequences(self, analysis_file_path):
        """
        Generate sequences with image and video prompts from an analysis file.
        
        Args:
            analysis_file_path: Path to the video analysis Markdown file
            
        Returns:
            str: Generated sequences in Markdown format
        """
        analysis_file_path = Path(analysis_file_path)
        
        if not analysis_file_path.exists():
            raise FileNotFoundError(f"Analysis file not found: {analysis_file_path}")
        
        print(f"[*] Reading analysis: {analysis_file_path.name}")
        
        with open(analysis_file_path, 'r', encoding='utf-8') as f:
            analysis_content = f.read()
        
        print(f"[*] Analysis length: {len(analysis_content)} characters")
        print("[*] Generating sequences with Gemini 2.5 Pro...")
        print("    This may take 1-2 minutes for detailed sequence breakdown...")
        
        response = self.client.models.generate_content(
            model=self.model,
            contents=[f"VIDEO ANALYSIS REPORT:\n\n{analysis_content}"] ,
            config=types.GenerateContentConfig(
                temperature=self.temperature,
                system_instruction=self._create_sequence_prompt(),
            )
        )
        
        return response.text
    
    def save_sequences(self, sequences, original_analysis_path, output_dir="sequences"):
        """
        Save the generated sequences to a Markdown file.
        
        Args:
            sequences: The generated sequences text
            original_analysis_path: Original analysis file path
            output_dir: Directory to save sequence files
            
        Returns:
            Path: Path to the saved sequences file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        analysis_name = Path(original_analysis_path).stem
        if analysis_name.endswith('_analysis'):
            base_name = analysis_name.replace('_analysis', '')
        else:
            base_name = analysis_name
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"{base_name}_sequences_{timestamp}.md"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(sequences)
        
        return output_file


def main():
    """Main function to handle command-line usage."""
    print("=" * 70)
    print("Video Sequence Generator - Powered by Gemini 2.5 Pro")
    print("Generate Smart Frames (Anchor or First+Last) + Motion Prompts")
    print("=" * 70)
    
    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  python sequence_generator.py <analysis_file.md>")
        print("\nDescription:")
        print("  Generates sequences with image and video prompts from analysis:")
        print("    • First Frame Image Prompt (for text-to-image models)")
        print("    • Last Frame Image Prompt (for text-to-image models)")
        print("    • Video Motion Prompt (for image-to-video models)")
        print("\nExamples:")
        print("  python sequence_generator.py reports/video_analysis_20240101.md")
        print("\nSupported formats:")
        print("  Markdown (.md) analysis files")
        print("\nOutput:")
        print("  Sequence guide saved to 'sequences/' directory")
        print("\nOptimized for:")
        print("  Kling 2.5 Pro (First Frame + Last Frame mode)")
        print("  Compatible with Runway, Pika, Sora, and other i2v models")
        sys.exit(1)
    
    analysis_file = sys.argv[1]
    
    try:
        generator = SequenceGenerator()
        
        sequences = generator.generate_sequences(analysis_file)
        
        output_file = generator.save_sequences(sequences, analysis_file)
        
        print("\n" + "=" * 70)
        print("[✓] Sequence generation completed successfully!")
        print(f"[✓] Sequences saved to: {output_file.absolute()}")
        print(f"[i] Model: {generator.model} | Temperature: {generator.temperature}")
        print("=" * 70)
        print("\nNext Steps:")
        print("  1. Review the sequence breakdown")
        print("  2. Generate first frames using text-to-image models")
        print("  3. Generate last frames using text-to-image models")
        print("  4. Use image-to-video models to generate motion")
        print("  5. Concatenate all sequences to reconstruct the video")
        print("=" * 70)
        print("\nSequence Preview:")
        print("-" * 70)
        print(sequences[:800] + "..." if len(sequences) > 800 else sequences)
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
