# AI Video Reconstruction Pipeline

Transform any video into AI-generated sequences using a complete **Video → Analysis → First Frame + Last Frame + Motion** pipeline.

## 🎯 Overview

This project recreates videos using a multi-stage AI pipeline powered by **Gemini 2.5 Pro**:

1. **Video Downloader** - Download videos from Instagram, Twitter, YouTube, and other platforms
2. **Video Analyzer** - Generate detailed technical analysis reports of cinematography and style
3. **Cinematography Quality Profiling** - Extract technical "DNA" (grain, lens, color science) for consistent recreation ⭐ NEW
4. **Sequence Generator** - Break videos into sequences with:
   - **First Frame Image Prompts** (for text-to-image models)
   - **Last Frame Image Prompts** (for text-to-image models)
   - **Video Motion Prompts** (for image-to-video models)
   - **Quality Keywords** (camera/lens/film characteristics automatically injected) ⭐ NEW

## 🎬 How It Works

```
Video → Analysis → Sequences → Image Generation → Video Generation
                                    ↓                    ↓
                              (First + Last          (Kling 2.5 Pro
                               Frames via            Image-to-Video)
                               FLUX/SD/MJ)
```

Each sequence is recreated by:
1. Generating the **first frame** as an image
2. Generating the **last frame** as an image
3. Using an **image-to-video model** to create motion between frames
4. Concatenating all sequences to rebuild the complete video

## 🚀 Quick Start

### Installation

```bash
# Clone or download this repository
cd ai-project

# Install dependencies
pip install -r requirements.txt

# Configure your Gemini API key in .env
echo "GEMINI_API_KEY=your_api_key_here" > .env
```

### Complete Workflow

```bash
# Option 1: Run complete pipeline with local video
python run_pipeline.py downloads/video.mp4

# Option 2: Run pipeline with video URL (auto-downloads)
python run_pipeline.py https://instagram.com/p/example
```

That's it! The pipeline will:
1. Analyze the video's technical composition
2. Break it into logical sequences
3. Generate first frame, last frame, and motion prompts for each sequence

## 📋 Pipeline Stages

### 1. Video Analyzer
Analyzes videos and generates technical reports:
- Scene overview (location, time, mood, atmosphere)
- Visual composition (camera angles, shot types, framing)
- Lighting & color (sources, style, palette, contrast)
- Characters & objects (detailed descriptions, clothing, props)
- Textures & environment (materials, surfaces, details)
- Cinematography style (influences, era, genre references)
- **⭐ NEW: Cinematography Quality Profile** - Extracts technical characteristics:
  - Camera/sensor characteristics (film vs digital, resolution, dynamic range)
  - Lens characteristics (sharpness, distortion, vignetting, bokeh character)
  - Film stock/color science (grain structure, color response curves)
  - Exposure & dynamic range (highlight/shadow handling)
  - Motion characteristics (frame rate feel, motion blur)
  - Generates **Core Technical Keywords** for consistent recreation

### 2. Sequence Generator
Breaks videos into sequences with three prompts each:

**First Frame Image Prompt**
- Complete description of opening frame
- Camera setup, subjects, lighting, composition
- Style tags for text-to-image generation
- 200-300 words optimized for Stable Diffusion/FLUX/Midjourney

**Last Frame Image Prompt**
- Complete description of ending frame
- Shows final positions after all motion
- Maintains consistency with first frame
- 200-300 words optimized for text-to-image generation

**Video Motion Prompt**
- Camera movements (pan, tilt, dolly, tracking)
- Subject movements (speed, direction, timing)
- Environmental changes
- 150-200 words optimized for image-to-video models

## 📁 Project Structure

```
ai-project/
├── video_downloader.py     # Download videos from various platforms
├── video_analyzer.py        # Analyze videos with Gemini 2.5 Pro
├── sequence_generator.py    # Generate sequences with image/video prompts
├── run_pipeline.py          # Run complete workflow
├── requirements.txt         # Python dependencies
├── .env                     # API keys (not committed)
├── downloads/              # Downloaded videos
├── reports/                # Video analysis reports (.md)
└── sequences/              # Generated sequences with prompts (.md)
```

## 💡 Use Cases

### Video Reconstruction
- Recreate any video using AI generation models
- Break complex videos into manageable sequences
- Generate consistent frames across sequences
- Perfect for video style transfer and recreation

### AI Content Creation
- Generate videos from reference footage
- Create variations of existing videos
- Build consistent visual narratives
- Experiment with different generation models

### Film Study & Analysis
- Understand cinematographic techniques
- Extract technical specifications
- Study shot composition and transitions
- Build reference libraries

## 🎨 Example Output

### Input Video
30-second video of a nighttime street scene in 1990s Taipei

### Generated Sequence (Excerpt)

**Sequence 1: Alley Walker**
- **Timestamp**: 00:00 - 00:02
- **Duration**: 2 seconds

**First Frame Image Prompt:**
```
A high-angle, static medium shot capturing a moment in a bustling East Asian 
city alley at dusk. A young East Asian man in his early 20s with short dark 
hair stands center frame, wearing a loose-fitting light beige jacket over a 
white graphic t-shirt and classic blue denim jeans. The alley is paved with 
light grey tiles. To the right, a brightly lit jewelry store displays watches 
in glass cases. The lighting is a mix of cool cyan fluorescent light and 
fading dusk ambiance. Shot on 35mm film with noticeable grain and slightly 
desaturated moody color palette, 1990s Taiwanese New Wave cinema aesthetic.

Tags: 1990s Taiwanese cinema, high-angle shot, urban alley, dusk, film grain, 
35mm, young man, vintage fashion, moody, cinematic, Tsai Ming-liang style
```

**Last Frame Image Prompt:**
```
Same high-angle view. The young man has walked a few steps forward toward the 
center of the alley, body angled slightly left. He's captured mid-stride with 
right leg forward, head turned slightly left. Silver minivan remains in 
foreground, lit storefronts on either side. Lighting, color, and filmic 
texture identical to first frame...
```

**Video Motion Prompt:**
```
Camera Movement: Static, fixed high-angle position
Subject Motion: Young man walks at slow, casual pace from starting position 
toward alley center, natural unhurried movement
Environmental Changes: None, lighting and background constant
Pacing: Slow, contemplative
Complete Description: From static high-angle viewpoint, the young man walks 
forward at relaxed pace taking a few steps deeper into alley. Gaze shifts 
slightly as he walks. Scene is observational and quiet with only his movement 
providing primary action...
```

## 🔧 Advanced Usage

### Individual Tools

```bash
# Download only
python video_downloader.py <url>

# Analyze only
python video_analyzer.py <video_file>

# Generate sequences only (from existing analysis)
python sequence_generator.py reports/video_analysis_20240101.md
```

### Batch Processing

```bash
# Download multiple videos
python video_downloader.py --file urls.txt

# Analyze all videos in downloads/
for video in downloads/*.mp4; do
    python video_analyzer.py "$video"
done

# Generate sequences for all analyses
for report in reports/*_analysis_*.md; do
    python sequence_generator.py "$report"
done
```

## 🎬 Generation Workflow

After running the pipeline, use the generated sequences to create your video:

### Step 1: Generate First Frames
Use text-to-image models (FLUX, Stable Diffusion, Midjourney) with the **First Frame Image Prompt** for each sequence.

### Step 2: Generate Last Frames
Use text-to-image models with the **Last Frame Image Prompt** for each sequence.

### Step 3: Generate Videos
Use image-to-video models (Kling 2.5 Pro, Runway, Pika, Sora) with:
- First frame image
- Last frame image  
- Motion prompt

**Recommended**: Kling 2.5 Pro (First Frame + Last Frame mode)

### Step 4: Concatenate Sequences
Use video editing software or ffmpeg to join all sequences in order.

```bash
# Example with ffmpeg
ffmpeg -f concat -safe 0 -i sequence_list.txt -c copy output.mp4
```

## 📖 Documentation

- **[CINEMATOGRAPHY_QUALITY_MATCHING.md](CINEMATOGRAPHY_QUALITY_MATCHING.md)** - ⭐ NEW: How the quality matching system works
- **[USAGE.md](USAGE.md)** - Detailed usage instructions, examples, and troubleshooting (if available)

## 🛠️ Requirements

- Python 3.8+
- Gemini API key (get one at [ai.google.dev](https://ai.google.dev))
- Dependencies: `google-genai`, `python-dotenv`, `yt-dlp`

## ⚙️ Configuration

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_api_key_here
```

## 🤝 Credits

- **Gemini 2.5 Pro** - Google's multimodal AI model for video understanding
- **yt-dlp** - Video download tool

## 📄 License

This project is for educational and research purposes.

## 🐛 Troubleshooting

### "GEMINI_API_KEY not found"
- Ensure `.env` file exists with valid API key
- Check that `python-dotenv` is installed

### Slow Processing
- Normal for videos > 1 minute
- Gemini 2.5 Pro may take 30-60 seconds for analysis

### Video Download Fails
- Verify the URL is accessible
- Some platforms may have restrictions
- Try updating yt-dlp: `pip install -U yt-dlp`

## 💬 Support

For issues or questions, check:
1. [USAGE.md](USAGE.md) for detailed documentation
2. Error messages for specific guidance
3. Gemini API documentation at [ai.google.dev](https://ai.google.dev)
# video-analyser
