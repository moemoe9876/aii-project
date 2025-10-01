#!/usr/bin/env python3
"""
Complete Video-to-Image-to-Video Pipeline
Orchestrates: Video Download → Analysis → Sequence Generation
"""

import sys
import os
from pathlib import Path
import subprocess


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*70}")
    print(f"[*] {description}")
    print(f"{'='*70}")
    
    result = subprocess.run(cmd, shell=False)
    
    if result.returncode != 0:
        print(f"[✗] Error: {description} failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    
    return result


def main():
    """Main function to orchestrate the complete pipeline."""
    print("=" * 70)
    print("AI Video Reconstruction Pipeline")
    print("Video → Analysis → Sequences (First Frame + Last Frame + Motion)")
    print("=" * 70)
    
    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  python run_pipeline.py <video_file_or_url>")
        print("\nDescription:")
        print("  Runs the complete pipeline:")
        print("  1. Downloads video (if URL provided)")
        print("  2. Analyzes video with Gemini 2.5 Pro")
        print("  3. Generates sequences with:")
        print("     • First Frame Image Prompts")
        print("     • Last Frame Image Prompts")
        print("     • Video Motion Prompts")
        print("\nExamples:")
        print("  python run_pipeline.py downloads/video.mp4")
        print("  python run_pipeline.py https://instagram.com/p/...")
        print("\nOutput:")
        print("  • Analysis report in reports/")
        print("  • Sequence prompts in sequences/")
        sys.exit(1)
    
    input_arg = sys.argv[1]
    
    # Get Python path from venv
    python_path = "venv/bin/python" if Path("venv/bin/python").exists() else "python"
    
    # Step 1: Check if input is URL or file
    if input_arg.startswith('http://') or input_arg.startswith('https://'):
        run_command(
            [python_path, "video_downloader.py", input_arg],
            "Step 1: Downloading video"
        )
        
        downloads_dir = Path("downloads")
        if not downloads_dir.exists():
            print("\n[✗] Error: Downloads directory not found")
            sys.exit(1)
        
        video_files = sorted(downloads_dir.glob("*.mp4"), key=lambda x: x.stat().st_mtime, reverse=True)
        if not video_files:
            video_files = sorted(downloads_dir.glob("*.*"), key=lambda x: x.stat().st_mtime, reverse=True)
        
        if not video_files:
            print("\n[✗] Error: No video file found after download")
            sys.exit(1)
        
        video_file = video_files[0]
        print(f"\n[*] Using downloaded video: {video_file.name}")
    else:
        video_file = Path(input_arg)
        if not video_file.exists():
            print(f"\n[✗] Error: Video file not found: {video_file}")
            sys.exit(1)
        print(f"\n[*] Using video: {video_file.name}")
    
    # Step 2: Analyze video
    run_command(
        [python_path, "video_analyzer.py", str(video_file)],
        "Step 2: Analyzing video"
    )
    
    # Find the most recent analysis file
    reports_dir = Path("reports")
    if not reports_dir.exists():
        print("\n[✗] Error: Reports directory not found")
        sys.exit(1)
    
    analysis_files = sorted(reports_dir.glob("*_analysis_*.md"), key=lambda x: x.stat().st_mtime, reverse=True)
    
    if not analysis_files:
        print("\n[✗] Error: No analysis file found")
        sys.exit(1)
    
    latest_analysis = analysis_files[0]
    print(f"\n[*] Found analysis: {latest_analysis.name}")
    
    # Step 3: Generate sequences
    run_command(
        [python_path, "sequence_generator.py", str(latest_analysis)],
        "Step 3: Generating sequences with image and video prompts"
    )
    
    # Find the most recent sequences file
    sequences_dir = Path("sequences")
    if sequences_dir.exists():
        sequence_files = sorted(sequences_dir.glob("*_sequences_*.md"), key=lambda x: x.stat().st_mtime, reverse=True)
        if sequence_files:
            latest_sequences = sequence_files[0]
            
            print("\n" + "=" * 70)
            print("✓ PIPELINE COMPLETED SUCCESSFULLY!")
            print("=" * 70)
            print(f"\nGenerated Files:")
            print(f"  📄 Video Analysis: {latest_analysis.name}")
            print(f"  🎬 Sequence Guide:  {latest_sequences.name}")
            print("\n" + "=" * 70)
            print("Next Steps:")
            print("=" * 70)
            print("1. Open the sequence guide to review all sequences")
            print("2. For each sequence:")
            print("   a) Generate FIRST FRAME using text-to-image model")
            print("      (Use: Stable Diffusion, FLUX, Midjourney)")
            print("   b) Generate LAST FRAME using text-to-image model")
            print("   c) Generate VIDEO using image-to-video model")
            print("      (Use: Kling 2.5 Pro First Frame + Last Frame mode)")
            print("3. Concatenate all video sequences in order")
            print("4. Add audio/music if needed")
            print("=" * 70)
            print(f"\n📁 Files Location:")
            print(f"   Analysis: {latest_analysis.absolute()}")
            print(f"   Sequences: {latest_sequences.absolute()}")
            print("=" * 70)


if __name__ == "__main__":
    main()
