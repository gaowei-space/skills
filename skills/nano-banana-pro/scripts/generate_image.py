#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "google-genai>=1.0.0",
#     "pillow>=10.0.0",
# ]
# ///
"""
Generate images using Google's Nano Banana Pro (Gemini 3 Pro Image) API.

Usage:
    uv run generate_image.py --prompt "your image description" --filename "output.png" [--resolution 1K|2K|4K] [--model MODEL] [--api-key KEY]

Multi-image editing (up to 14 images):
    uv run generate_image.py --prompt "combine these images" --filename "output.png" -i img1.png -i img2.png -i img3.png
"""

import argparse
import os
import re
import sys
from pathlib import Path


def get_api_key(provided_key: str | None) -> str | None:
    """Get API key from argument first, then environment."""
    if provided_key:
        return provided_key
    return os.environ.get("GEMINI_API_KEY")


def get_base_url(provided_url: str | None) -> str | None:
    """Get custom API base URL from argument first, then environment."""
    if provided_url:
        return provided_url
    return os.environ.get("GEMINI_BASE_URL")


def get_api_version(provided_version: str | None) -> str | None:
    """Get API version from argument first, then environment."""
    if provided_version:
        return provided_version
    return os.environ.get("GEMINI_API_VERSION")


def main():
    parser = argparse.ArgumentParser(
        description="Generate images using Nano Banana Pro (Gemini 3 Pro Image)"
    )
    parser.add_argument(
        "--prompt", "-p",
        required=True,
        help="Image description/prompt"
    )
    parser.add_argument(
        "--filename", "-f",
        required=True,
        help="Output filename (e.g., sunset-mountains.png)"
    )
    parser.add_argument(
        "--input-image", "-i",
        action="append",
        dest="input_images",
        metavar="IMAGE",
        help="Input image path(s) for editing/composition. Can be specified multiple times (up to 14 images)."
    )
    parser.add_argument(
        "--resolution", "-r",
        choices=["1K", "2K", "4K"],
        default=None,
        help="Output resolution: 1K, 2K, or 4K (default behavior auto-detects when input images are provided)"
    )
    parser.add_argument(
        "--api-key", "-k",
        help="Gemini API key (overrides GEMINI_API_KEY env var)"
    )
    parser.add_argument(
        "--base-url",
        help="Custom Gemini-compatible base URL (overrides GEMINI_BASE_URL env var)"
    )
    parser.add_argument(
        "--api-version",
        help="Gemini API version, e.g. v1beta (overrides GEMINI_API_VERSION env var)"
    )
    parser.add_argument(
        "--model",
        default="gemini-3.1-flash-image-preview",
        help="Model name for image generation (default: gemini-3.1-flash-image-preview)"
    )

    args = parser.parse_args()

    # Get API key
    api_key = get_api_key(args.api_key)
    if not api_key:
        print("Error: No API key provided.", file=sys.stderr)
        print("Please either:", file=sys.stderr)
        print("  1. Provide --api-key argument", file=sys.stderr)
        print("  2. Set GEMINI_API_KEY environment variable", file=sys.stderr)
        sys.exit(1)

    base_url = get_base_url(args.base_url)
    api_version = get_api_version(args.api_version)

    # Import here after checking API key to avoid slow import on error
    from google import genai
    from google.genai import types
    from PIL import Image as PILImage

    # Initialise client
    http_options = types.HttpOptions(
        base_url=base_url,
        api_version=api_version,
    )
    client = genai.Client(api_key=api_key, http_options=http_options)

    # Set up output path
    output_path = Path(args.filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load input images if provided (up to 14 supported by Nano Banana Pro)
    input_images = []
    output_resolution = args.resolution or "1K"
    if args.input_images:
        if len(args.input_images) > 14:
            print(f"Error: Too many input images ({len(args.input_images)}). Maximum is 14.", file=sys.stderr)
            sys.exit(1)

        max_input_dim = 0
        for img_path in args.input_images:
            try:
                with PILImage.open(img_path) as img:
                    copied = img.copy()
                    width, height = copied.size
                input_images.append(copied)
                print(f"Loaded input image: {img_path}")

                # Track largest dimension for auto-resolution
                max_input_dim = max(max_input_dim, width, height)
            except Exception as e:
                print(f"Error loading input image '{img_path}': {e}", file=sys.stderr)
                sys.exit(1)

        # Auto-detect resolution only when user did not explicitly pass --resolution
        if args.resolution is None and max_input_dim > 0:
            if max_input_dim >= 3000:
                output_resolution = "4K"
            elif max_input_dim >= 1500:
                output_resolution = "2K"
            else:
                output_resolution = "1K"
            print(f"Auto-detected resolution: {output_resolution} (from max input dimension {max_input_dim})")

    # Build contents (images first if editing, prompt only if generating)
    if input_images:
        contents = [*input_images, args.prompt]
        img_count = len(input_images)
        print(f"Processing {img_count} image{'s' if img_count > 1 else ''} with resolution {output_resolution}...")
    else:
        contents = args.prompt
        print(f"Generating image with resolution {output_resolution}...")

    try:
        response = client.models.generate_content(
            model=args.model,
            contents=contents,
            config=types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"],
                image_config=types.ImageConfig(
                    image_size=output_resolution
                )
            )
        )

        # Process response and convert to PNG
        image_saved = False
        for part in response.parts:
            if part.text is not None:
                print(f"Model response: {part.text}")
                # Some compatible gateways return image URL in markdown/plain text
                text = part.text.strip()
                md_match = re.search(r"\((https?://[^\s)]+)\)", text)
                url_match = re.search(r"https?://\S+", text)
                image_url = md_match.group(1) if md_match else (url_match.group(0) if url_match else None)
                if image_url and not image_saved:
                    try:
                        import requests
                        from io import BytesIO

                        resp = requests.get(image_url, timeout=120)
                        resp.raise_for_status()
                        image = PILImage.open(BytesIO(resp.content))
                        if image.mode == 'RGBA':
                            rgb_image = PILImage.new('RGB', image.size, (255, 255, 255))
                            rgb_image.paste(image, mask=image.split()[3])
                            rgb_image.save(str(output_path), 'PNG')
                        elif image.mode == 'RGB':
                            image.save(str(output_path), 'PNG')
                        else:
                            image.convert('RGB').save(str(output_path), 'PNG')
                        image_saved = True
                    except Exception as dl_err:
                        print(f"Warning: failed to download generated image URL: {dl_err}", file=sys.stderr)
            elif part.inline_data is not None:
                # Convert inline data to PIL Image and save as PNG
                from io import BytesIO

                # inline_data.data is already bytes, not base64
                image_data = part.inline_data.data
                if isinstance(image_data, str):
                    # If it's a string, it might be base64
                    import base64
                    image_data = base64.b64decode(image_data)

                image = PILImage.open(BytesIO(image_data))

                # Ensure RGB mode for PNG (convert RGBA to RGB with white background if needed)
                if image.mode == 'RGBA':
                    rgb_image = PILImage.new('RGB', image.size, (255, 255, 255))
                    rgb_image.paste(image, mask=image.split()[3])
                    rgb_image.save(str(output_path), 'PNG')
                elif image.mode == 'RGB':
                    image.save(str(output_path), 'PNG')
                else:
                    image.convert('RGB').save(str(output_path), 'PNG')
                image_saved = True

        if image_saved:
            full_path = output_path.resolve()
            print(f"\nImage saved: {full_path}")
            # OpenClaw parses MEDIA tokens and will attach the file on supported providers.
            print(f"MEDIA: {full_path}")
        else:
            print("Error: No image was generated in the response.", file=sys.stderr)
            sys.exit(1)

    except Exception as e:
        print(f"Error generating image: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
