# Minimax Nodes for Griptape

A comprehensive node library for integrating Minimax's powerful AI generation APIs into Griptape Nodes workflows. Generate stunning images, videos, and music using state-of-the-art AI models.

## 📦 Installation

### Prerequisites

- Griptape Nodes installed
- Python 3.12 or higher
- Minimax API key ([Get one here](https://platform.minimax.chat/))

### Install from Repository

1. Clone this repository:
```bash
git clone https://github.com/yourusername/griptape-nodes-library-minimax.git
cd griptape-nodes-library-minimax
```

2. Open the **Settings** → **Libraries** menu and add the full path to `/minimax/griptape_nodes_library.json` in your current working directory. You can get your current working directory with the `pwd` command on MacOS or Linux.

3. Add your MINIMAX_API_KEY to Griptape Nodes in the **Settings** → **API Keys & Secrets** menu

4. Drag a node from the Minimax category in the Node Sidebar onto your canvas and start creating!

## 🎨 Available Nodes

### 🖼️ Image Generation

#### **Minimax Text-to-Image**

Generate high-quality images from text descriptions using Minimax's image generation API.

**Inputs:**
- **Prompt** (str): Text description of the image to generate (up to 1500 characters)
- **Model** (str): Model selection (default: `image-01`)
- **Aspect Ratio** (str): Image dimensions or "Use height and width" for custom sizes
  - Options: `1:1 (1024x1024)`, `16:9 (1920x1080)`, `9:16 (1080x1920)`, `4:3 (1536x1152)`, `3:4 (1152x1536)`, `21:9 (2520x1080)`, `9:21 (1080x2520)`, or custom
- **Height** (int): Custom height when using custom dimensions (512-2048, step 8, default: 1024)
- **Width** (int): Custom width when using custom dimensions (512-2048, step 8, default: 1024)
- **Seed** (int): Random seed for reproducibility (-1 for random, default: -1)
- **Number of Images** (int, hidden): Generate multiple images (1-9, default: 1)
- **Prompt Optimizer** (bool): Automatically enhance prompts (default: False)

**Outputs:**
- **Image** (ImageUrlArtifact): Generated image
- **Images** (list[ImageUrlArtifact]): All generated images when num_images > 1
- **Provider Response** (dict): Full API response

**Features:**
- Support for multiple aspect ratios and custom dimensions
- Automatic dimension rounding to multiples of 8
- Multiple image generation (up to 9 images)
- Optional prompt optimization
- Seed-based reproducibility

#### **Minimax Image-to-Image**

Generate images from reference images using Minimax's image-to-image API. Perfect for portrait generation and character consistency.

**Inputs:**
- **Prompt** (str): Text description of the image (up to 1500 characters)
- **Reference Image** (ImageArtifact | ImageUrlArtifact): Reference image for subject
  - Best results: single front-facing portrait photo
  - Formats: JPG, JPEG, PNG (< 10MB)
- **Model** (str): Model selection
  - `image-01` (default): Standard quality
  - `image-01-live`: Real-time optimized
- **Subject Type** (str, hidden): Subject type (`character` - currently only option)
- **Aspect Ratio** (str): Image dimensions or "Use height and width" for custom sizes
  - Options: `1:1`, `16:9`, `4:3`, `3:2`, `2:3`, `3:4`, `9:16`, `21:9`, or custom
- **Height** (int, hidden): Custom height when using custom dimensions (512-2048, multiple of 8)
- **Width** (int, hidden): Custom width when using custom dimensions (512-2048, multiple of 8)
- **Seed** (int): Random seed for reproducibility (-1 for random, default: -1)
- **Number of Images** (int, hidden): Generate multiple images (1-9, default: 1)
- **Prompt Optimizer** (bool): Automatically enhance prompts (default: False)

**Outputs:**
- **Image** (ImageUrlArtifact): Generated image (single result)
- **Images** (list[ImageUrlArtifact]): All generated images (when num_images > 1)
- **Provider Response** (dict): Full API response

**Features:**
- Reference image support with automatic format validation
- Smart localhost URL detection and base64 conversion
- Multiple aspect ratios and custom dimensions (image-01 only)
- Multiple image generation (up to 9 images)
- Character consistency across generations

**Use Cases:**
- **Portrait Generation**: Create consistent character portraits
  - Game character design variations
  - Avatar customization
  - Profile picture generation
- **Style Transfer**: Apply styles to reference images
  - Artistic interpretations
  - Costume/outfit variations
  - Different lighting and poses

### 🎥 Video Generation

#### **Minimax Text-to-Video**

Generate videos from text prompts with advanced camera controls and motion descriptions.

**Inputs:**
- **Prompt** (str): Text description of the video (up to 2000 characters)
  - Supports camera movement commands: `[Truck left/right]`, `[Pan left/right]`, `[Push in]`, `[Pull out]`, `[Pedestal up/down]`, `[Tilt up/down]`, `[Zoom in/out]`, `[Shake]`, `[Tracking shot]`, `[Static shot]`
- **Model** (str): Model selection
  - `MiniMax-Hailuo-02` (default): Advanced model with multiple resolutions
  - `T2V-01-Director`: Director-optimized model
  - `T2V-01`: Standard text-to-video model
- **Duration** (int): Video length in seconds (6 or 10)
  - 10s only available for MiniMax-Hailuo-02 at 512P/768P
- **Resolution** (str): Video resolution
  - MiniMax-Hailuo-02: 512P, 768P, 1080P
  - Other models: 720P only
- **Prompt Optimizer** (bool): Automatically optimize prompts (default: True)
- **Fast Pretreatment** (bool): Reduce optimization time for Hailuo-02 (default: False)

**Outputs:**
- **Video URL** (VideoUrlArtifact): Generated video
- **Task ID** (str): API task identifier
- **Provider Response** (dict): Full API response

**Features:**
- Advanced camera movement controls
- Multiple resolution options
- Asynchronous processing with polling
- Model-specific parameter validation
- 10-minute maximum processing time

#### **Minimax Image-to-Video**

Generate videos from a starting frame image with motion descriptions.

**Inputs:**
- **First Frame Image** (ImageArtifact | ImageUrlArtifact): Starting frame for the video
  - Formats: JPG, JPEG, PNG, WebP
  - Size: < 20MB
  - Dimensions: Short edge > 300px
  - Aspect ratio: Between 2:5 and 5:2
- **Prompt** (str): Video motion description (up to 2000 characters, optional)
- **Model** (str): Model selection
  - `MiniMax-Hailuo-02` (default)
  - `I2V-01-Director`: Director-optimized image-to-video
  - `I2V-01-live`: Live-action optimized
  - `I2V-01`: Standard image-to-video
- **Duration** (int): Video length (6 or 10 seconds)
- **Resolution** (str): Video resolution
  - MiniMax-Hailuo-02: 512P, 768P, 1080P
  - Other models: 720P only
- **Prompt Optimizer** (bool): Optimize prompts (default: False)
- **Fast Pretreatment** (bool): Faster processing for Hailuo-02 (default: False)

**Outputs:**
- **Video URL** (VideoUrlArtifact): Generated video
- **Task ID** (str): API task identifier
- **Provider Response** (dict): Full API response

**Features:**
- Comprehensive image validation (format, size, dimensions, aspect ratio)
- Automatic localhost URL to base64 conversion
- Public URL passthrough support
- Smart ImageArtifact handling using `.base64` property
- Model-specific requirements validation

#### **Minimax First-Last Frame-to-Video**

Generate smooth video transitions between two key frames.

**Inputs:**
- **First Frame Image** (ImageArtifact | ImageUrlArtifact): Starting frame
  - Same requirements as Image-to-Video
- **Last Frame Image** (ImageArtifact | ImageUrlArtifact): Ending frame
  - Same requirements as Image-to-Video
- **Prompt** (str): Motion description between frames (up to 2000 characters, optional)
- **Model** (str): Currently only `MiniMax-Hailuo-02` supports this feature
- **Duration** (int): Video length (6 or 10 seconds)
- **Resolution** (str): 512P, 768P, or 1080P
- **Prompt Optimizer** (bool): Optimize prompts (default: False)
- **Fast Pretreatment** (bool): Faster processing (default: False)

**Outputs:**
- **Video URL** (VideoUrlArtifact): Generated video
- **Task ID** (str): API task identifier
- **Provider Response** (dict): Full API response

**Features:**
- Dual image input with independent validation
- Creates smooth transitions between keyframes
- All image handling features from Image-to-Video
- Perfect for creating consistent video sequences

### 🎵 Music Generation

#### **Minimax Music Generation**

Generate original music from text descriptions and lyrics using Minimax's music generation API.

**Inputs:**
- **Prompt** (str): Description of music style, mood, and scenario (10-300 characters)
  - Example: "Pop, melancholic, perfect for a rainy night"
- **Lyrics** (str): Song lyrics with optional structure tags (10-3000 characters)
  - Use `\n` to separate lines
  - Add structure tags: `[Intro]`, `[Verse]`, `[Chorus]`, `[Bridge]`, `[Outro]`
- **Model** (str): Model selection (default: `music-1.5`)
- **Audio Settings** (collapsible group):
  - **Sample Rate** (int): Audio sampling rate (16000, 24000, 32000, 44100 Hz) - default: 44100
  - **Bitrate** (int): Audio bitrate (32000, 64000, 128000, 256000 bps) - default: 128000
  - **Format** (str): Output format (mp3, wav, pcm) - default: mp3

**Outputs:**
- **Audio URL** (AudioUrlArtifact): Generated music file
- **Provider Response** (dict): Full API response

**Features:**
- Text-to-music generation with lyrics
- Configurable audio quality settings
- Song structure tags for better arrangement
- Automatic download and storage to static files
- Synchronous processing (instant response)

**Use Cases:**
- **Custom Songs**: Generate original music with specific lyrics
  - Personalized songs for special occasions
  - Demo tracks for songwriting
  - Background music for videos
- **Music Production**: Create music with specific moods
  - Game soundtracks
  - Podcast intros/outros
  - Video background music
  - Atmospheric music for content

## 🔑 Configuration

### Griptape Nodes Settings (Recommended)

The recommended way to configure your Minimax API key is through Griptape Nodes:

1. Open Griptape Nodes
2. Navigate to **Settings** → **API Keys & Secrets**
3. Set `MINIMAX_API_KEY` to your API key

### Environment Variables (Alternative)

Alternatively, you can set your Minimax API key as an environment variable:

```bash
export MINIMAX_API_KEY="your_api_key_here"
```

**Note**: Using the Settings menu is recommended as it provides a centralized, secure location for managing all your API keys.

## 📋 Requirements

Image requirements for video generation nodes:

- **Formats**: JPG, JPEG, PNG, WebP
- **File Size**: Less than 20MB
- **Dimensions**: 
  - Short edge must be greater than 300px
  - Aspect ratio must be between 2:5 and 5:2 (0.4 to 2.5)
- **Examples**:
  - ✅ Valid: 1920x1080 (16:9 = 1.78)
  - ✅ Valid: 800x600 (4:3 = 1.33)
  - ✅ Valid: 1080x2400 (9:20 = 0.45)
  - ❌ Invalid: 200x400 (short edge too small)
  - ❌ Invalid: 1000x3000 (aspect ratio 0.33, outside range)

## 🎯 Use Cases

### Image Generation
- Create concept art and illustrations
- Generate marketing visuals
- Design prototyping
- Creative exploration
- Social media content

### Video Generation
- **Text-to-Video**: Create videos from descriptions
  - Animated sequences
  - Camera movement demonstrations
  - Scene visualization
- **Image-to-Video**: Animate static images
  - Bring photos to life
  - Create dynamic social media content
  - Product demonstrations
- **First-Last Frame-to-Video**: Create smooth transitions
  - Morphing effects
  - Consistent video sequences
  - Animation interpolation

## 🔧 Advanced Features

### Asynchronous Processing

All video generation nodes use an asynchronous 3-step process:

1. **Submit**: Task is submitted to Minimax API
2. **Poll**: Status checked every 10 seconds (max 10 minutes)
3. **Retrieve**: Final video downloaded and saved

This ensures efficient processing of long-running video generation tasks.

### Localhost URL Handling

The nodes automatically detect and convert localhost URLs to base64:
- Static files served from localhost are downloaded and converted
- Public URLs are passed through directly to the API
- Ensures compatibility with Minimax API requirements

### Image Artifact Support

Smart handling of different image input types:
- **ImageArtifact**: Uses `.base64` and `.mime_type` properties
- **ImageUrlArtifact**: Automatic localhost detection and conversion
- **Fallback**: Manual byte extraction and encoding with PIL

### Error Handling

Comprehensive validation and error reporting:
- Pre-execution parameter validation
- Image format and size checks
- Model compatibility verification
- Detailed error messages with current values
- Full API response logging for debugging

## 🐛 Troubleshooting

### "Invalid params, first_frame_image" Error

This usually means:
- Image is from localhost (automatically converted now)
- Image format doesn't match MIME type
- Image dimensions outside valid range

**Solution**: The nodes now automatically handle these cases with enhanced logging.

### "Task did not complete" Error

Video generation timed out after 10 minutes.

**Solution**: Try with:
- Lower resolution
- Shorter duration
- Simpler prompts

### Image Validation Failures

**Solution**: Ensure your images meet all requirements:
- Format: JPG, JPEG, PNG, or WebP
- Size: Under 20MB
- Short edge: Greater than 300px
- Aspect ratio: Between 2:5 and 5:2

## 📚 API Documentation

For detailed API documentation, visit:
- [Minimax Platform](https://platform.minimax.chat/)
- [API Documentation](https://platform.minimax.chat/document)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

This library is provided for use with Griptape Nodes and the Minimax API.

## 🔗 Links

- [Griptape Nodes](https://github.com/griptape-ai/griptape-nodes)
- [Griptape Nodes Directory](https://github.com/griptape-ai/griptape-nodes-directory)
- [Minimax Platform](https://platform.minimax.chat/)

## 🙏 Acknowledgments

Built for the Griptape Nodes ecosystem, enabling seamless integration with Minimax's powerful AI generation capabilities.

---

**Note**: This is an unofficial community library. For official support, please contact Minimax or Griptape support channels.

