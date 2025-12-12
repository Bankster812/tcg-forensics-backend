# TCG-Forensics CV Backend

Computer Vision analysis backend for TCG-Forensics.

## Features

- **8 PRO Tier Algorithms**: Canny Edge Detection, LAB Color Delta-E, Local Binary Patterns, Histogram of Gradients, Entropy Analysis, Laplacian Sharpness, Harris Corner Detection, Hough Line Transform
- **Node.js Canvas**: Server-side image processing with `canvas` library
- **Unlimited CPU**: No stack overflow issues - handles high-resolution images
- **RESTful API**: Simple POST endpoint for analysis

## Local Development

```bash
npm install
npm start
```

Server runs on `http://localhost:4000`

## API Endpoints

### Health Check
```bash
GET /health
```

### CV Analysis
```bash
POST /api/cv
Content-Type: application/json

{
  "image": "data:image/png;base64,...",
  "tier": "pro"
}
```

**Response:**
```json
{
  "success": true,
  "tier": "PRO",
  "algorithmsRun": 8,
  "processingTime": "1234ms",
  "results": [
    {
      "name": "Canny Edge Detection",
      "score": "8.5",
      "description": "..."
    },
    ...
  ]
}
```

## Railway Deployment

### Option 1: Web UI (Easiest)

1. Go to [railway.app](https://railway.app)
2. Click "Start a New Project"
3. Select "Deploy from GitHub repo"
4. Connect this repository
5. Railway auto-detects Node.js and deploys!

### Option 2: CLI

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login
railway login

# Init project
railway init

# Deploy
railway up
```

### Environment Variables (if needed)

- `PORT`: Railway sets this automatically
- `NODE_ENV`: production

## Tech Stack

- **Express.js**: Web framework
- **canvas**: Node.js canvas implementation for image processing
- **cors**: Enable cross-origin requests from frontend

## Notes

- Bundle size: ~15KB (compressed)
- Processing time: ~1-2s per image (PRO tier)
- Memory usage: ~60MB per request
- Cold start: ~2-3s on Railway free tier
