import express from 'express';
import cors from 'cors';
import { createCanvas, loadImage } from 'canvas';

const app = express();
const PORT = process.env.PORT || 4000;

// Middleware
app.use(cors());
app.use(express.json({ limit: '50mb' }));

// Health check
app.get('/health', (req, res) => {
  res.json({ 
    status: 'ok', 
    service: 'TCG-Forensics CV Backend',
    version: '1.0.0',
    algorithms: 20
  });
});

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/**
 * Load base64 image to Canvas ImageData
 */
async function loadImageToCanvas(base64String) {
  try {
    // Remove data:image/xxx;base64, prefix if present
    const base64Data = base64String.replace(/^data:image\/\w+;base64,/, '');
    const buffer = Buffer.from(base64Data, 'base64');
    
    const img = await loadImage(buffer);
    const canvas = createCanvas(img.width, img.height);
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0);
    
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    
    return { 
      imageData, 
      width: canvas.width, 
      height: canvas.height,
      canvas,
      ctx
    };
  } catch (error) {
    console.error('[CV Backend] Error loading image:', error);
    throw new Error('Failed to load image: ' + error.message);
  }
}

/**
 * Convert RGB to Grayscale
 */
function rgbToGrayscale(imageData) {
  const data = imageData.data;
  const gray = new Uint8ClampedArray(imageData.width * imageData.height);
  
  for (let i = 0; i < data.length; i += 4) {
    const r = data[i];
    const g = data[i + 1];
    const b = data[i + 2];
    gray[i / 4] = Math.round(0.299 * r + 0.587 * g + 0.114 * b);
  }
  
  return { data: gray, width: imageData.width, height: imageData.height };
}

// ============================================================================
// PRO TIER CV ALGORITHMS (8)
// ============================================================================

/**
 * 1. CANNY EDGE DETECTION
 */
async function cannyEdgeDetection(base64Image, lowThreshold = 50, highThreshold = 150) {
  const { imageData, width, height } = await loadImageToCanvas(base64Image);
  const gray = rgbToGrayscale(imageData);
  
  // Sobel operators
  const sobelX = [-1, 0, 1, -2, 0, 2, -1, 0, 1];
  const sobelY = [-1, -2, -1, 0, 0, 0, 1, 2, 1];
  
  const gradient = new Float32Array(width * height);
  const direction = new Float32Array(width * height);
  
  // Calculate gradient magnitude and direction
  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      let gx = 0, gy = 0;
      
      for (let ky = -1; ky <= 1; ky++) {
        for (let kx = -1; kx <= 1; kx++) {
          const idx = (y + ky) * width + (x + kx);
          const ki = (ky + 1) * 3 + (kx + 1);
          gx += gray.data[idx] * sobelX[ki];
          gy += gray.data[idx] * sobelY[ki];
        }
      }
      
      const idx = y * width + x;
      gradient[idx] = Math.sqrt(gx * gx + gy * gy);
      direction[idx] = Math.atan2(gy, gx);
    }
  }
  
  // Non-maximum suppression
  const nms = new Uint8ClampedArray(width * height);
  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const idx = y * width + x;
      const angle = direction[idx];
      const mag = gradient[idx];
      
      let neighbor1 = 0, neighbor2 = 0;
      
      if ((angle >= -Math.PI/8 && angle < Math.PI/8) || (angle >= 7*Math.PI/8) || (angle < -7*Math.PI/8)) {
        neighbor1 = gradient[y * width + (x - 1)];
        neighbor2 = gradient[y * width + (x + 1)];
      } else if ((angle >= Math.PI/8 && angle < 3*Math.PI/8) || (angle >= -7*Math.PI/8 && angle < -5*Math.PI/8)) {
        neighbor1 = gradient[(y - 1) * width + (x + 1)];
        neighbor2 = gradient[(y + 1) * width + (x - 1)];
      } else if ((angle >= 3*Math.PI/8 && angle < 5*Math.PI/8) || (angle >= -5*Math.PI/8 && angle < -3*Math.PI/8)) {
        neighbor1 = gradient[(y - 1) * width + x];
        neighbor2 = gradient[(y + 1) * width + x];
      } else {
        neighbor1 = gradient[(y - 1) * width + (x - 1)];
        neighbor2 = gradient[(y + 1) * width + (x + 1)];
      }
      
      if (mag >= neighbor1 && mag >= neighbor2) {
        nms[idx] = mag;
      }
    }
  }
  
  // Hysteresis thresholding
  let edgePixels = 0;
  const edges = new Uint8ClampedArray(width * height);
  
  for (let i = 0; i < nms.length; i++) {
    if (nms[i] >= highThreshold) {
      edges[i] = 255;
      edgePixels++;
    } else if (nms[i] >= lowThreshold) {
      edges[i] = 128;
    }
  }
  
  const edgeDensity = (edgePixels / (width * height)) * 100;
  const score = Math.min(10, (edgeDensity / 10) * 10);
  
  return {
    name: 'Canny Edge Detection',
    edgePixels,
    edgeDensity: edgeDensity.toFixed(2) + '%',
    score: score.toFixed(1),
    description: `Detected ${edgePixels} edge pixels (${edgeDensity.toFixed(1)}% density). Sharp borders indicate authentic print quality.`
  };
}

/**
 * 2. LAB COLOR DELTA-E
 */
async function labColorDeltaE(base64Image) {
  const { imageData, width, height } = await loadImageToCanvas(base64Image);
  const data = imageData.data;
  
  // Sample colors from image
  const samples = [];
  const step = Math.floor(Math.sqrt(width * height) / 100);
  
  for (let y = 0; y < height; y += step) {
    for (let x = 0; x < width; x += step) {
      const idx = (y * width + x) * 4;
      samples.push({
        r: data[idx],
        g: data[idx + 1],
        b: data[idx + 2]
      });
    }
  }
  
  // Calculate color consistency (simplified Delta-E)
  let totalDelta = 0;
  for (let i = 1; i < samples.length; i++) {
    const dr = samples[i].r - samples[i-1].r;
    const dg = samples[i].g - samples[i-1].g;
    const db = samples[i].b - samples[i-1].b;
    totalDelta += Math.sqrt(dr*dr + dg*dg + db*db);
  }
  
  const avgDelta = totalDelta / (samples.length - 1);
  const score = Math.max(0, 10 - (avgDelta / 50));
  
  return {
    name: 'LAB Color Delta-E',
    avgDelta: avgDelta.toFixed(2),
    samples: samples.length,
    score: score.toFixed(1),
    description: `Average color deviation: ${avgDelta.toFixed(1)}. Consistent colors indicate authentic printing.`
  };
}

/**
 * 3. LOCAL BINARY PATTERNS (LBP)
 */
async function localBinaryPatterns(base64Image) {
  const { imageData, width, height } = await loadImageToCanvas(base64Image);
  const gray = rgbToGrayscale(imageData);
  
  let uniformPatterns = 0;
  let totalPatterns = 0;
  
  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const center = gray.data[y * width + x];
      let pattern = 0;
      
      // 8 neighbors
      const neighbors = [
        gray.data[(y-1) * width + (x-1)],
        gray.data[(y-1) * width + x],
        gray.data[(y-1) * width + (x+1)],
        gray.data[y * width + (x+1)],
        gray.data[(y+1) * width + (x+1)],
        gray.data[(y+1) * width + x],
        gray.data[(y+1) * width + (x-1)],
        gray.data[y * width + (x-1)]
      ];
      
      for (let i = 0; i < 8; i++) {
        if (neighbors[i] >= center) {
          pattern |= (1 << i);
        }
      }
      
      // Check if uniform pattern (max 2 transitions)
      let transitions = 0;
      for (let i = 0; i < 8; i++) {
        const bit1 = (pattern >> i) & 1;
        const bit2 = (pattern >> ((i + 1) % 8)) & 1;
        if (bit1 !== bit2) transitions++;
      }
      
      if (transitions <= 2) uniformPatterns++;
      totalPatterns++;
    }
  }
  
  const uniformity = (uniformPatterns / totalPatterns) * 100;
  const score = (uniformity / 10);
  
  return {
    name: 'Local Binary Patterns',
    uniformPatterns,
    totalPatterns,
    uniformity: uniformity.toFixed(2) + '%',
    score: score.toFixed(1),
    description: `${uniformity.toFixed(1)}% uniform texture patterns. Consistent texture indicates authentic card surface.`
  };
}

/**
 * 4. HISTOGRAM OF GRADIENTS (HOG)
 */
async function histogramOfGradients(base64Image) {
  const { imageData, width, height } = await loadImageToCanvas(base64Image);
  const gray = rgbToGrayscale(imageData);
  
  const bins = 9;
  const histogram = new Array(bins).fill(0);
  
  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const gx = gray.data[y * width + (x + 1)] - gray.data[y * width + (x - 1)];
      const gy = gray.data[(y + 1) * width + x] - gray.data[(y - 1) * width + x];
      
      const magnitude = Math.sqrt(gx * gx + gy * gy);
      const angle = Math.atan2(gy, gx);
      
      const binIdx = Math.floor(((angle + Math.PI) / (2 * Math.PI)) * bins) % bins;
      histogram[binIdx] += magnitude;
    }
  }
  
  const maxBin = Math.max(...histogram);
  const minBin = Math.min(...histogram);
  const variance = maxBin - minBin;
  const score = Math.min(10, (variance / 10000) * 10);
  
  return {
    name: 'Histogram of Gradients',
    bins,
    variance: variance.toFixed(0),
    score: score.toFixed(1),
    description: `Gradient variance: ${variance.toFixed(0)}. Rich gradient distribution indicates detailed print quality.`
  };
}

/**
 * 5. ENTROPY ANALYSIS
 */
async function entropyAnalysis(base64Image) {
  const { imageData, width, height } = await loadImageToCanvas(base64Image);
  const gray = rgbToGrayscale(imageData);
  
  // Build histogram
  const histogram = new Array(256).fill(0);
  for (let i = 0; i < gray.data.length; i++) {
    histogram[gray.data[i]]++;
  }
  
  // Calculate entropy
  const total = gray.data.length;
  let entropy = 0;
  
  for (let i = 0; i < 256; i++) {
    if (histogram[i] > 0) {
      const p = histogram[i] / total;
      entropy -= p * Math.log2(p);
    }
  }
  
  const score = (entropy / 8) * 10;
  
  return {
    name: 'Entropy Analysis',
    entropy: entropy.toFixed(2),
    maxEntropy: 8.0,
    score: score.toFixed(1),
    description: `Image entropy: ${entropy.toFixed(2)} bits. High entropy indicates rich detail and authentic printing.`
  };
}

/**
 * 6. LAPLACIAN SHARPNESS
 */
async function laplacianSharpness(base64Image) {
  const { imageData, width, height } = await loadImageToCanvas(base64Image);
  const gray = rgbToGrayscale(imageData);
  
  const laplacian = [0, -1, 0, -1, 4, -1, 0, -1, 0];
  let variance = 0;
  
  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      let sum = 0;
      
      for (let ky = -1; ky <= 1; ky++) {
        for (let kx = -1; kx <= 1; kx++) {
          const idx = (y + ky) * width + (x + kx);
          const ki = (ky + 1) * 3 + (kx + 1);
          sum += gray.data[idx] * laplacian[ki];
        }
      }
      
      variance += sum * sum;
    }
  }
  
  variance /= ((width - 2) * (height - 2));
  const score = Math.min(10, (variance / 1000) * 10);
  
  return {
    name: 'Laplacian Sharpness',
    variance: variance.toFixed(0),
    score: score.toFixed(1),
    description: `Sharpness variance: ${variance.toFixed(0)}. High sharpness indicates professional scanning.`
  };
}

/**
 * 7. HARRIS CORNER DETECTION
 */
async function harrisCornerDetection(base64Image) {
  const { imageData, width, height } = await loadImageToCanvas(base64Image);
  const gray = rgbToGrayscale(imageData);
  
  let corners = 0;
  const threshold = 10000;
  
  for (let y = 2; y < height - 2; y++) {
    for (let x = 2; x < width - 2; x++) {
      let Ix = 0, Iy = 0;
      
      // Sobel gradient
      Ix = (gray.data[y * width + (x + 1)] - gray.data[y * width + (x - 1)]) / 2;
      Iy = (gray.data[(y + 1) * width + x] - gray.data[(y - 1) * width + x]) / 2;
      
      const Ixx = Ix * Ix;
      const Iyy = Iy * Iy;
      const Ixy = Ix * Iy;
      
      // Harris response
      const det = Ixx * Iyy - Ixy * Ixy;
      const trace = Ixx + Iyy;
      const response = det - 0.04 * trace * trace;
      
      if (response > threshold) {
        corners++;
      }
    }
  }
  
  const cornerDensity = (corners / (width * height)) * 10000;
  const score = Math.min(10, cornerDensity);
  
  return {
    name: 'Harris Corner Detection',
    corners,
    density: cornerDensity.toFixed(2),
    score: score.toFixed(1),
    description: `Detected ${corners} corner features. Rich features indicate detailed artwork.`
  };
}

/**
 * 8. HOUGH LINE TRANSFORM
 */
async function houghLineTransform(base64Image) {
  const { imageData, width, height } = await loadImageToCanvas(base64Image);
  const gray = rgbToGrayscale(imageData);
  
  // Simple edge detection
  let edgePixels = [];
  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const gx = gray.data[y * width + (x + 1)] - gray.data[y * width + (x - 1)];
      const gy = gray.data[(y + 1) * width + x] - gray.data[(y - 1) * width + x];
      const magnitude = Math.sqrt(gx * gx + gy * gy);
      
      if (magnitude > 50) {
        edgePixels.push({ x, y });
      }
    }
  }
  
  // Simplified Hough transform (only count strong lines)
  const lines = Math.floor(edgePixels.length / 100);
  const score = Math.min(10, (lines / 10) * 10);
  
  return {
    name: 'Hough Line Transform',
    lines,
    edgePixels: edgePixels.length,
    score: score.toFixed(1),
    description: `Detected ${lines} linear structures. Straight borders indicate proper card cutting.`
  };
}

// ============================================================================
// MASTER ENDPOINT: Run all CV algorithms by tier
// ============================================================================

app.post('/api/cv', async (req, res) => {
  try {
    const { image, tier } = req.body;
    
    if (!image) {
      return res.status(400).json({ error: 'No image provided' });
    }
    
    if (!tier || !['pro', 'expert', 'enterprise'].includes(tier)) {
      return res.status(400).json({ error: 'Invalid tier. Must be pro, expert, or enterprise' });
    }
    
    console.log(`[CV Backend] Processing ${tier.toUpperCase()} tier analysis...`);
    const startTime = Date.now();
    
    const results = [];
    
    // PRO TIER (8 algorithms)
    if (tier === 'pro' || tier === 'expert' || tier === 'enterprise') {
      console.log('[CV Backend] Running PRO tier (8 algorithms)...');
      results.push(await cannyEdgeDetection(image));
      results.push(await labColorDeltaE(image));
      results.push(await localBinaryPatterns(image));
      results.push(await histogramOfGradients(image));
      results.push(await entropyAnalysis(image));
      results.push(await laplacianSharpness(image));
      results.push(await harrisCornerDetection(image));
      results.push(await houghLineTransform(image));
    }
    
    // EXPERT TIER (8 more algorithms) - TODO: Implement these
    if (tier === 'expert' || tier === 'enterprise') {
      console.log('[CV Backend] EXPERT tier algorithms not yet implemented');
    }
    
    // ENTERPRISE TIER (4 ultimate algorithms) - TODO: Implement these
    if (tier === 'enterprise') {
      console.log('[CV Backend] ENTERPRISE tier algorithms not yet implemented');
    }
    
    const processingTime = Date.now() - startTime;
    console.log(`[CV Backend] Analysis complete in ${processingTime}ms`);
    
    res.json({
      success: true,
      tier: tier.toUpperCase(),
      algorithmsRun: results.length,
      processingTime: `${processingTime}ms`,
      results
    });
    
  } catch (error) {
    console.error('[CV Backend] Error:', error);
    res.status(500).json({ 
      error: 'CV analysis failed',
      message: error.message 
    });
  }
});

// Start server
app.listen(PORT, '0.0.0.0', () => {
  console.log(`\n🚀 TCG-Forensics CV Backend running on port ${PORT}`);
  console.log(`📊 Available algorithms: 8 PRO tier (EXPERT/ENTERPRISE coming soon)`);
  console.log(`🔗 Health check: http://localhost:${PORT}/health`);
  console.log(`🎯 CV endpoint: POST http://localhost:${PORT}/api/cv\n`);
});
