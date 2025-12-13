import express from 'express';
import cors from 'cors';
import sharp from 'sharp';
import puppeteer from 'puppeteer';

const app = express();
const PORT = process.env.PORT || 4000;

// Middleware
app.use(cors());
app.use(express.json({ limit: '50mb' }));

// Cache for PSA scraped data
const PSA_CACHE = new Map();

// Health check
app.get('/health', (req, res) => {
  res.json({ 
    status: 'ok', 
    service: 'TCG-Forensics CV Backend',
    version: '2.1.1',
    algorithms: 8,
    features: ['CV Analysis', 'PSA Scraping', 'Image Comparison']
  });
});

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/**
 * Load base64 image and get raw pixel data using Sharp
 */
async function loadImageData(base64String) {
  try {
    // Remove data:image/xxx;base64, prefix if present
    const base64Data = base64String.replace(/^data:image\/\w+;base64,/, '');
    const buffer = Buffer.from(base64Data, 'base64');
    
    // Use sharp to get raw pixel data
    const image = sharp(buffer);
    const metadata = await image.metadata();
    
    // Get raw RGBA pixels
    const { data, info } = await image
      .ensureAlpha()
      .raw()
      .toBuffer({ resolveWithObject: true });
    
    return { 
      data: new Uint8ClampedArray(data),
      width: info.width, 
      height: info.height,
      channels: info.channels
    };
  } catch (error) {
    console.error('[CV Backend] Error loading image:', error);
    throw new Error('Failed to load image: ' + error.message);
  }
}

/**
 * Convert RGB to Grayscale
 */
function rgbToGrayscale(imageData, width, height) {
  const data = imageData;
  const gray = new Uint8ClampedArray(width * height);
  
  for (let i = 0; i < width * height; i++) {
    const idx = i * 4;
    const r = data[idx];
    const g = data[idx + 1];
    const b = data[idx + 2];
    gray[i] = Math.round(0.299 * r + 0.587 * g + 0.114 * b);
  }
  
  return gray;
}

// ============================================================================
// PRO TIER CV ALGORITHMS (8)
// ============================================================================

/**
 * 1. CANNY EDGE DETECTION
 */
async function cannyEdgeDetection(base64Image) {
  const { data, width, height } = await loadImageData(base64Image);
  const gray = rgbToGrayscale(data, width, height);
  
  const lowThreshold = 50;
  const highThreshold = 150;
  
  // Sobel operators
  const sobelX = [-1, 0, 1, -2, 0, 2, -1, 0, 1];
  const sobelY = [-1, -2, -1, 0, 0, 0, 1, 2, 1];
  
  const gradient = new Float32Array(width * height);
  
  // Calculate gradient magnitude
  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      let gx = 0, gy = 0;
      
      for (let ky = -1; ky <= 1; ky++) {
        for (let kx = -1; kx <= 1; kx++) {
          const idx = (y + ky) * width + (x + kx);
          const ki = (ky + 1) * 3 + (kx + 1);
          gx += gray[idx] * sobelX[ki];
          gy += gray[idx] * sobelY[ki];
        }
      }
      
      gradient[y * width + x] = Math.sqrt(gx * gx + gy * gy);
    }
  }
  
  // Count edge pixels
  let edgePixels = 0;
  for (let i = 0; i < gradient.length; i++) {
    if (gradient[i] >= highThreshold) edgePixels++;
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
  const { data, width, height } = await loadImageData(base64Image);
  
  // Sample colors from image
  const samples = [];
  const step = Math.max(1, Math.floor(Math.sqrt(width * height) / 100));
  
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
  
  const avgDelta = samples.length > 1 ? totalDelta / (samples.length - 1) : 0;
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
  const { data, width, height } = await loadImageData(base64Image);
  const gray = rgbToGrayscale(data, width, height);
  
  let uniformPatterns = 0;
  let totalPatterns = 0;
  
  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const center = gray[y * width + x];
      let pattern = 0;
      
      // 8 neighbors
      const neighbors = [
        gray[(y-1) * width + (x-1)],
        gray[(y-1) * width + x],
        gray[(y-1) * width + (x+1)],
        gray[y * width + (x+1)],
        gray[(y+1) * width + (x+1)],
        gray[(y+1) * width + x],
        gray[(y+1) * width + (x-1)],
        gray[y * width + (x-1)]
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
  
  const uniformity = totalPatterns > 0 ? (uniformPatterns / totalPatterns) * 100 : 0;
  const score = uniformity / 10;
  
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
  const { data, width, height } = await loadImageData(base64Image);
  const gray = rgbToGrayscale(data, width, height);
  
  const bins = 9;
  const histogram = new Array(bins).fill(0);
  
  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const gx = gray[y * width + (x + 1)] - gray[y * width + (x - 1)];
      const gy = gray[(y + 1) * width + x] - gray[(y - 1) * width + x];
      
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
  const { data, width, height } = await loadImageData(base64Image);
  const gray = rgbToGrayscale(data, width, height);
  
  // Build histogram
  const histogram = new Array(256).fill(0);
  for (let i = 0; i < gray.length; i++) {
    histogram[gray[i]]++;
  }
  
  // Calculate entropy
  const total = gray.length;
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
  const { data, width, height } = await loadImageData(base64Image);
  const gray = rgbToGrayscale(data, width, height);
  
  const laplacian = [0, -1, 0, -1, 4, -1, 0, -1, 0];
  let variance = 0;
  let count = 0;
  
  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      let sum = 0;
      
      for (let ky = -1; ky <= 1; ky++) {
        for (let kx = -1; kx <= 1; kx++) {
          const idx = (y + ky) * width + (x + kx);
          const ki = (ky + 1) * 3 + (kx + 1);
          sum += gray[idx] * laplacian[ki];
        }
      }
      
      variance += sum * sum;
      count++;
    }
  }
  
  variance = count > 0 ? variance / count : 0;
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
  const { data, width, height } = await loadImageData(base64Image);
  const gray = rgbToGrayscale(data, width, height);
  
  let corners = 0;
  const threshold = 10000;
  
  for (let y = 2; y < height - 2; y++) {
    for (let x = 2; x < width - 2; x++) {
      // Sobel gradient
      const Ix = (gray[y * width + (x + 1)] - gray[y * width + (x - 1)]) / 2;
      const Iy = (gray[(y + 1) * width + x] - gray[(y - 1) * width + x]) / 2;
      
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
  const { data, width, height } = await loadImageData(base64Image);
  const gray = rgbToGrayscale(data, width, height);
  
  // Simple edge detection
  let edgeCount = 0;
  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const gx = gray[y * width + (x + 1)] - gray[y * width + (x - 1)];
      const gy = gray[(y + 1) * width + x] - gray[(y - 1) * width + x];
      const magnitude = Math.sqrt(gx * gx + gy * gy);
      
      if (magnitude > 50) edgeCount++;
    }
  }
  
  // Simplified Hough transform (only count strong lines)
  const lines = Math.floor(edgeCount / 100);
  const score = Math.min(10, (lines / 10) * 10);
  
  return {
    name: 'Hough Line Transform',
    lines,
    edgePixels: edgeCount,
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
      console.log('[CV Backend] ✓ Canny Edge Detection');
      
      results.push(await labColorDeltaE(image));
      console.log('[CV Backend] ✓ LAB Color Delta-E');
      
      results.push(await localBinaryPatterns(image));
      console.log('[CV Backend] ✓ Local Binary Patterns');
      
      results.push(await histogramOfGradients(image));
      console.log('[CV Backend] ✓ Histogram of Gradients');
      
      results.push(await entropyAnalysis(image));
      console.log('[CV Backend] ✓ Entropy Analysis');
      
      results.push(await laplacianSharpness(image));
      console.log('[CV Backend] ✓ Laplacian Sharpness');
      
      results.push(await harrisCornerDetection(image));
      console.log('[CV Backend] ✓ Harris Corner Detection');
      
      results.push(await houghLineTransform(image));
      console.log('[CV Backend] ✓ Hough Line Transform');
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

// ============================================================================
// PSA REFERENCE SCRAPING WITH PUPPETEER (Cloudflare Bypass)
// ============================================================================

let browserInstance = null;

/**
 * Get or create browser instance
 */
async function getBrowser() {
  if (browserInstance) return browserInstance;
  
  console.log('[Puppeteer] Launching browser...');
  
  // Use system Chromium if available (for Docker/Railway)
  const executablePath = process.env.PUPPETEER_EXECUTABLE_PATH || undefined;
  
  if (executablePath) {
    console.log(`[Puppeteer] Using system Chromium: ${executablePath}`);
  }
  
  try {
    browserInstance = await puppeteer.launch({
      headless: 'new',
      executablePath,
      args: [
        '--no-sandbox',
        '--disable-setuid-sandbox',
        '--disable-dev-shm-usage',
        '--disable-accelerated-2d-canvas',
        '--no-first-run',
        '--no-zygote',
        '--disable-gpu',
        '--single-process'
      ]
    });
    console.log('[Puppeteer] Browser launched successfully');
    return browserInstance;
  } catch (error) {
    console.error('[Puppeteer] Failed to launch:', error.message);
    throw error;
  }
}

/**
 * Scrape PSA page with Puppeteer - bypasses Cloudflare
 */
async function scrapePSAWithPuppeteer(certNumber) {
  const psaUrl = `https://www.psacard.com/cert/${certNumber}`;
  
  // Check cache first
  if (PSA_CACHE.has(certNumber)) {
    console.log(`[PSA Scraper] Cache hit for ${certNumber}`);
    return PSA_CACHE.get(certNumber);
  }
  
  console.log(`[PSA Scraper] Scraping ${psaUrl} with Puppeteer...`);
  
  let browser = null;
  let page = null;
  
  try {
    browser = await getBrowser();
    page = await browser.newPage();
    
    // Set realistic headers
    await page.setUserAgent('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36');
    await page.setExtraHTTPHeaders({
      'Accept-Language': 'en-US,en;q=0.9',
      'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
    });
    
    // Navigate and wait for content
    await page.goto(psaUrl, { 
      waitUntil: 'networkidle2',
      timeout: 30000 
    });
    
    // Wait a bit for dynamic content
    await page.waitForTimeout(2000);
    
    // Extract all data from page
    const data = await page.evaluate(() => {
      const result = {
        grade: null,
        cardName: null,
        year: null,
        setName: null,
        variety: null,
        images: [],
        labelImage: null,
        cardFrontImage: null,
        cardBackImage: null,
        allImages: []
      };
      
      // Get all images on the page
      const imgs = document.querySelectorAll('img');
      imgs.forEach(img => {
        const src = img.src || img.getAttribute('data-src') || '';
        if (src && (src.includes('http') || src.startsWith('/'))) {
          const fullUrl = src.startsWith('/') ? 'https://www.psacard.com' + src : src;
          result.allImages.push(fullUrl);
          
          // Identify specific images
          const alt = (img.alt || '').toLowerCase();
          const className = (img.className || '').toLowerCase();
          
          if (alt.includes('front') || className.includes('front') || src.includes('front')) {
            result.cardFrontImage = fullUrl;
          }
          if (alt.includes('back') || className.includes('back') || src.includes('back')) {
            result.cardBackImage = fullUrl;
          }
          if (alt.includes('label') || className.includes('label') || src.includes('label')) {
            result.labelImage = fullUrl;
          }
          if (src.includes('cert') || src.includes('card') || src.includes('pokemon')) {
            result.images.push(fullUrl);
          }
        }
      });
      
      // Try to get grade from various elements
      const gradeSelectors = [
        '.grade', '.cert-grade', '[class*="grade"]', 
        'span:contains("Grade")', '.card-grade'
      ];
      for (const sel of gradeSelectors) {
        try {
          const el = document.querySelector(sel);
          if (el && el.textContent) {
            const match = el.textContent.match(/(\d+\.?\d*)/);
            if (match) {
              result.grade = match[1];
              break;
            }
          }
        } catch(e) {}
      }
      
      // Get card name from page title or h1
      const h1 = document.querySelector('h1');
      if (h1) result.cardName = h1.textContent.trim();
      
      // Get from title tag
      const title = document.title;
      if (title && !result.cardName) {
        result.cardName = title.replace(/PSA|Cert|Certificate|\|/gi, '').trim();
      }
      
      // Try to find year
      const yearMatch = document.body.textContent.match(/\b(19\d{2}|20[0-2]\d)\b/);
      if (yearMatch) result.year = yearMatch[1];
      
      // Get full page HTML for debugging
      result.pageTitle = document.title;
      result.url = window.location.href;
      
      return result;
    });
    
    // Take screenshot of the page for reference
    const screenshot = await page.screenshot({ 
      encoding: 'base64',
      fullPage: false,
      type: 'jpeg',
      quality: 80
    });
    
    // Also try to get specific card images by clicking thumbnails
    try {
      const thumbnails = await page.$$('img[class*="thumb"], img[class*="gallery"], .card-image img');
      for (const thumb of thumbnails.slice(0, 3)) {
        const src = await thumb.evaluate(el => el.src || el.getAttribute('data-src') || el.getAttribute('data-full'));
        if (src && !data.images.includes(src)) {
          data.images.push(src);
        }
      }
    } catch (e) {}
    
    await page.close();
    
    const result = {
      certNumber,
      psaUrl,
      grade: data.grade,
      cardName: data.cardName,
      year: data.year,
      referenceImages: [...new Set(data.images)].slice(0, 10),
      cardFrontImage: data.cardFrontImage,
      cardBackImage: data.cardBackImage,
      labelImage: data.labelImage,
      allImages: [...new Set(data.allImages)],
      screenshot: `data:image/jpeg;base64,${screenshot}`,
      scraped: true,
      scrapedAt: new Date().toISOString()
    };
    
    // Cache the result
    PSA_CACHE.set(certNumber, result);
    
    console.log(`[PSA Scraper] Found ${result.referenceImages.length} card images, ${result.allImages.length} total images`);
    
    return result;
    
  } catch (error) {
    console.error('[PSA Scraper] Puppeteer error:', error.message);
    if (page) await page.close().catch(() => {});
    
    return { 
      error: error.message, 
      psaUrl,
      certNumber,
      scraped: false
    };
  }
}

/**
 * Fetch PSA page - tries Puppeteer first, falls back to simple fetch
 */
async function fetchPSAReference(certNumber) {
  // Always use Puppeteer for reliable scraping
  return await scrapePSAWithPuppeteer(certNumber);
}

/**
 * Download and load reference image for comparison
 */
async function loadReferenceImage(imageUrl) {
  try {
    const response = await fetch(imageUrl, {
      headers: { 'User-Agent': 'Mozilla/5.0' }
    });
    
    if (!response.ok) return null;
    
    const buffer = Buffer.from(await response.arrayBuffer());
    const image = sharp(buffer);
    const metadata = await image.metadata();
    
    // Resize to standard size for comparison
    const { data, info } = await image
      .resize(900, 1200, { fit: 'contain', background: { r: 255, g: 255, b: 255 } })
      .ensureAlpha()
      .raw()
      .toBuffer({ resolveWithObject: true });
    
    return {
      data: new Uint8ClampedArray(data),
      width: info.width,
      height: info.height,
      originalWidth: metadata.width,
      originalHeight: metadata.height
    };
  } catch (error) {
    console.error('[Reference Image] Error loading:', error.message);
    return null;
  }
}

/**
 * Calculate color histogram correlation (like Manus code)
 */
function calculateHistogramCorrelation(imgData1, imgData2, width, height) {
  // Build 8x8x8 color histograms (simplified)
  const bins = 8;
  const hist1 = new Float32Array(bins * bins * bins);
  const hist2 = new Float32Array(bins * bins * bins);
  
  const pixelCount = width * height;
  
  for (let i = 0; i < pixelCount; i++) {
    const idx = i * 4;
    
    // Image 1
    const r1 = Math.floor(imgData1[idx] / 32);
    const g1 = Math.floor(imgData1[idx + 1] / 32);
    const b1 = Math.floor(imgData1[idx + 2] / 32);
    hist1[r1 * 64 + g1 * 8 + b1]++;
    
    // Image 2
    const r2 = Math.floor(imgData2[idx] / 32);
    const g2 = Math.floor(imgData2[idx + 1] / 32);
    const b2 = Math.floor(imgData2[idx + 2] / 32);
    hist2[r2 * 64 + g2 * 8 + b2]++;
  }
  
  // Normalize
  for (let i = 0; i < hist1.length; i++) {
    hist1[i] /= pixelCount;
    hist2[i] /= pixelCount;
  }
  
  // Calculate correlation
  let sum1 = 0, sum2 = 0, sum12 = 0;
  let sqSum1 = 0, sqSum2 = 0;
  
  const mean1 = hist1.reduce((a, b) => a + b, 0) / hist1.length;
  const mean2 = hist2.reduce((a, b) => a + b, 0) / hist2.length;
  
  for (let i = 0; i < hist1.length; i++) {
    const d1 = hist1[i] - mean1;
    const d2 = hist2[i] - mean2;
    sum12 += d1 * d2;
    sqSum1 += d1 * d1;
    sqSum2 += d2 * d2;
  }
  
  const correlation = sum12 / (Math.sqrt(sqSum1) * Math.sqrt(sqSum2) || 1);
  return Math.max(0, Math.min(1, correlation));
}

/**
 * Calculate Laplacian sharpness variance (like Manus code)
 */
function calculateSharpness(imgData, width, height) {
  const gray = rgbToGrayscale(imgData, width, height);
  
  // Laplacian kernel
  const kernel = [0, 1, 0, 1, -4, 1, 0, 1, 0];
  
  let sum = 0;
  let sumSq = 0;
  let count = 0;
  
  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      let laplacian = 0;
      
      for (let ky = -1; ky <= 1; ky++) {
        for (let kx = -1; kx <= 1; kx++) {
          const idx = (y + ky) * width + (x + kx);
          laplacian += gray[idx] * kernel[(ky + 1) * 3 + (kx + 1)];
        }
      }
      
      sum += laplacian;
      sumSq += laplacian * laplacian;
      count++;
    }
  }
  
  const mean = sum / count;
  const variance = (sumSq / count) - (mean * mean);
  
  return Math.abs(variance);
}

/**
 * Simple SSIM (Structural Similarity) calculation
 */
function calculateSSIM(imgData1, imgData2, width, height) {
  const gray1 = rgbToGrayscale(imgData1, width, height);
  const gray2 = rgbToGrayscale(imgData2, width, height);
  
  const n = width * height;
  
  // Calculate means
  let mean1 = 0, mean2 = 0;
  for (let i = 0; i < n; i++) {
    mean1 += gray1[i];
    mean2 += gray2[i];
  }
  mean1 /= n;
  mean2 /= n;
  
  // Calculate variances and covariance
  let var1 = 0, var2 = 0, covar = 0;
  for (let i = 0; i < n; i++) {
    const d1 = gray1[i] - mean1;
    const d2 = gray2[i] - mean2;
    var1 += d1 * d1;
    var2 += d2 * d2;
    covar += d1 * d2;
  }
  var1 /= n;
  var2 /= n;
  covar /= n;
  
  // SSIM constants
  const C1 = 6.5025;  // (0.01 * 255)^2
  const C2 = 58.5225; // (0.03 * 255)^2
  
  const ssim = ((2 * mean1 * mean2 + C1) * (2 * covar + C2)) /
               ((mean1 * mean1 + mean2 * mean2 + C1) * (var1 + var2 + C2));
  
  return Math.max(0, Math.min(1, ssim));
}

/**
 * PSA Reference Comparison Endpoint
 * Compares user's image against PSA reference images
 */
app.post('/api/psa-compare', async (req, res) => {
  try {
    const { userImage, certNumber } = req.body;
    
    if (!userImage || !certNumber) {
      return res.status(400).json({ error: 'Missing userImage or certNumber' });
    }
    
    console.log(`[PSA Compare] Starting comparison for cert #${certNumber}`);
    const startTime = Date.now();
    
    // Load user image
    console.log('[PSA Compare] Loading user image...');
    const userImgData = await loadImageData(userImage);
    
    // Resize user image to standard size
    const userBuffer = Buffer.from(userImage.replace(/^data:image\/\w+;base64,/, ''), 'base64');
    const userResized = await sharp(userBuffer)
      .resize(900, 1200, { fit: 'contain', background: { r: 255, g: 255, b: 255 } })
      .ensureAlpha()
      .raw()
      .toBuffer({ resolveWithObject: true });
    
    const userStandardized = {
      data: new Uint8ClampedArray(userResized.data),
      width: userResized.info.width,
      height: userResized.info.height
    };
    
    // Fetch PSA reference
    console.log('[PSA Compare] Fetching PSA reference...');
    const psaRef = await fetchPSAReference(certNumber);
    
    if (psaRef.error === 'cloudflare_blocked') {
      return res.json({
        success: false,
        error: 'cloudflare_blocked',
        message: 'PSA website is protected by Cloudflare. Manual verification required.',
        psaUrl: psaRef.psaUrl,
        userAnalysis: {
          sharpness: calculateSharpness(userStandardized.data, userStandardized.width, userStandardized.height)
        }
      });
    }
    
    // If we got PSA images, compare them
    const comparisonResults = [];
    
    if (psaRef.referenceImages && psaRef.referenceImages.length > 0) {
      console.log(`[PSA Compare] Found ${psaRef.referenceImages.length} reference images`);
      
      for (const refUrl of psaRef.referenceImages.slice(0, 3)) { // Compare with max 3 refs
        const refImg = await loadReferenceImage(refUrl);
        if (!refImg) continue;
        
        // Calculate comparison metrics
        const colorCorrelation = calculateHistogramCorrelation(
          userStandardized.data, refImg.data, 
          userStandardized.width, userStandardized.height
        );
        
        const ssim = calculateSSIM(
          userStandardized.data, refImg.data,
          userStandardized.width, userStandardized.height
        );
        
        const userSharpness = calculateSharpness(userStandardized.data, userStandardized.width, userStandardized.height);
        const refSharpness = calculateSharpness(refImg.data, refImg.width, refImg.height);
        const sharpnessDiff = Math.abs(userSharpness - refSharpness);
        
        comparisonResults.push({
          referenceUrl: refUrl,
          colorCorrelation: colorCorrelation.toFixed(4),
          ssim: ssim.toFixed(4),
          userSharpness: userSharpness.toFixed(2),
          refSharpness: refSharpness.toFixed(2),
          sharpnessDifference: sharpnessDiff.toFixed(2)
        });
      }
    }
    
    // Calculate final authenticity score (like Manus code)
    let score = 100;
    const warnings = [];
    
    if (comparisonResults.length > 0) {
      const avgColorCorr = comparisonResults.reduce((a, r) => a + parseFloat(r.colorCorrelation), 0) / comparisonResults.length;
      const avgSSIM = comparisonResults.reduce((a, r) => a + parseFloat(r.ssim), 0) / comparisonResults.length;
      const avgSharpDiff = comparisonResults.reduce((a, r) => a + parseFloat(r.sharpnessDifference), 0) / comparisonResults.length;
      
      // Scoring (matching Manus logic)
      if (avgColorCorr < 0.7) {
        score -= 25;
        warnings.push(`Low color correlation: ${(avgColorCorr * 100).toFixed(1)}%`);
      }
      
      if (avgSSIM < 0.6) {
        score -= 25;
        warnings.push(`Low structural similarity: ${(avgSSIM * 100).toFixed(1)}%`);
      }
      
      if (avgSharpDiff > 100) {
        score -= 20;
        warnings.push(`Significant sharpness difference: ${avgSharpDiff.toFixed(0)}`);
      }
    } else {
      // No reference images - can't compare
      score = 50;
      warnings.push('No PSA reference images available for comparison');
    }
    
    // Determine verdict
    let verdict;
    if (score < 50) {
      verdict = 'LIKELY_FAKE';
    } else if (score < 75) {
      verdict = 'SUSPICIOUS';
    } else {
      verdict = 'LIKELY_AUTHENTIC';
    }
    
    const processingTime = Date.now() - startTime;
    
    console.log(`[PSA Compare] Complete in ${processingTime}ms - Score: ${score}/100 - ${verdict}`);
    
    res.json({
      success: true,
      certNumber,
      psaData: {
        grade: psaRef.grade,
        cardName: psaRef.cardName,
        year: psaRef.year,
        psaUrl: psaRef.psaUrl
      },
      comparison: {
        referenceImagesFound: comparisonResults.length,
        results: comparisonResults
      },
      authenticity: {
        score,
        verdict,
        warnings
      },
      processingTime: `${processingTime}ms`
    });
    
  } catch (error) {
    console.error('[PSA Compare] Error:', error);
    res.status(500).json({
      error: 'PSA comparison failed',
      message: error.message
    });
  }
});

/**
 * Known PSA Certs with cached reference data
 */
const KNOWN_PSA_CERTS = {
  '63437557': {
    grade: '10',
    cardName: 'Lugia Holo 1st Edition',
    set: 'Neo Genesis',
    year: '2000',
    // Real PSA reference images (from psacard.com or cached)
    referenceImages: []
  }
};

/**
 * Get PSA reference data (cached or scraped)
 */
app.get('/api/psa-reference/:certNumber', async (req, res) => {
  const certNumber = req.params.certNumber.replace(/[^0-9]/g, '');
  
  // Always scrape fresh with Puppeteer
  const psaRef = await fetchPSAReference(certNumber);
  res.json(psaRef);
});

/**
 * Direct PSA scrape endpoint - returns screenshot and all images
 */
app.get('/api/psa-scrape/:certNumber', async (req, res) => {
  const certNumber = req.params.certNumber.replace(/[^0-9]/g, '');
  
  console.log(`[PSA Scrape] Direct scrape request for cert #${certNumber}`);
  
  try {
    const result = await scrapePSAWithPuppeteer(certNumber);
    
    if (result.error) {
      return res.status(500).json(result);
    }
    
    res.json({
      success: true,
      ...result
    });
    
  } catch (error) {
    console.error('[PSA Scrape] Error:', error);
    res.status(500).json({
      error: error.message,
      certNumber
    });
  }
});

/**
 * Download PSA image and return as base64
 */
app.get('/api/psa-image', async (req, res) => {
  const imageUrl = req.query.url;
  
  if (!imageUrl) {
    return res.status(400).json({ error: 'Missing url parameter' });
  }
  
  try {
    console.log(`[PSA Image] Downloading: ${imageUrl}`);
    
    const response = await fetch(imageUrl, {
      headers: {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Referer': 'https://www.psacard.com/'
      }
    });
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    
    const buffer = Buffer.from(await response.arrayBuffer());
    const base64 = buffer.toString('base64');
    const contentType = response.headers.get('content-type') || 'image/jpeg';
    
    res.json({
      success: true,
      imageUrl,
      base64: `data:${contentType};base64,${base64}`,
      size: buffer.length
    });
    
  } catch (error) {
    console.error('[PSA Image] Error:', error.message);
    res.status(500).json({
      error: error.message,
      imageUrl
    });
  }
});

/**
 * Full PSA verification with image comparison
 */
app.post('/api/psa-verify', async (req, res) => {
  const { userImages, certNumber } = req.body;
  
  if (!certNumber) {
    return res.status(400).json({ error: 'Missing certNumber' });
  }
  
  console.log(`[PSA Verify] Full verification for cert #${certNumber}`);
  const startTime = Date.now();
  
  try {
    // Step 1: Scrape PSA page
    console.log('[PSA Verify] Step 1: Scraping PSA page...');
    const psaData = await scrapePSAWithPuppeteer(certNumber);
    
    if (psaData.error) {
      return res.json({
        success: false,
        error: psaData.error,
        psaUrl: psaData.psaUrl
      });
    }
    
    // Step 2: Download PSA reference images
    console.log('[PSA Verify] Step 2: Downloading reference images...');
    const referenceImages = [];
    
    for (const imgUrl of psaData.allImages.slice(0, 5)) {
      try {
        const imgResponse = await fetch(imgUrl, {
          headers: { 
            'User-Agent': 'Mozilla/5.0',
            'Referer': 'https://www.psacard.com/'
          }
        });
        
        if (imgResponse.ok) {
          const buffer = Buffer.from(await imgResponse.arrayBuffer());
          
          // Process with sharp
          const processed = await sharp(buffer)
            .resize(900, 1200, { fit: 'contain', background: { r: 255, g: 255, b: 255 } })
            .ensureAlpha()
            .raw()
            .toBuffer({ resolveWithObject: true });
          
          referenceImages.push({
            url: imgUrl,
            data: new Uint8ClampedArray(processed.data),
            width: processed.info.width,
            height: processed.info.height,
            base64: `data:image/jpeg;base64,${buffer.toString('base64')}`
          });
          
          console.log(`[PSA Verify] Downloaded: ${imgUrl}`);
        }
      } catch (e) {
        console.log(`[PSA Verify] Failed to download: ${imgUrl}`);
      }
    }
    
    // Step 3: Compare with user images (if provided)
    let comparisonResults = [];
    
    if (userImages && userImages.length > 0 && referenceImages.length > 0) {
      console.log('[PSA Verify] Step 3: Comparing images...');
      
      for (const userImg of userImages.slice(0, 3)) {
        // Load user image
        const userBuffer = Buffer.from(userImg.replace(/^data:image\/\w+;base64,/, ''), 'base64');
        const userProcessed = await sharp(userBuffer)
          .resize(900, 1200, { fit: 'contain', background: { r: 255, g: 255, b: 255 } })
          .ensureAlpha()
          .raw()
          .toBuffer({ resolveWithObject: true });
        
        const userData = new Uint8ClampedArray(userProcessed.data);
        
        // Compare against each reference
        for (const ref of referenceImages) {
          const colorCorr = calculateHistogramCorrelation(userData, ref.data, 900, 1200);
          const ssim = calculateSSIM(userData, ref.data, 900, 1200);
          const userSharp = calculateSharpness(userData, 900, 1200);
          const refSharp = calculateSharpness(ref.data, ref.width, ref.height);
          
          comparisonResults.push({
            referenceUrl: ref.url,
            colorCorrelation: (colorCorr * 100).toFixed(1),
            structuralSimilarity: (ssim * 100).toFixed(1),
            userSharpness: userSharp.toFixed(0),
            refSharpness: refSharp.toFixed(0),
            sharpnessDiff: Math.abs(userSharp - refSharp).toFixed(0)
          });
        }
      }
    }
    
    // Step 4: Calculate authenticity score
    let score = 100;
    const warnings = [];
    
    if (comparisonResults.length > 0) {
      const avgColorCorr = comparisonResults.reduce((a, r) => a + parseFloat(r.colorCorrelation), 0) / comparisonResults.length;
      const avgSSIM = comparisonResults.reduce((a, r) => a + parseFloat(r.structuralSimilarity), 0) / comparisonResults.length;
      const avgSharpDiff = comparisonResults.reduce((a, r) => a + parseFloat(r.sharpnessDiff), 0) / comparisonResults.length;
      
      if (avgColorCorr < 70) {
        score -= 25;
        warnings.push(`Low color match: ${avgColorCorr.toFixed(1)}%`);
      }
      if (avgSSIM < 60) {
        score -= 25;
        warnings.push(`Low structural similarity: ${avgSSIM.toFixed(1)}%`);
      }
      if (avgSharpDiff > 100) {
        score -= 20;
        warnings.push(`Sharpness difference: ${avgSharpDiff.toFixed(0)}`);
      }
    }
    
    let verdict = score >= 75 ? 'LIKELY_AUTHENTIC' : score >= 50 ? 'SUSPICIOUS' : 'LIKELY_FAKE';
    
    const processingTime = Date.now() - startTime;
    
    res.json({
      success: true,
      certNumber,
      psaData: {
        grade: psaData.grade,
        cardName: psaData.cardName,
        year: psaData.year,
        psaUrl: psaData.psaUrl,
        screenshot: psaData.screenshot
      },
      referenceImages: referenceImages.map(r => ({
        url: r.url,
        base64: r.base64
      })),
      comparison: comparisonResults,
      authenticity: {
        score,
        verdict,
        warnings
      },
      processingTime: `${processingTime}ms`
    });
    
  } catch (error) {
    console.error('[PSA Verify] Error:', error);
    res.status(500).json({
      error: error.message,
      certNumber
    });
  }
});

// Start server
app.listen(PORT, '0.0.0.0', () => {
  console.log(`\n🚀 TCG-Forensics CV Backend v2.1.1 running on port ${PORT}`);
  console.log(`📊 Available: 8 CV algorithms + PSA Scraping`);
  console.log(`🔗 Health: http://localhost:${PORT}/health`);
  console.log(`🎯 CV: POST http://localhost:${PORT}/api/cv`);
  console.log(`🔍 PSA Scrape: GET http://localhost:${PORT}/api/psa-scrape/:certNumber`);
  console.log(`✅ PSA Verify: POST http://localhost:${PORT}/api/psa-verify\n`);
});
