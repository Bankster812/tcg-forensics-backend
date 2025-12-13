# Use official Puppeteer image (comes with Chrome)
FROM ghcr.io/puppeteer/puppeteer:24.4.0

# Set working directory
WORKDIR /app

# Set environment variables for Puppeteer
ENV PUPPETEER_SKIP_CHROMIUM_DOWNLOAD=true
ENV PUPPETEER_EXECUTABLE_PATH=/usr/bin/google-chrome-stable

# Copy package files
COPY --chown=pptruser:pptruser package*.json ./

# Install dependencies
RUN npm ci --only=production

# Copy source code
COPY --chown=pptruser:pptruser . .

# Expose port
EXPOSE 4000

# Start the server
CMD ["node", "server.js"]
