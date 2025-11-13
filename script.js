// Wait for DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function() {
  initializeApp();
});

function initializeApp() {
  // Background parallax effect
  let targetX = 50;
  let targetY = 50;
  let currentX = 50;
  let currentY = 50;

  document.addEventListener("mousemove", (event) => {
    targetX = 50 - ((event.clientX / window.innerWidth - 0.5) * 10);
    targetY = 50 - ((event.clientY / window.innerHeight - 0.5) * 10);
  });

  function animateBackground() {
    currentX += (targetX - currentX) * 0.1;
    currentY += (targetY - currentY) * 0.1;
    document.body.style.backgroundPosition = `${currentX}% ${currentY}%`;
    requestAnimationFrame(animateBackground);
  }

  animateBackground();

  // Developer faces physics simulation
  const developers = [
    { name: "Daniel Tran", image: "daniel.png" },
    { name: "Khang Luong", image: "khang.png" },
    { name: "Mary Tran", image: "mary.png" },
    { name: "Cat Dinh", image: "cat.png" },
  ];

  let faces = [];
  let animationId = null;

  class DeveloperFace {
    constructor(dev, index) {
      this.element = document.createElement('img');
      this.element.src = dev.image;
      this.element.alt = dev.name;
      this.element.className = 'developer-face';
      
      this.x = Math.random() * (window.innerWidth - 80);
      this.y = Math.random() * (window.innerHeight - 80);
      
      this.vx = (Math.random() - 0.5) * 8;
      this.vy = (Math.random() - 0.5) * 8;
      
      this.radius = 40;
      this.mass = 1;
      this.restitution = 0.85;
      this.gravity = 0.3;
      
      this.element.style.left = this.x + 'px';
      this.element.style.top = this.y + 'px';
    }
    
    update() {
      this.vy += this.gravity;
      this.x += this.vx;
      this.y += this.vy;
      
      if (this.x <= 0) {
        this.x = 0;
        this.vx = Math.abs(this.vx) * this.restitution;
      }
      if (this.x >= window.innerWidth - this.radius * 2) {
        this.x = window.innerWidth - this.radius * 2;
        this.vx = -Math.abs(this.vx) * this.restitution;
      }
      if (this.y <= 0) {
        this.y = 0;
        this.vy = Math.abs(this.vy) * this.restitution;
      }
      if (this.y >= window.innerHeight - this.radius * 2) {
        this.y = window.innerHeight - this.radius * 2;
        this.vy = -Math.abs(this.vy) * this.restitution;
        this.vx *= 0.98;
      }
      
      this.vx *= 0.995;
      this.vy *= 0.995;
      
      this.element.style.left = this.x + 'px';
      this.element.style.top = this.y + 'px';
    }
    
    checkCollision(other) {
      const dx = other.x - this.x;
      const dy = other.y - this.y;
      const distance = Math.sqrt(dx * dx + dy * dy);
      const minDistance = this.radius + other.radius;
      
      if (distance < minDistance) {
        const angle = Math.atan2(dy, dx);
        const sin = Math.sin(angle);
        const cos = Math.cos(angle);
        
        const pos0 = { x: 0, y: 0 };
        const pos1 = rotate(dx, dy, sin, cos, true);
        
        const vel0 = rotate(this.vx, this.vy, sin, cos, true);
        const vel1 = rotate(other.vx, other.vy, sin, cos, true);
        
        const vxTotal = vel0.x - vel1.x;
        vel0.x = ((this.mass - other.mass) * vel0.x + 2 * other.mass * vel1.x) / (this.mass + other.mass);
        vel1.x = vxTotal + vel0.x;
        
        const absV = Math.abs(vel0.x) + Math.abs(vel1.x);
        const overlap = (this.radius + other.radius) - Math.abs(pos0.x - pos1.x);
        pos0.x += vel0.x / absV * overlap;
        pos1.x += vel1.x / absV * overlap;
        
        const pos0F = rotate(pos0.x, pos0.y, sin, cos, false);
        const pos1F = rotate(pos1.x, pos1.y, sin, cos, false);
        
        other.x = this.x + pos1F.x;
        other.y = this.y + pos1F.y;
        this.x = this.x + pos0F.x;
        this.y = this.y + pos0F.y;
        
        const vel0F = rotate(vel0.x, vel0.y, sin, cos, false);
        const vel1F = rotate(vel1.x, vel1.y, sin, cos, false);
        
        this.vx = vel0F.x * this.restitution;
        this.vy = vel0F.y * this.restitution;
        other.vx = vel1F.x * other.restitution;
        other.vy = vel1F.y * other.restitution;
      }
    }
  }

  function rotate(x, y, sin, cos, reverse) {
    return {
      x: reverse ? (x * cos + y * sin) : (x * cos - y * sin),
      y: reverse ? (y * cos - x * sin) : (y * cos + x * sin)
    };
  }

  function spawnDeveloperFaces() {
    const container = document.getElementById('developerFacesContainer');
    container.innerHTML = '';
    faces = [];
    
    developers.forEach((dev, index) => {
      const face = new DeveloperFace(dev, index);
      faces.push(face);
      container.appendChild(face.element);
    });
    
    if (animationId) {
      cancelAnimationFrame(animationId);
    }
    
    function animate() {
      faces.forEach(face => face.update());
      
      for (let i = 0; i < faces.length; i++) {
        for (let j = i + 1; j < faces.length; j++) {
          faces[i].checkCollision(faces[j]);
        }
      }
      
      animationId = requestAnimationFrame(animate);
    }
    
    animate();
    
    setTimeout(() => {
      faces.forEach(face => {
        face.element.classList.add('removing');
      });
      
      setTimeout(() => {
        if (animationId) {
          cancelAnimationFrame(animationId);
          animationId = null;
        }
        container.innerHTML = '';
        faces = [];
      }, 500);
    }, 5000);
  }

  // Get DOM elements
  const dropArea = document.getElementById("drop-area");
  const inputFile = document.getElementById("input-file");
  const cameraInput = document.getElementById("camera-input");
  const imgView = document.getElementById("img-view");
  const analyzeBtn = document.getElementById("analyzeBtn");
  const loadingSpinner = document.getElementById("loadingSpinner");
  const resultsContainer = document.getElementById("resultsContainer");

  // Event listeners
  inputFile.addEventListener("change", uploadImage);
  cameraInput.addEventListener("change", uploadImage);

  // Camera button click handler
  document.getElementById("cameraBtn").addEventListener("click", (e) => {
    e.preventDefault();
    e.stopPropagation();
    cameraInput.click();
  });

  function uploadImage() {
    const file = inputFile.files[0] || cameraInput.files[0];
    if (file) {
      const imgLink = URL.createObjectURL(file);
      imgView.style.backgroundImage = `url(${imgLink})`;
      imgView.style.backgroundSize = "cover";
      imgView.style.backgroundPosition = "center";
      imgView.innerHTML = "";
      imgView.classList.add("has-image");
      analyzeBtn.disabled = false;
    }
  }

  dropArea.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropArea.classList.add("dragover");
  });

  dropArea.addEventListener("dragleave", () => {
    dropArea.classList.remove("dragover");
  });

  dropArea.addEventListener("drop", (e) => {
    e.preventDefault();
    dropArea.classList.remove("dragover");
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      inputFile.files = files;
      uploadImage();
    }
  });

  // Top button event listener - NOW INSIDE DOMContentLoaded
  document.getElementById("topLeftButton").addEventListener("click", () => {
    showDeveloperIntro();
  });

  document.querySelector("h1").addEventListener("click", () => {
    if (confirm("Reset and analyze a new banana?")) {
      resetAnalysis();
    }
  });

  document.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && document.activeElement === dropArea) {
      inputFile.click();
    }
  });

  console.log("%c🍌 BANANA EATS 🍌", "font-size: 24px; font-weight: bold; color: #fff176; text-shadow: 2px 2px 4px #333;");
  console.log("%cWelcome to Banana Eats! Upload a banana image to check its ripeness.", "font-size: 14px; color: #666;");
}

// Global functions that need to be accessible from onclick attributes
async function sendImage() {
  const inputFile = document.getElementById("input-file");
  const cameraInput = document.getElementById("camera-input");
  const analyzeBtn = document.getElementById("analyzeBtn");
  const loadingSpinner = document.getElementById("loadingSpinner");
  const resultsContainer = document.getElementById("resultsContainer");

  if (!inputFile.files.length && !cameraInput.files.length) {
    alert("Please select an image");
    return;
  }

  const formData = new FormData();
  const file = inputFile.files.length > 0 ? inputFile.files[0] : cameraInput.files[0];
  formData.append("image", file);
  const backendURL = "http://127.0.0.1:5001/api/process-image";

  analyzeBtn.disabled = true;
  loadingSpinner.classList.add("active");
  resultsContainer.classList.remove("active");

  try {
    const response = await fetch(backendURL, {
      method: "POST",
      body: formData
    });

    if (!response.ok) {
      throw new Error("Server returned " + response.status);
    }

    const data = await response.json();
    loadingSpinner.classList.remove("active");
    analyzeBtn.disabled = false;
    displayResults(data);

  } catch (err) {
    loadingSpinner.classList.remove("active");
    analyzeBtn.disabled = false;
    alert(`Error: ${err.message}. Make sure the backend server is running.`);
  }
}

function displayResults(data) {
  const resultsContainer = document.getElementById("resultsContainer");
  resultsContainer.classList.add("active");
  const percentages = data.percentages;
  
  updateProgressBar("yellow", percentages.yellow);
  updateProgressBar("brown", percentages.brown);
  updateProgressBar("black", percentages.black);
  updateProgressBar("green", percentages.green);
  updateRipenessStatus(percentages);

  const outputImg = document.getElementById("outputImage");
  outputImg.src = "data:image/png;base64," + data.processed_image;
  outputImg.classList.add("active");

  // Display prediction details
  displayPredictionDetails(data);

  setTimeout(() => {
    resultsContainer.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }, 300);
  
  // Display the banana comment from backend
  if (data.banana_comment) {
    displayBananaComment(data.banana_comment);
  }
}

function displayPredictionDetails(data) {
  const detailsSection = document.getElementById("predictionDetails");
  detailsSection.classList.add("active");
  
  // Update values
  document.getElementById("classificationValue").textContent = data.classification || "Unknown";
  document.getElementById("confidenceValue").textContent = 
    data.confidence ? `${(data.confidence * 100).toFixed(1)}%` : "N/A";
  document.getElementById("daysValue").textContent = 
    data.days_estimate ? `${data.days_estimate.toFixed(1)} days` : "N/A";
  document.getElementById("uncertaintyValue").textContent = 
    data.uncertainty ? `±${data.uncertainty.toFixed(1)} days` : "N/A";
  
  // Update uncertainty explanation
  const uncertaintyExplanation = document.getElementById("uncertaintyExplanation");
  if (data.uncertainty) {
    if (data.uncertainty < 1.5) {
      uncertaintyExplanation.textContent = 
        "✅ Very Low Uncertainty - The model is highly confident in this prediction!";
      uncertaintyExplanation.style.color = "#2e7d32";
    } else if (data.uncertainty < 3) {
      uncertaintyExplanation.textContent = 
        "✓ Low Uncertainty - The model is confident in this prediction.";
      uncertaintyExplanation.style.color = "#558b2f";
    } else if (data.uncertainty < 5) {
      uncertaintyExplanation.textContent = 
        "⚠ Moderate Uncertainty - The prediction is reasonable but less certain.";
      uncertaintyExplanation.style.color = "#f57c00";
    } else {
      uncertaintyExplanation.textContent = 
        "⚠️ High Uncertainty - The model is less confident. The banana might have unusual features.";
      uncertaintyExplanation.style.color = "#d32f2f";
    }
  }
}

function updateProgressBar(color, percentage) {
  const bar = document.getElementById(`${color}Bar`);
  const percentText = document.getElementById(`${color}Percent`);
  
  setTimeout(() => {
    bar.style.width = `${percentage}%`;
    animateValue(percentText, 0, percentage, 1000);
  }, 100);
}

function animateValue(element, start, end, duration) {
  const range = end - start;
  const increment = range / (duration / 16);
  let current = start;
  
  const timer = setInterval(() => {
    current += increment;
    if ((increment > 0 && current >= end) || (increment < 0 && current <= end)) {
      current = end;
      clearInterval(timer);
    }
    element.textContent = `${Math.round(current)}%`;
  }, 16);
}

function updateRipenessStatus(percentages) {
  const statusIndicator = document.getElementById("statusIndicator");
  const yellow = percentages.yellow;
  const brown = percentages.brown;
  const black = percentages.black;
  
  let emoji, status;
  
  if (black > 20) {
    emoji = "🚫";
    status = "Too Ripe - Not Edible";
  } else if (brown > 40) {
    emoji = "⚠️";
    status = "Very Ripe - Eat Soon!";
  } else if (brown > 20 || yellow > 70) {
    emoji = "✅";
    status = "Perfect - Ready to Eat!";
  } else if (yellow > 40) {
    emoji = "⏰";
    status = "Nearly Ripe - Wait a Day";
  } else {
    emoji = "🟢";
    status = "Unripe - Wait a Few Days";
  }
  
  statusIndicator.innerHTML = `
    <div class="status-emoji">${emoji}</div>
    <p class="status-text">${status}</p>
  `;
}

function displayBananaComment(comment) {
  const quoteContainer = document.querySelector('.banana-quote-container');
  const quoteText = document.getElementById('bananaQuote');
  
  if (!quoteContainer || !quoteText) {
    console.error('Banana quote elements not found');
    return;
  }
  
  // Display the comment from backend
  quoteText.textContent = comment;
  quoteContainer.classList.add('active');
}

function resetAnalysis() {
  const imgView = document.getElementById("img-view");
  const inputFile = document.getElementById("input-file");
  const cameraInput = document.getElementById("camera-input");
  const analyzeBtn = document.getElementById("analyzeBtn");
  const resultsContainer = document.getElementById("resultsContainer");
  const loadingSpinner = document.getElementById("loadingSpinner");

  imgView.style.backgroundImage = "";
  imgView.classList.remove("has-image");
  imgView.innerHTML = `
    <img src="rotating-banana-banana.gif" alt="Upload Icon" class="upload-icon">
    <p class="upload-title">Click or Drag to Upload</p>
    <span class="upload-subtitle">Upload an image of your banana</span>
    <p class="upload-description"><b>Send an image of your banana and we will determine its edibility 🍌</b></p>
    <button id="cameraBtn" class="camera-btn" type="button">
      <span class="camera-icon">📷</span>
      Open Camera
    </button>
  `;
  
  inputFile.value = "";
  cameraInput.value = "";
  analyzeBtn.disabled = true;
  resultsContainer.classList.remove("active");
  loadingSpinner.classList.remove("active");
  
  // Re-attach camera button listener after reset
  document.getElementById("cameraBtn").addEventListener("click", (e) => {
    e.preventDefault();
    e.stopPropagation();
    cameraInput.click();
  });
  
  // Hide the banana quote
  const quoteContainer = document.querySelector('.banana-quote-container');
  if (quoteContainer) {
    quoteContainer.classList.remove('active');
  }
  
  // Hide prediction details
  const predictionDetails = document.getElementById('predictionDetails');
  if (predictionDetails) {
    predictionDetails.classList.remove('active');
  }
  
  window.scrollTo({ top: 0, behavior: "smooth" });
}

// Toggle Agent function for the AI widget
function toggleAgent() {
  const aiAgent = document.getElementById('aiAgent');
  aiAgent.classList.toggle('minimized');
}

// Carousel variables
let currentSlide = 0;
let carouselInterval = null;
let progressInterval = null;
const SLIDE_DURATION = 7000; // 7 seconds per slide

// Show Developer Intro Modal
function showDeveloperIntro() {
  const modal = document.getElementById('developerIntroModal');
  const music = document.getElementById('introMusic');
  
  modal.classList.add('active');
  
  // Play background music
  music.volume = 0.3; // Set volume to 30%
  music.play().catch(error => {
    console.log('Audio autoplay prevented:', error);
  });
  
  // Start carousel
  startCarousel();
  
  // Prevent body scroll when modal is open
  document.body.style.overflow = 'hidden';
}

// Close Developer Intro Modal
function closeDeveloperIntro() {
  const modal = document.getElementById('developerIntroModal');
  const music = document.getElementById('introMusic');
  
  modal.classList.remove('active');
  
  // Stop carousel
  stopCarousel();
  
  // Fade out and stop music
  const fadeOut = setInterval(() => {
    if (music.volume > 0.05) {
      music.volume -= 0.05;
    } else {
      music.pause();
      music.currentTime = 0;
      music.volume = 0.3;
      clearInterval(fadeOut);
    }
  }, 50);
  
  // Re-enable body scroll
  document.body.style.overflow = 'auto';
  
  // Reset to first slide
  currentSlide = 0;
  updateSlides();
}

// Start Carousel Auto-rotation
function startCarousel() {
  // Clear any existing intervals
  stopCarousel();
  
  // Start progress bar animation
  startProgressBar();
  
  // Auto-advance slides
  carouselInterval = setInterval(() => {
    nextSlide();
  }, SLIDE_DURATION);
  
  // Add hover pause functionality
  const carouselContainer = document.querySelector('.carousel-container');
  if (carouselContainer) {
    carouselContainer.addEventListener('mouseenter', pauseCarousel);
    carouselContainer.addEventListener('mouseleave', resumeCarousel);
  }
}

// Pause Carousel
function pauseCarousel() {
  if (carouselInterval) {
    clearInterval(carouselInterval);
    carouselInterval = null;
  }
  if (progressInterval) {
    clearInterval(progressInterval);
    progressInterval = null;
  }
}

// Resume Carousel
function resumeCarousel() {
  startCarousel();
}

// Stop Carousel
function stopCarousel() {
  if (carouselInterval) {
    clearInterval(carouselInterval);
    carouselInterval = null;
  }
  if (progressInterval) {
    clearInterval(progressInterval);
    progressInterval = null;
  }
}

// Progress Bar Animation
function startProgressBar() {
  const progressBar = document.getElementById('carouselProgress');
  let progress = 0;
  const increment = 100 / (SLIDE_DURATION / 50); // Update every 50ms
  
  if (progressInterval) {
    clearInterval(progressInterval);
  }
  
  progressBar.style.width = '0%';
  
  progressInterval = setInterval(() => {
    progress += increment;
    if (progress >= 100) {
      progress = 0;
    }
    progressBar.style.width = progress + '%';
  }, 50);
}

// Navigate to specific slide
function goToSlide(slideIndex) {
  currentSlide = slideIndex;
  updateSlides();
  
  // Restart carousel timer
  stopCarousel();
  startCarousel();
}

// Next Slide
function nextSlide() {
  currentSlide = (currentSlide + 1) % 4; // 4 slides total
  updateSlides();
  
  // Restart progress bar
  startProgressBar();
}

// Previous Slide
function previousSlide() {
  currentSlide = (currentSlide - 1 + 4) % 4; // 4 slides total
  updateSlides();
  
  // Restart carousel timer
  stopCarousel();
  startCarousel();
}

// Update Slides Display
function updateSlides() {
  const slides = document.querySelectorAll('.carousel-slide');
  const dots = document.querySelectorAll('.carousel-dot');
  
  slides.forEach((slide, index) => {
    slide.classList.remove('active', 'prev');
    
    if (index === currentSlide) {
      slide.classList.add('active');
    } else if (index < currentSlide) {
      slide.classList.add('prev');
    }
  });
  
  // Update dots
  dots.forEach((dot, index) => {
    dot.classList.toggle('active', index === currentSlide);
  });
}

// Close modal when clicking outside
document.addEventListener('click', (e) => {
  const modal = document.getElementById('developerIntroModal');
  if (e.target === modal) {
    closeDeveloperIntro();
  }
});

// Close modal with Escape key and add arrow key navigation
document.addEventListener('keydown', (e) => {
  const modal = document.getElementById('developerIntroModal');
  
  if (modal.classList.contains('active')) {
    if (e.key === 'Escape') {
      closeDeveloperIntro();
    } else if (e.key === 'ArrowRight') {
      nextSlide();
      stopCarousel();
      startCarousel();
    } else if (e.key === 'ArrowLeft') {
      previousSlide();
      stopCarousel();
      startCarousel();
    }
  }
});