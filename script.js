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

const dropArea = document.getElementById("drop-area");
const inputFile = document.getElementById("input-file");
const imgView = document.getElementById("img-view");
const analyzeBtn = document.getElementById("analyzeBtn");
const loadingSpinner = document.getElementById("loadingSpinner");
const resultsContainer = document.getElementById("resultsContainer");

inputFile.addEventListener("change", uploadImage);

function uploadImage() {
  const file = inputFile.files[0];
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

async function sendImage() {
  if (!inputFile.files.length) {
    alert("Please select an image");
    return;
  }

  const formData = new FormData();
  formData.append("image", inputFile.files[0]);
  const backendURL = "http://127.0.0.1:5000/api/process-image";

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
  resultsContainer.classList.add("active");
  const percentages = data.percentages;
  
  updateProgressBar("yellow", percentages.yellow);
  updateProgressBar("brown", percentages.brown);
  updateProgressBar("black", percentages.black);
  updateProgressBar("white", percentages.white);
  updateRipenessStatus(percentages);

  const outputImg = document.getElementById("outputImage");
  outputImg.src = "data:image/png;base64," + data.processed_image;
  outputImg.classList.add("active");

  setTimeout(() => {
    resultsContainer.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }, 300);
  
  addAgentComment(percentages);
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

document.getElementById("topLeftButton").addEventListener("click", () => {
  spawnDeveloperFaces();
});

document.querySelector("h1").addEventListener("click", () => {
  if (confirm("Reset and analyze a new banana?")) {
    resetAnalysis();
  }
});

function resetAnalysis() {
  imgView.style.backgroundImage = "";
  imgView.classList.remove("has-image");
  imgView.innerHTML = `
    <img src="rotating-banana-banana.gif" alt="Upload Icon" class="upload-icon">
    <p class="upload-title">Click or Drag to Upload</p>
    <span class="upload-subtitle">Upload an image of your banana</span>
    <p class="upload-description"><b>Send an image of your banana and we will determine its edibility 🍌</b></p>
  `;
  
  inputFile.value = "";
  analyzeBtn.disabled = true;
  resultsContainer.classList.remove("active");
  loadingSpinner.classList.remove("active");
  window.scrollTo({ top: 0, behavior: "smooth" });
}

document.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && document.activeElement === dropArea) {
    inputFile.click();
  }
});

console.log("%c🍌 BANANA EATS 🍌", "font-size: 24px; font-weight: bold; color: #fff176; text-shadow: 2px 2px 4px #333;");
console.log("%cWelcome to Banana Eats! Upload a banana image to check its ripeness.", "font-size: 14px; color: #666;");

function toggleAgent() {
  const agent = document.getElementById('aiAgent');
  agent.classList.toggle('minimized');
}

async function getBananaComment(percentages) {
  const prompt = `You are a friendly, enthusiastic banana expert AI with a quirky personality. 
You just analyzed a banana with these color percentages:
- Yellow: ${percentages.yellow}%
- Brown: ${percentages.brown}%
- Black: ${percentages.black}%
- White: ${percentages.white}%

Give a SHORT (2-3 sentences max), fun, and personality-filled comment about this banana. 
Be encouraging, use banana puns if appropriate, and show genuine emotion about the banana's condition.`;

  try {
    const response = await fetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "claude-sonnet-4-20250514",
        max_tokens: 1000,
        messages: [{ role: "user", content: prompt }]
      })
    });

    const data = await response.json();
    return data.content[0].text;
  } catch (error) {
    console.error('Agent error:', error);
    return "Oh no! I'm having trouble analyzing this banana right now. 🍌💔";
  }
}

async function addAgentComment(percentages) {
  const agentMessages = document.getElementById('agentMessages');
  const thinking = document.querySelector('.thinking');
  const emptyState = document.querySelector('.empty-state');
  
  if (emptyState) emptyState.remove();
  
  thinking.style.display = 'flex';
  const comment = await getBananaComment(percentages);
  thinking.style.display = 'none';
  
  const messageDiv = document.createElement('div');
  messageDiv.className = 'chat-message';
  messageDiv.innerHTML = `
    <div class="message-time">${new Date().toLocaleTimeString()}</div>
    <div class="message-text">${comment}</div>
  `;
  
  agentMessages.appendChild(messageDiv);
  agentMessages.scrollTop = agentMessages.scrollHeight;
  
  const agent = document.getElementById('aiAgent');
  agent.classList.remove('minimized');
}