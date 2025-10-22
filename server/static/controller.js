const wsUrl = location.origin.replace(/^http/, "ws") + "/ws";

// UI Elements
const statusBadge = document.getElementById("status-badge");
const statusText = document.getElementById("status-text");
const latencyEl = document.getElementById("latency");
const axisX = document.getElementById("axis-x");
const axisY = document.getElementById("axis-y");
const motorLeft = document.getElementById("motor-left");
const motorRight = document.getElementById("motor-right");
const speedValue = document.getElementById("speed-value");
const speedSlider = document.getElementById("speed-slider");
const directionIndicator = document.getElementById("direction-indicator");

// Buttons
const btnForward = document.getElementById("btn-forward");
const btnBackward = document.getElementById("btn-backward");
const btnLeft = document.getElementById("btn-left");
const btnRight = document.getElementById("btn-right");
const btnEstop = document.getElementById("estop");
const btnStartEpisode = document.getElementById("start-episode");
const btnEndEpisode = document.getElementById("end-episode");

// State
let ws;
let connected = false;
let lastSend = 0;
const sendRateHz = 60;
const sendInterval = 1000 / sendRateHz;

let currentX = 0;
let currentY = 0;
let speedMultiplier = 0.7;

// Direction button state
const activeButtons = new Set();

// WebSocket Connection
function connect() {
  ws = new WebSocket(wsUrl);

  ws.onopen = () => {
    connected = true;
    statusBadge.classList.add("connected");
    statusBadge.classList.remove("disconnected");
    statusText.textContent = "接続済み";
  };

  ws.onclose = () => {
    connected = false;
    statusBadge.classList.remove("connected");
    statusBadge.classList.add("disconnected");
    statusText.textContent = "切断";
    latencyEl.textContent = "— ms";
    setTimeout(connect, 1000);
  };

  ws.onmessage = (event) => {
    try {
      const msg = JSON.parse(event.data);
      if (msg.type === "ping") {
        const latency = Math.round((performance.now() / 1000 - msg.ts) * 1000);
        latencyEl.textContent = `${latency} ms`;
      }
    } catch (e) {
      // Ignore parse errors
    }
  };
}

// Send stick command
function sendStick(x, y) {
  if (!connected || !ws) return;

  const now = performance.now();
  if (now - lastSend < sendInterval) return;

  lastSend = now;
  ws.send(JSON.stringify({
    type: "stick",
    x: x,
    y: y,
    ts: now / 1000
  }));
}

// Update UI and send command
function updateControl() {
  // Apply speed multiplier
  const x = currentX * speedMultiplier;
  const y = currentY * speedMultiplier;

  // Update UI
  axisX.textContent = x.toFixed(2);
  axisY.textContent = y.toFixed(2);

  // Update direction indicator
  if (x !== 0 || y !== 0) {
    directionIndicator.classList.add("active");
  } else {
    directionIndicator.classList.remove("active");
  }

  // Calculate motor values (differential drive)
  // Y is forward/backward, X is turning
  // Turning sensitivity: balance between sharp turns and control
  // 0.7 allows clear turning while maintaining forward momentum
  const turningSensitivity = 0.7;
  const adjustedX = x * turningSensitivity;

  let left = y + adjustedX;
  let right = y - adjustedX;

  // Normalize to -1 to 1 range
  const maxVal = Math.max(Math.abs(left), Math.abs(right), 1);
  left = left / maxVal;
  right = right / maxVal;

  // Scale to -100 to 100 and apply speed multiplier
  left = Math.round(left * 100 * speedMultiplier);
  right = Math.round(right * 100 * speedMultiplier);

  motorLeft.textContent = left;
  motorRight.textContent = right;

  // Send to WebSocket
  sendStick(x, y);
}

// Handle direction buttons
function handleButtonPress(direction) {
  activeButtons.add(direction);
  updateDirectionFromButtons();
}

function handleButtonRelease(direction) {
  activeButtons.delete(direction);
  updateDirectionFromButtons();
}

function updateDirectionFromButtons() {
  // Reset
  currentX = 0;
  currentY = 0;

  // Calculate based on active buttons
  if (activeButtons.has("forward")) currentY += 1;
  if (activeButtons.has("backward")) currentY -= 1;
  if (activeButtons.has("left")) currentX -= 1;
  if (activeButtons.has("right")) currentX += 1;

  // Update visual state
  btnForward.classList.toggle("active", activeButtons.has("forward"));
  btnBackward.classList.toggle("active", activeButtons.has("backward"));
  btnLeft.classList.toggle("active", activeButtons.has("left"));
  btnRight.classList.toggle("active", activeButtons.has("right"));

  updateControl();
}

// Button Event Listeners (touch and mouse)
function addButtonListeners(button, direction) {
  // Touch events
  button.addEventListener("touchstart", (e) => {
    e.preventDefault();
    handleButtonPress(direction);
  });

  button.addEventListener("touchend", (e) => {
    e.preventDefault();
    handleButtonRelease(direction);
  });

  button.addEventListener("touchcancel", (e) => {
    e.preventDefault();
    handleButtonRelease(direction);
  });

  // Mouse events (for desktop)
  button.addEventListener("mousedown", () => {
    handleButtonPress(direction);
  });

  button.addEventListener("mouseup", () => {
    handleButtonRelease(direction);
  });

  button.addEventListener("mouseleave", () => {
    handleButtonRelease(direction);
  });
}

// Setup direction buttons
addButtonListeners(btnForward, "forward");
addButtonListeners(btnBackward, "backward");
addButtonListeners(btnLeft, "left");
addButtonListeners(btnRight, "right");

// Speed slider
speedSlider.addEventListener("input", (e) => {
  const value = parseInt(e.target.value);
  speedMultiplier = value / 100;
  speedValue.textContent = `${value}%`;
  updateControl();
});

// Emergency stop
btnEstop.addEventListener("click", () => {
  if (ws && connected) {
    ws.send(JSON.stringify({ type: "estop", reason: "user" }));
  }
  activeButtons.clear();
  currentX = 0;
  currentY = 0;
  updateControl();
});

// Recording controls
btnStartEpisode.addEventListener("click", () => {
  if (ws && connected) {
    ws.send(JSON.stringify({ type: "recording", command: "start_episode" }));
    btnStartEpisode.disabled = true;
    btnEndEpisode.disabled = false;
  }
});

btnEndEpisode.addEventListener("click", () => {
  if (ws && connected) {
    ws.send(JSON.stringify({ type: "recording", command: "end_episode" }));
    btnStartEpisode.disabled = false;
    btnEndEpisode.disabled = true;
  }
});

// Keyboard controls
const keyMap = {
  'w': 'forward',
  'W': 'forward',
  'ArrowUp': 'forward',
  's': 'backward',
  'S': 'backward',
  'ArrowDown': 'backward',
  'a': 'left',
  'A': 'left',
  'ArrowLeft': 'left',
  'd': 'right',
  'D': 'right',
  'ArrowRight': 'right',
};

window.addEventListener("keydown", (e) => {
  const direction = keyMap[e.key];
  if (direction && !e.repeat) {
    e.preventDefault();
    handleButtonPress(direction);
  }
});

window.addEventListener("keyup", (e) => {
  const direction = keyMap[e.key];
  if (direction) {
    e.preventDefault();
    handleButtonRelease(direction);
  }
});

// Prevent context menu on long press
document.addEventListener("contextmenu", (e) => {
  e.preventDefault();
});

// Initialize connection
connect();

// Send periodic updates when buttons are pressed
setInterval(() => {
  if (activeButtons.size > 0) {
    updateControl();
  }
}, sendInterval);
