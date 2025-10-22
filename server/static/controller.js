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
const joystickStick = document.getElementById("joystick-stick");

// Joystick direction indicators
const joystickIndicators = {
  n: document.querySelector('.joystick-dir-n'),
  ne: document.querySelector('.joystick-dir-ne'),
  e: document.querySelector('.joystick-dir-e'),
  se: document.querySelector('.joystick-dir-se'),
  s: document.querySelector('.joystick-dir-s'),
  sw: document.querySelector('.joystick-dir-sw'),
  w: document.querySelector('.joystick-dir-w'),
  nw: document.querySelector('.joystick-dir-nw'),
};

// Buttons
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

// Joystick state
let joystickActive = false;

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

  // Update joystick visual position (digital snapping)
  const maxOffset = 60; // max pixels to move the stick
  joystickStick.style.transform = `translate(calc(-50% + ${x * maxOffset}px), calc(-50% + ${-y * maxOffset}px))`;

  // Calculate motor values (differential drive)
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

// Calculate digital direction from angle and distance
// Returns 8-way digital input: N, NE, E, SE, S, SW, W, NW, or neutral
function calculateDigitalDirection(touchX, touchY, baseRect) {
  const centerX = baseRect.left + baseRect.width / 2;
  const centerY = baseRect.top + baseRect.height / 2;

  const deltaX = touchX - centerX;
  const deltaY = touchY - centerY;

  const distance = Math.sqrt(deltaX * deltaX + deltaY * deltaY);
  const deadzone = 30; // pixels from center

  if (distance < deadzone) {
    return { x: 0, y: 0, direction: null };
  }

  // Calculate angle in degrees (0 = right, 90 = down, 180 = left, 270 = up)
  let angle = Math.atan2(deltaY, deltaX) * 180 / Math.PI;

  // Normalize to 0-360
  if (angle < 0) angle += 360;

  // Convert to direction with Y-up coordinate system (inverted Y)
  // Adjust angle so 0° = up
  angle = (450 - angle) % 360;

  // 8-way digital direction (45° segments)
  let x = 0, y = 0, direction = null;

  if (angle >= 337.5 || angle < 22.5) {
    // N (up)
    y = 1;
    direction = 'n';
  } else if (angle >= 22.5 && angle < 67.5) {
    // NE
    x = 1; y = 1;
    direction = 'ne';
  } else if (angle >= 67.5 && angle < 112.5) {
    // E (right)
    x = 1;
    direction = 'e';
  } else if (angle >= 112.5 && angle < 157.5) {
    // SE
    x = 1; y = -1;
    direction = 'se';
  } else if (angle >= 157.5 && angle < 202.5) {
    // S (down)
    y = -1;
    direction = 's';
  } else if (angle >= 202.5 && angle < 247.5) {
    // SW
    x = -1; y = -1;
    direction = 'sw';
  } else if (angle >= 247.5 && angle < 292.5) {
    // W (left)
    x = -1;
    direction = 'w';
  } else if (angle >= 292.5 && angle < 337.5) {
    // NW
    x = -1; y = 1;
    direction = 'nw';
  }

  return { x, y, direction };
}

// Update direction indicators
function updateDirectionLabels(activeDirection) {
  Object.keys(joystickIndicators).forEach(dir => {
    if (joystickIndicators[dir]) {
      joystickIndicators[dir].classList.toggle('active', dir === activeDirection);
    }
  });
}

// Handle joystick interaction
function handleJoystickMove(touchX, touchY) {
  const baseRect = joystickStick.parentElement.getBoundingClientRect();
  const { x, y, direction } = calculateDigitalDirection(touchX, touchY, baseRect);

  currentX = x;
  currentY = y;

  if (x !== 0 || y !== 0) {
    joystickStick.classList.add('active');
    joystickActive = true;
  } else {
    joystickStick.classList.remove('active');
    joystickActive = false;
  }

  updateDirectionLabels(direction);
  updateControl();
}

function resetJoystick() {
  currentX = 0;
  currentY = 0;
  joystickActive = false;
  joystickStick.classList.remove('active');
  joystickStick.style.transform = 'translate(-50%, -50%)';
  updateDirectionLabels(null);
  updateControl();
}

// Touch events for joystick
joystickStick.addEventListener('touchstart', (e) => {
  e.preventDefault();
  const touch = e.touches[0];
  handleJoystickMove(touch.clientX, touch.clientY);
});

joystickStick.addEventListener('touchmove', (e) => {
  e.preventDefault();
  const touch = e.touches[0];
  handleJoystickMove(touch.clientX, touch.clientY);
});

joystickStick.addEventListener('touchend', (e) => {
  e.preventDefault();
  resetJoystick();
});

joystickStick.addEventListener('touchcancel', (e) => {
  e.preventDefault();
  resetJoystick();
});

// Mouse events for joystick (desktop)
let mouseDown = false;

joystickStick.addEventListener('mousedown', (e) => {
  mouseDown = true;
  handleJoystickMove(e.clientX, e.clientY);
});

document.addEventListener('mousemove', (e) => {
  if (mouseDown) {
    handleJoystickMove(e.clientX, e.clientY);
  }
});

document.addEventListener('mouseup', () => {
  if (mouseDown) {
    mouseDown = false;
    resetJoystick();
  }
});

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
  resetJoystick();
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

// Keyboard controls (8-way digital)
const keyMap = {
  'w': { x: 0, y: 1, dir: 'n' },
  'W': { x: 0, y: 1, dir: 'n' },
  'ArrowUp': { x: 0, y: 1, dir: 'n' },
  's': { x: 0, y: -1, dir: 's' },
  'S': { x: 0, y: -1, dir: 's' },
  'ArrowDown': { x: 0, y: -1, dir: 's' },
  'a': { x: -1, y: 0, dir: 'w' },
  'A': { x: -1, y: 0, dir: 'w' },
  'ArrowLeft': { x: -1, y: 0, dir: 'w' },
  'd': { x: 1, y: 0, dir: 'e' },
  'D': { x: 1, y: 0, dir: 'e' },
  'ArrowRight': { x: 1, y: 0, dir: 'e' },
};

let activeKeys = new Set();

window.addEventListener("keydown", (e) => {
  const mapping = keyMap[e.key];
  if (mapping && !e.repeat) {
    e.preventDefault();
    activeKeys.add(e.key);

    // Combine active keys for diagonal movement
    let x = 0, y = 0, dir = null;
    for (const key of activeKeys) {
      const m = keyMap[key];
      x += m.x;
      y += m.y;
    }

    // Clamp to -1, 0, 1
    x = Math.max(-1, Math.min(1, x));
    y = Math.max(-1, Math.min(1, y));

    // Determine direction
    if (x === 0 && y === 1) dir = 'n';
    else if (x === 1 && y === 1) dir = 'ne';
    else if (x === 1 && y === 0) dir = 'e';
    else if (x === 1 && y === -1) dir = 'se';
    else if (x === 0 && y === -1) dir = 's';
    else if (x === -1 && y === -1) dir = 'sw';
    else if (x === -1 && y === 0) dir = 'w';
    else if (x === -1 && y === 1) dir = 'nw';

    currentX = x;
    currentY = y;

    if (x !== 0 || y !== 0) {
      joystickStick.classList.add('active');
      const maxOffset = 60;
      joystickStick.style.transform = `translate(calc(-50% + ${x * maxOffset}px), calc(-50% + ${-y * maxOffset}px))`;
    } else {
      joystickStick.classList.remove('active');
      joystickStick.style.transform = 'translate(-50%, -50%)';
    }

    updateDirectionLabels(dir);
    updateControl();
  }
});

window.addEventListener("keyup", (e) => {
  const mapping = keyMap[e.key];
  if (mapping) {
    e.preventDefault();
    activeKeys.delete(e.key);

    // Recalculate direction from remaining keys
    let x = 0, y = 0, dir = null;
    for (const key of activeKeys) {
      const m = keyMap[key];
      x += m.x;
      y += m.y;
    }

    x = Math.max(-1, Math.min(1, x));
    y = Math.max(-1, Math.min(1, y));

    if (x === 0 && y === 1) dir = 'n';
    else if (x === 1 && y === 1) dir = 'ne';
    else if (x === 1 && y === 0) dir = 'e';
    else if (x === 1 && y === -1) dir = 'se';
    else if (x === 0 && y === -1) dir = 's';
    else if (x === -1 && y === -1) dir = 'sw';
    else if (x === -1 && y === 0) dir = 'w';
    else if (x === -1 && y === 1) dir = 'nw';

    currentX = x;
    currentY = y;

    if (x !== 0 || y !== 0) {
      const maxOffset = 60;
      joystickStick.style.transform = `translate(calc(-50% + ${x * maxOffset}px), calc(-50% + ${-y * maxOffset}px))`;
    } else {
      joystickStick.classList.remove('active');
      joystickStick.style.transform = 'translate(-50%, -50%)';
    }

    updateDirectionLabels(dir);
    updateControl();
  }
});

// Prevent context menu on long press
document.addEventListener("contextmenu", (e) => {
  e.preventDefault();
});

// Initialize connection
connect();

// Send periodic updates when joystick is active
setInterval(() => {
  if (joystickActive || activeKeys.size > 0) {
    updateControl();
  }
}, sendInterval);
