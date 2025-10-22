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
const joystickBase = joystickStick.parentElement;

// Joystick direction indicators (4-way only)
const joystickIndicators = {
  n: document.querySelector('.joystick-dir-n'),
  e: document.querySelector('.joystick-dir-e'),
  s: document.querySelector('.joystick-dir-s'),
  w: document.querySelector('.joystick-dir-w'),
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

// Rotation control - for pulsed 45° rotation
let rotationTimeout = null;

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
  console.log(`[SEND] x=${x.toFixed(2)}, y=${y.toFixed(2)}`);
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

  console.log(`[UPDATE] currentX=${currentX}, currentY=${currentY}, x=${x.toFixed(2)}, y=${y.toFixed(2)}`);

  // Update UI
  axisX.textContent = x.toFixed(2);
  axisY.textContent = y.toFixed(2);

  // Update joystick visual position (snap to 4 directions only)
  const maxOffset = 60; // max pixels to move the stick
  const visualX = x * maxOffset;
  const visualY = -y * maxOffset; // Invert Y for screen coordinates (screen down = negative visual offset)
  joystickStick.style.transform = `translate(calc(-50% + ${visualX}px), calc(-50% + ${visualY}px))`;

  // Calculate motor values (differential drive)
  let left, right;

  if (y === 0 && x !== 0) {
    // Pure rotation (left/right only): slow rotation for precise 45° turns
    // Left motor and right motor spin in opposite directions at low speed
    const rotationSpeed = 30; // Low speed for controlled rotation
    left = x * rotationSpeed;
    right = -x * rotationSpeed;
  } else {
    // Forward/backward or combined movement
    const turningSensitivity = 0.7;
    const adjustedX = x * turningSensitivity;

    left = y + adjustedX;
    right = y - adjustedX;

    // Normalize to -1 to 1 range
    const maxVal = Math.max(Math.abs(left), Math.abs(right), 1);
    left = left / maxVal;
    right = right / maxVal;

    // Scale to -100 to 100 and apply speed multiplier
    left = Math.round(left * 100 * speedMultiplier);
    right = Math.round(right * 100 * speedMultiplier);
  }

  motorLeft.textContent = left;
  motorRight.textContent = right;

  // Send to WebSocket
  sendStick(x, y);
}

// Calculate digital direction from angle and distance
// Returns 8-way digital input: N, NE, E, SE, S, SW, W, NW, or neutral
// Prioritizes cardinal directions (N/S/E/W) with wider zones for easier operation
function calculateDigitalDirection(touchX, touchY, baseRect) {
  const centerX = baseRect.left + baseRect.width / 2;
  const centerY = baseRect.top + baseRect.height / 2;

  const deltaX = touchX - centerX;
  const deltaY = touchY - centerY; // Screen down = positive deltaY = move down/backward

  const distance = Math.sqrt(deltaX * deltaX + deltaY * deltaY);
  const deadzone = 30; // pixels from center

  if (distance < deadzone) {
    return { x: 0, y: 0, direction: null };
  }

  // Calculate angle in screen coordinates
  // atan2(deltaY, deltaX): 0° = right, 90° = down, 180° = left, -90° = up
  let angle = Math.atan2(deltaY, deltaX) * 180 / Math.PI;

  // Normalize to 0-360
  if (angle < 0) angle += 360;

  // Debug: log angle
  console.log(`Angle: ${angle.toFixed(1)}°, deltaX=${deltaX.toFixed(1)}, deltaY=${deltaY.toFixed(1)}`);

  // Map screen coordinates to robot coordinates
  // Screen: 0°=right, 90°=down, 180°=left, 270°=up
  // User expects: drag UP = forward, drag DOWN = backward
  // 4-way only (no diagonals) with 90° zones each
  let x = 0, y = 0, direction = null;

  if (angle >= 225 && angle < 315) {
    // Screen UP (270°) = Robot FORWARD
    y = 1;
    direction = 'n';
  } else if ((angle >= 315 && angle < 360) || (angle >= 0 && angle < 45)) {
    // Screen RIGHT (0°) = Robot RIGHT turn
    x = 1;
    direction = 'e';
  } else if (angle >= 45 && angle < 135) {
    // Screen DOWN (90°) = Robot BACKWARD
    y = -1;
    direction = 's';
  } else if (angle >= 135 && angle < 225) {
    // Screen LEFT (180°) = Robot LEFT turn
    x = -1;
    direction = 'w';
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

  // Debug: log direction
  console.log(`Touch: direction=${direction}, x=${x}, y=${y}`);

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
  console.log('[RESET] Stopping all movement');
  currentX = 0;
  currentY = 0;
  joystickActive = false;
  joystickStick.classList.remove('active');
  joystickStick.style.transform = 'translate(-50%, -50%)';
  updateDirectionLabels(null);

  // Force send stop command immediately
  if (connected && ws) {
    ws.send(JSON.stringify({
      type: "stick",
      x: 0,
      y: 0,
      ts: performance.now() / 1000
    }));
  }

  updateControl();
}

// Touch events for joystick - attach to base for better touch tracking
joystickBase.addEventListener('touchstart', (e) => {
  e.preventDefault();
  const touch = e.touches[0];
  handleJoystickMove(touch.clientX, touch.clientY);
});

joystickBase.addEventListener('touchmove', (e) => {
  e.preventDefault();
  const touch = e.touches[0];
  handleJoystickMove(touch.clientX, touch.clientY);
});

joystickBase.addEventListener('touchend', (e) => {
  e.preventDefault();
  console.log('[TOUCH END] Resetting joystick');
  resetJoystick();
});

joystickBase.addEventListener('touchcancel', (e) => {
  e.preventDefault();
  console.log('[TOUCH CANCEL] Resetting joystick');
  resetJoystick();
});

// Mouse events for joystick (desktop)
let mouseDown = false;

joystickBase.addEventListener('mousedown', (e) => {
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
    console.log('[MOUSE UP] Resetting joystick');
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

// Keyboard controls (4-way only - no diagonals)
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

    // Only allow one key at a time (4-way only, no combining for diagonals)
    activeKeys.clear();
    activeKeys.add(e.key);

    currentX = mapping.x;
    currentY = mapping.y;

    joystickStick.classList.add('active');
    const maxOffset = 60;
    joystickStick.style.transform = `translate(calc(-50% + ${currentX * maxOffset}px), calc(-50% + ${-currentY * maxOffset}px))`;

    updateDirectionLabels(mapping.dir);
    updateControl();
  }
});

window.addEventListener("keyup", (e) => {
  const mapping = keyMap[e.key];
  if (mapping) {
    e.preventDefault();
    activeKeys.delete(e.key);

    if (activeKeys.size === 0) {
      currentX = 0;
      currentY = 0;
      joystickStick.classList.remove('active');
      joystickStick.style.transform = 'translate(-50%, -50%)';
      updateDirectionLabels(null);
      updateControl();
    }
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
