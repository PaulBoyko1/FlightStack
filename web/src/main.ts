import * as THREE from "three";
import "./style.css";

type Vec3 = [number, number, number];
type QuatWxyz = [number, number, number, number];

interface GateWire {
  id: string;
  center_world_m: Vec3;
  normal_world: Vec3;
  right_world: Vec3;
  up_world: Vec3;
  half_width_m: number;
  half_height_m: number;
}

interface StateWire {
  sim_time_s: number;
  position_world_m: Vec3;
  velocity_world_m_s: Vec3;
  q_body_to_world_wxyz: QuatWxyz;
  body_rate_rad_s: Vec3;
  motor_thrust_n: [number, number, number, number];
}

interface StateMessage {
  type: "state";
  pilot: "human" | "classical" | "learned";
  vehicle: {
    name: string;
    version: string;
    motor_max_thrust_n: number;
  };
  state: StateWire;
  race: {
    lap: number;
    next_gate: number;
    lap_time_s: number;
    best_lap_s: number | null;
    collisions: number;
    status: string;
  };
  track?: GateWire[];
}

interface InputMessage {
  type: "manual_input";
  roll: number;
  pitch: number;
  yaw: number;
  throttle: number;
}

type ServerMessage = StateMessage | { type: "notice"; message: string };

const byId = <T extends HTMLElement>(id: string): T => {
  const element = document.getElementById(id);
  if (!element) throw new Error(`Missing #${id}`);
  return element as T;
};

const canvas = byId<HTMLCanvasElement>("flight-view");
const minimap = byId<HTMLCanvasElement>("minimap");
const minimapContext = minimap.getContext("2d");
if (!minimapContext) throw new Error("Could not create minimap context");

const dom = {
  connection: byId<HTMLDivElement>("connection-text"),
  connectionParent: byId<HTMLDivElement>("connection-text").parentElement as HTMLDivElement,
  speed: byId<HTMLElement>("speed"),
  altitude: byId<HTMLElement>("altitude"),
  gate: byId<HTMLElement>("gate"),
  lap: byId<HTMLElement>("lap"),
  rateX: byId<HTMLElement>("rate-x"),
  rateY: byId<HTMLElement>("rate-y"),
  rateZ: byId<HTMLElement>("rate-z"),
  thrust: byId<HTMLElement>("thrust"),
  motors: byId<HTMLDivElement>("motors"),
  technicalDrawer: byId<HTMLElement>("technical-drawer"),
  toast: byId<HTMLElement>("toast"),
  reset: byId<HTMLButtonElement>("reset"),
  camera: byId<HTMLButtonElement>("camera"),
  technical: byId<HTMLButtonElement>("technical"),
};

const scene = new THREE.Scene();
scene.background = new THREE.Color("#9fc7cd");
scene.fog = new THREE.Fog("#9fc7cd", 28, 90);

const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: "high-performance" });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.1;

const chaseCamera = new THREE.PerspectiveCamera(64, 1, 0.05, 300);
const fpvCamera = new THREE.PerspectiveCamera(92, 1, 0.03, 300);
const spectatorCamera = new THREE.PerspectiveCamera(56, 1, 0.05, 300);
spectatorCamera.position.set(14, 12, 18);
spectatorCamera.lookAt(0, 0, 0);
let cameraMode: "chase" | "fpv" | "spectator" = "chase";

// Renderer basis: Three is Y-up, FlightStack is Z-up.  This single adapter
// maps FlightStack [x, y, z] to render [x, z, -y] and is the only UI frame
// conversion; all physics stays canonical on the server.
const flightToRenderBasis = new THREE.Quaternion().setFromAxisAngle(
  new THREE.Vector3(1, 0, 0),
  -Math.PI / 2,
);
const renderToFlightBasis = flightToRenderBasis.clone().invert();

const flightVectorToRender = (vector: Vec3): THREE.Vector3 =>
  new THREE.Vector3(vector[0], vector[2], -vector[1]);

const flightQuaternionToRender = (quaternion: QuatWxyz): THREE.Quaternion => {
  const flight = new THREE.Quaternion(quaternion[1], quaternion[2], quaternion[3], quaternion[0]);
  return flightToRenderBasis.clone().multiply(flight).multiply(renderToFlightBasis);
};

const worldGroup = new THREE.Group();
scene.add(worldGroup);

const sun = new THREE.DirectionalLight("#fff1cf", 3.1);
sun.position.set(-18, 28, 14);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
sun.shadow.camera.left = -35;
sun.shadow.camera.right = 35;
sun.shadow.camera.top = 35;
sun.shadow.camera.bottom = -35;
scene.add(sun);
scene.add(new THREE.HemisphereLight("#c8f1fa", "#254241", 2.2));

const ground = new THREE.Mesh(
  new THREE.PlaneGeometry(150, 150),
  new THREE.MeshStandardMaterial({ color: "#1f3d40", roughness: 0.86, metalness: 0.08 }),
);
ground.receiveShadow = true;
ground.rotation.x = -Math.PI / 2;
worldGroup.add(ground);

const grid = new THREE.GridHelper(80, 80, "#4b7778", "#35585a");
grid.position.y = 0.008;
worldGroup.add(grid);

const pad = new THREE.Mesh(
  new THREE.CylinderGeometry(1.35, 1.55, 0.08, 48),
  new THREE.MeshStandardMaterial({ color: "#294f50", metalness: 0.45, roughness: 0.5 }),
);
pad.position.set(0, 0.04, 0);
pad.receiveShadow = true;
worldGroup.add(pad);

const ringMaterial = new THREE.MeshStandardMaterial({
  color: "#e0a251",
  emissive: "#6d3512",
  emissiveIntensity: 0.6,
  roughness: 0.35,
  metalness: 0.45,
});
const alternateRingMaterial = new THREE.MeshStandardMaterial({
  color: "#67c7c2",
  emissive: "#154a4b",
  emissiveIntensity: 0.8,
  roughness: 0.3,
  metalness: 0.38,
});

const gateGroup = new THREE.Group();
worldGroup.add(gateGroup);
let track: GateWire[] = [];

const addGate = (gate: GateWire, index: number): void => {
  const group = new THREE.Group();
  const material = index % 2 === 0 ? ringMaterial : alternateRingMaterial;
  const thickness = 0.13;
  const width = gate.half_width_m * 2;
  const height = gate.half_height_m * 2;
  const verticalGeometry = new THREE.BoxGeometry(thickness, height + thickness * 2, thickness);
  const horizontalGeometry = new THREE.BoxGeometry(width + thickness * 2, thickness, thickness);
  const sides = [
    [-width / 2, 0, 0],
    [width / 2, 0, 0],
  ] as const;
  for (const position of sides) {
    const mesh = new THREE.Mesh(verticalGeometry, material);
    mesh.position.set(...position);
    mesh.castShadow = true;
    group.add(mesh);
  }
  for (const y of [-height / 2, height / 2]) {
    const mesh = new THREE.Mesh(horizontalGeometry, material);
    mesh.position.set(0, y, 0);
    mesh.castShadow = true;
    group.add(mesh);
  }
  const marker = new THREE.Mesh(
    new THREE.BoxGeometry(width * 0.28, 0.035, 0.035),
    new THREE.MeshBasicMaterial({ color: "#eaf9ee" }),
  );
  marker.position.set(0, height / 2 + 0.05, 0.01);
  group.add(marker);
  const right = flightVectorToRender(gate.right_world).normalize();
  const up = flightVectorToRender(gate.up_world).normalize();
  const normal = flightVectorToRender(gate.normal_world).normalize();
  const basis = new THREE.Matrix4().makeBasis(right, up, normal);
  group.quaternion.setFromRotationMatrix(basis);
  group.position.copy(flightVectorToRender(gate.center_world_m));
  gateGroup.add(group);
};

const setTrack = (newTrack: GateWire[]): void => {
  if (track.length === newTrack.length && track.every((gate, index) => gate.id === newTrack[index]?.id)) return;
  track = newTrack;
  gateGroup.clear();
  track.forEach(addGate);
};

const createDrone = (): { group: THREE.Group; props: THREE.Mesh[] } => {
  const group = new THREE.Group();
  const bodyMaterial = new THREE.MeshStandardMaterial({ color: "#172629", metalness: 0.72, roughness: 0.34 });
  const accentMaterial = new THREE.MeshStandardMaterial({ color: "#53b8aa", emissive: "#0c3934", emissiveIntensity: 1.4 });
  const propMaterial = new THREE.MeshStandardMaterial({ color: "#262f30", transparent: true, opacity: 0.72, roughness: 0.28 });
  const props: THREE.Mesh[] = [];

  const core = new THREE.Mesh(new THREE.BoxGeometry(0.2, 0.075, 0.16), bodyMaterial);
  core.castShadow = true;
  group.add(core);
  const camera = new THREE.Mesh(new THREE.BoxGeometry(0.08, 0.055, 0.07), accentMaterial);
  camera.position.set(0.13, -0.005, 0);
  camera.castShadow = true;
  group.add(camera);

  const motorPositions: Array<[number, number, number]> = [
    [0.18, 0, -0.18], [0.18, 0, 0.18], [-0.18, 0, -0.18], [-0.18, 0, 0.18],
  ];
  for (const position of motorPositions) {
    const arm = new THREE.Mesh(new THREE.BoxGeometry(0.37, 0.026, 0.028), bodyMaterial);
    arm.rotation.y = Math.atan2(position[2], position[0]);
    arm.castShadow = true;
    group.add(arm);
    const motor = new THREE.Mesh(new THREE.CylinderGeometry(0.036, 0.036, 0.055, 12), bodyMaterial);
    motor.position.set(...position);
    motor.castShadow = true;
    group.add(motor);
    const prop = new THREE.Mesh(new THREE.CylinderGeometry(0.14, 0.14, 0.007, 28), propMaterial);
    prop.position.set(position[0], 0.032, position[2]);
    prop.scale.set(1.25, 1, 0.22);
    prop.castShadow = true;
    group.add(prop);
    props.push(prop);
  }
  const rearLight = new THREE.PointLight("#59e4d0", 0.4, 1.5);
  rearLight.position.set(-0.11, 0.03, 0);
  group.add(rearLight);
  group.castShadow = true;
  return { group, props };
};

const drone = createDrone();
drone.group.position.set(0, 1, 0);
worldGroup.add(drone.group);
const bodyAxes = new THREE.AxesHelper(0.7);
bodyAxes.visible = false;
drone.group.add(bodyAxes);
const velocityArrow = new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), drone.group.position, 0.01, 0x9ef0d9);
velocityArrow.visible = false;
worldGroup.add(velocityArrow);

for (let index = 0; index < 4; index += 1) {
  const motor = document.createElement("div");
  motor.className = "motor";
  motor.innerHTML = `<span></span><label>M${index + 1}</label>`;
  dom.motors.append(motor);
}

let currentState: StateWire = {
  sim_time_s: 0,
  position_world_m: [0, 0, 1],
  velocity_world_m_s: [0, 0, 0],
  q_body_to_world_wxyz: [1, 0, 0, 0],
  body_rate_rad_s: [0, 0, 0],
  motor_thrust_n: [0, 0, 0, 0],
};
let technicalMode = false;
let activePilot: "human" | "classical" | "learned" = "human";

const input = { roll: 0, pitch: 0, yaw: 0, throttle: 0 };
const pressed = new Set<string>();
let toastTimeout = 0;

const showToast = (message: string): void => {
  dom.toast.textContent = message;
  dom.toast.classList.add("visible");
  window.clearTimeout(toastTimeout);
  toastTimeout = window.setTimeout(() => dom.toast.classList.remove("visible"), 2600);
};

const updateConnection = (connected: boolean): void => {
  dom.connection.textContent = connected ? "SIM LINK ONLINE" : "SIM LINK LOST";
  dom.connectionParent.classList.toggle("online", connected);
};

const updateTelemetry = (message: StateMessage): void => {
  currentState = message.state;
  if (message.track) setTrack(message.track);
  const velocity = flightVectorToRender(message.state.velocity_world_m_s);
  const speed = velocity.length();
  dom.speed.textContent = speed.toFixed(1);
  dom.altitude.textContent = Math.max(0, message.state.position_world_m[2]).toFixed(1);
  dom.gate.textContent = message.race.status === "finished"
    ? "FINISHED"
    : message.race.status === "preflight"
      ? "READY"
      : `GATE ${message.race.next_gate + 1}`;
  dom.lap.textContent = `LAP ${message.race.lap} // ${message.race.lap_time_s.toFixed(1)} S`;
  dom.rateX.textContent = message.state.body_rate_rad_s[0].toFixed(2);
  dom.rateY.textContent = message.state.body_rate_rad_s[1].toFixed(2);
  dom.rateZ.textContent = message.state.body_rate_rad_s[2].toFixed(2);
  const totalThrust = message.state.motor_thrust_n.reduce((sum, value) => sum + value, 0);
  dom.thrust.textContent = `${totalThrust.toFixed(1)} N`;
  document.querySelectorAll<HTMLDivElement>(".motor").forEach((motor, index) => {
    const percentage = Math.min(
      100,
      (message.state.motor_thrust_n[index] / message.vehicle.motor_max_thrust_n) * 100,
    );
    const fill = motor.querySelector("span");
    if (fill) fill.style.height = `${percentage}%`;
  });
  document.querySelectorAll<HTMLButtonElement>(".mode").forEach((button) => {
    button.classList.toggle("active", button.dataset.pilot === message.pilot);
  });
};

const drawMinimap = (): void => {
  const context = minimapContext;
  const width = minimap.width;
  const height = minimap.height;
  context.clearRect(0, 0, width, height);
  context.fillStyle = "#0e1a1d";
  context.fillRect(0, 0, width, height);
  context.strokeStyle = "#29464a";
  context.lineWidth = 1;
  for (let x = 10; x < width; x += 20) {
    context.beginPath(); context.moveTo(x, 0); context.lineTo(x, height); context.stroke();
  }
  for (let y = 10; y < height; y += 20) {
    context.beginPath(); context.moveTo(0, y); context.lineTo(width, y); context.stroke();
  }
  const points = [...track.map((gate) => gate.center_world_m), currentState.position_world_m];
  const extent = Math.max(10, ...points.flatMap((point) => [Math.abs(point[0]), Math.abs(point[1])])) + 2;
  const project = (point: Vec3): [number, number] => [
    width / 2 + (point[0] / extent) * (width * 0.42),
    height / 2 - (point[1] / extent) * (height * 0.42),
  ];
  if (track.length > 1) {
    context.strokeStyle = "#537f80";
    context.lineWidth = 1.5;
    context.beginPath();
    track.forEach((gate, index) => {
      const [x, y] = project(gate.center_world_m);
      if (index === 0) context.moveTo(x, y); else context.lineTo(x, y);
    });
    context.stroke();
  }
  track.forEach((gate, index) => {
    const [x, y] = project(gate.center_world_m);
    context.fillStyle = index % 2 === 0 ? "#e5aa58" : "#6ed1c7";
    context.fillRect(x - 3, y - 3, 6, 6);
  });
  const [x, y] = project(currentState.position_world_m);
  const heading = new THREE.Vector3(1, 0, 0).applyQuaternion(flightQuaternionToRender(currentState.q_body_to_world_wxyz));
  const planarAngle = Math.atan2(-heading.z, heading.x);
  context.save();
  context.translate(x, y);
  context.rotate(-planarAngle);
  context.fillStyle = "#f4fff9";
  context.beginPath();
  context.moveTo(7, 0); context.lineTo(-5, -4); context.lineTo(-3, 0); context.lineTo(-5, 4); context.closePath(); context.fill();
  context.restore();
};

const applyStateToDrone = (elapsedSeconds: number): void => {
  const nextPosition = flightVectorToRender(currentState.position_world_m);
  const nextQuaternion = flightQuaternionToRender(currentState.q_body_to_world_wxyz);
  drone.group.position.lerp(nextPosition, 0.25);
  drone.group.quaternion.slerp(nextQuaternion, 0.28);
  drone.props.forEach((prop, index) => {
    const direction = index % 2 === 0 ? 1 : -1;
    prop.rotation.y += direction * (20 + currentState.motor_thrust_n[index] * 24) * elapsedSeconds;
  });
  const velocity = flightVectorToRender(currentState.velocity_world_m_s);
  const speed = velocity.length();
  velocityArrow.position.copy(drone.group.position);
  if (speed * speed > 0.002) {
    velocityArrow.setDirection(velocity.normalize());
    velocityArrow.setLength(Math.min(2.2, speed * 0.2));
  }
};

const updateCamera = (): void => {
  const forward = new THREE.Vector3(1, 0, 0).applyQuaternion(drone.group.quaternion);
  const up = new THREE.Vector3(0, 1, 0).applyQuaternion(drone.group.quaternion);
  if (cameraMode === "chase") {
    const desired = drone.group.position.clone().addScaledVector(forward, -4.6).addScaledVector(up, 1.8);
    chaseCamera.position.lerp(desired, 0.08);
    chaseCamera.lookAt(drone.group.position.clone().addScaledVector(forward, 2.4));
  } else if (cameraMode === "fpv") {
    fpvCamera.position.copy(drone.group.position).addScaledVector(forward, 0.17).addScaledVector(up, 0.045);
    fpvCamera.up.copy(up);
    fpvCamera.lookAt(drone.group.position.clone().addScaledVector(forward, 20));
  }
};

const activeCamera = (): THREE.PerspectiveCamera => {
  if (cameraMode === "fpv") return fpvCamera;
  if (cameraMode === "spectator") return spectatorCamera;
  return chaseCamera;
};

const resize = (): void => {
  const width = window.innerWidth;
  const height = window.innerHeight;
  renderer.setSize(width, height, false);
  for (const camera of [chaseCamera, fpvCamera, spectatorCamera]) {
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
  }
};
window.addEventListener("resize", resize);
resize();

let socket: WebSocket | undefined;
let reconnectTimer: number | undefined;
let lastInputSentAt = 0;

const send = (message: object): void => {
  if (socket?.readyState === WebSocket.OPEN) socket.send(JSON.stringify(message));
};

const connect = (): void => {
  if (socket?.readyState === WebSocket.OPEN || socket?.readyState === WebSocket.CONNECTING) return;
  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  socket = new WebSocket(`${protocol}://${window.location.host}/ws`);
  socket.addEventListener("open", () => {
    updateConnection(true);
    showToast("Connected to fixed-step FlightStack runtime");
    send({ type: "set_pilot", pilot: activePilot });
  });
  socket.addEventListener("message", (event) => {
    const message = JSON.parse(String(event.data)) as ServerMessage;
    if (message.type === "state") updateTelemetry(message);
    else if (message.type === "notice") showToast(message.message);
  });
  socket.addEventListener("close", () => {
    updateConnection(false);
    window.clearTimeout(reconnectTimer);
    reconnectTimer = window.setTimeout(connect, 1200);
  });
  socket.addEventListener("error", () => socket?.close());
};

const updateInput = (elapsedSeconds: number): void => {
  const target = {
    roll: (pressed.has("KeyD") ? 1 : 0) - (pressed.has("KeyA") ? 1 : 0),
    pitch: (pressed.has("KeyW") ? 1 : 0) - (pressed.has("KeyS") ? 1 : 0),
    yaw: (pressed.has("KeyE") ? 1 : 0) - (pressed.has("KeyQ") ? 1 : 0),
  };
  const gamepad = navigator.getGamepads().find((candidate) => candidate?.connected);
  if (gamepad) {
    target.roll = Math.abs(gamepad.axes[2] ?? 0) > 0.08 ? gamepad.axes[2] ?? 0 : target.roll;
    target.pitch = Math.abs(gamepad.axes[3] ?? 0) > 0.08 ? -(gamepad.axes[3] ?? 0) : target.pitch;
    target.yaw = Math.abs(gamepad.axes[0] ?? 0) > 0.08 ? gamepad.axes[0] ?? 0 : target.yaw;
    const throttleAxis = gamepad.axes[1] ?? 1;
    input.throttle = THREE.MathUtils.clamp((1 - throttleAxis) * 0.5, 0, 1);
  } else {
    input.throttle = THREE.MathUtils.clamp(
      input.throttle + ((pressed.has("Space") ? 1 : 0) - (pressed.has("ControlLeft") ? 1 : 0)) * elapsedSeconds * 0.42,
      0,
      1,
    );
  }
  input.roll = THREE.MathUtils.damp(input.roll, target.roll, 18, elapsedSeconds);
  input.pitch = THREE.MathUtils.damp(input.pitch, target.pitch, 18, elapsedSeconds);
  input.yaw = THREE.MathUtils.damp(input.yaw, target.yaw, 18, elapsedSeconds);
};

window.addEventListener("keydown", (event) => {
  if (["Space", "ControlLeft", "KeyW", "KeyS", "KeyA", "KeyD", "KeyQ", "KeyE"].includes(event.code)) {
    event.preventDefault();
    pressed.add(event.code);
  }
  if (!event.repeat) {
    if (event.code === "KeyR") send({ type: "reset" });
    if (event.code === "KeyC") cycleCamera();
    if (event.code === "KeyT") toggleTechnical();
    if (event.code === "Digit1") selectPilot("human");
    if (event.code === "Digit2") selectPilot("classical");
    if (event.code === "Digit3") selectPilot("learned");
  }
});
window.addEventListener("keyup", (event) => pressed.delete(event.code));

const cycleCamera = (): void => {
  cameraMode = cameraMode === "chase" ? "fpv" : cameraMode === "fpv" ? "spectator" : "chase";
  dom.camera.textContent = `${cameraMode.toUpperCase()} C`;
  showToast(`${cameraMode.toUpperCase()} camera`);
};

const toggleTechnical = (): void => {
  technicalMode = !technicalMode;
  dom.technicalDrawer.classList.toggle("open", technicalMode);
  bodyAxes.visible = technicalMode;
  velocityArrow.visible = technicalMode;
  dom.technical.classList.toggle("active", technicalMode);
};

const selectPilot = (pilot: "human" | "classical" | "learned"): void => {
  activePilot = pilot;
  send({ type: "set_pilot", pilot });
  showToast(`${pilot.toUpperCase()} pilot requested`);
};

dom.reset.addEventListener("click", () => send({ type: "reset" }));
dom.camera.addEventListener("click", cycleCamera);
dom.technical.addEventListener("click", toggleTechnical);
document.querySelectorAll<HTMLButtonElement>(".mode").forEach((button) => {
  button.addEventListener("click", () => selectPilot(button.dataset.pilot as typeof activePilot));
});

let lastFrameAt = performance.now();
const render = (now: number): void => {
  const elapsedSeconds = Math.min(0.05, (now - lastFrameAt) / 1000);
  lastFrameAt = now;
  updateInput(elapsedSeconds);
  applyStateToDrone(elapsedSeconds);
  updateCamera();
  drawMinimap();
  if (now - lastInputSentAt > 33) {
    const message: InputMessage = { type: "manual_input", ...input };
    send(message);
    lastInputSentAt = now;
  }
  renderer.render(scene, activeCamera());
  requestAnimationFrame(render);
};

connect();
requestAnimationFrame(render);