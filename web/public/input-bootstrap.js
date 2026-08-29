// FlightStack default playable mode is intentionally keyboard-only.
//
// The previous client automatically consumed the first browser-reported gamepad.
// On Windows that can include stale/virtual HID devices whose throttle axis is not
// centered, which silently overrides keyboard input and can command a climb.
// A future explicit Gamepad mode can opt back into device input with calibration.
(() => {
  const noGamepads = () => [];

  try {
    Object.defineProperty(navigator, "getGamepads", {
      configurable: true,
      value: noGamepads,
    });
  } catch {
    try {
      Navigator.prototype.getGamepads = noGamepads;
    } catch {
      // If a browser refuses both overrides, keyboard behavior still works;
      // the main client will simply see the browser's normal gamepad list.
    }
  }

  let firstSpaceStartsTakeoff = true;
  let suppressFirstSpaceRepeats = false;

  const rearmTakeoffKey = () => {
    firstSpaceStartsTakeoff = true;
    suppressFirstSpaceRepeats = false;
  };

  window.addEventListener(
    "keydown",
    (event) => {
      // Reset and returning to Human mode both create a new grounded run.
      if (!event.repeat && (event.code === "KeyR" || event.code === "Digit1")) {
        rearmTakeoffKey();
        return;
      }

      if (event.code !== "Space") return;

      // While the physical first Space is still held, block OS key-repeat from
      // turning the takeoff press into a continuous climb command.
      if (suppressFirstSpaceRepeats) {
        event.preventDefault();
        event.stopImmediatePropagation();
        return;
      }

      if (!firstSpaceStartsTakeoff || event.repeat) return;

      firstSpaceStartsTakeoff = false;
      suppressFirstSpaceRepeats = true;

      // Let the initial keydown reach main.ts long enough to send the existing
      // takeoff trigger, then synthesize a keyup. The server keeps the automatic
      // takeoff active until it settles at hover altitude.
      window.setTimeout(() => {
        window.dispatchEvent(
          new KeyboardEvent("keyup", {
            bubbles: true,
            code: "Space",
            key: " ",
          }),
        );
      }, 90);
    },
    true,
  );

  window.addEventListener(
    "keyup",
    (event) => {
      if (event.code === "Space" && event.isTrusted) {
        suppressFirstSpaceRepeats = false;
      }
    },
    true,
  );

  window.addEventListener("DOMContentLoaded", () => {
    document.getElementById("reset")?.addEventListener("click", rearmTakeoffKey, true);
    document
      .querySelector('[data-pilot="human"]')
      ?.addEventListener("click", rearmTakeoffKey, true);
  });
})();
