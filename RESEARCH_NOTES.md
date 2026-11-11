# Research notes: tendon-driven robotic tail (master's project)

Working notes on possible novel directions for the thesis, what the simulator
already does, and what is still missing. **Not a literature review** - the
"gap" below comes from a quick search (3 web queries) and must be checked
properly before claiming novelty (Google Scholar, IEEE Xplore, recent
ICRA / IROS / RoboSoft proceedings, and expressive-robot / animatronics work).

---

## 1. The hardware, in short

- Chain of 3D-printed vertebrae (Part 8, double fork) joined by rings (Part 5).
  Each ring + the two forks pinned to it = one universal joint (pitch + yaw).
- First ring fixed to the base. The **end ring is free**: it hangs on one pin
  axis of the last vertebra and only the cables hold it.
- 4 cables through holes in the rings at r = 30 mm, on the diagonals
  (45/135/225/315 deg), tied off at the end ring.
- Driven by **spool wheels = length (position) control**. No torque/force
  sensing on the real tail.

## 2. What exists (from a quick search)

| Work | What it is | Relevance |
|---|---|---|
| TALE-teller [1] | Tendon-actuated robotic tail testbed, routing modelled on mammalian tail muscles; for studying tail function | Closest hardware |
| Toronto TDCR in MuJoCo [2] | Tendon-driven continuum robots simulated in MuJoCo; imitation-learned policies moved zero-shot to real hardware | Closest sim + sim-to-real setup |
| Learning-based control for TDCRs [3] | Deep RL (DDPG) tuning a controller for tendon-driven arms; focus on sim-to-real | Learned control, but arms and positioning |
| Digital twins for tendon-driven swimmers [4] | RL agent (SAC) learns steering with tail motions in a digital twin | Learned tail use, but for swimming |
| Tensegrity dolphin [5] | Cable-length control gives smooth, wave-like body motion | Same "length drive" principle |

**Apparent gap:** nothing found on *lifelike, expressive* motion for a
tendon-driven tail, or on doing it with *spool-only position control and no
force sensing*.

## 3. Possible novel directions

1. **Expressive tail motion learned from animals** *(candidate headline)*
   - Extract dog tail motion from video, retarget it to the 4-cable tail
     through a calibrated simulator, run it on the hardware.
   - Evaluation: user/perception study - does the robot's motion read as
     happy / alert / anxious, compared with a plain sine-wave wag?

2. **Vision-calibrated digital twin of a friction-dominated, spool-driven tail**
   *(method chapter / foundation)*
   - Calibrate friction, stiffness and end-ring behaviour from spool angles
     plus a camera tracking markers on the rings. No force sensors.
   - Also documents and verifies real mechanics of this design seen in the
     simulator (see section 5).

3. **Time-varying stiffness for organic motion** *(the ingredient)*
   - Vary pretension through a motion (loose mid-swing, firm at the ends) for
     whip-like, animal-like movement a constant-stiffness tail can't do.
   - Learn or optimise the tension timing; compare against constant tension.

4. **Exploiting dynamics for travelling waves** *(riskier, more research-y)*
   - 4 cables, 16 joints: can learned open-loop commands use inertia,
     gravity and friction to produce waves running along the tail?

**Suggested shape for a master's:** (2) as the method, (1) as the headline
result, (3) as the thing that makes the motion organic. Agree the direction
with the supervisor before committing.

## 4. Where machine learning fits

1. **The simulator must match the real tail first.** A policy trained on a
   mismatched sim learns the sim's mistakes. Sim-to-real is where most of the
   effort in [2]-[4] goes. Current friction values are guesses.
2. **Organic motion often doesn't need ML.** Animatronics typically uses
   hand-designed animation curves (timing, ease-in/out, overlap,
   follow-through). Here: a motion generator varying bend *and* tension over
   time, played back on the spools.
3. **Light "learning" as the middle step:** take a reference motion (e.g. tail
   tip traced from a dog video) and let an optimiser fit the motion
   generator's parameters (amplitude, frequency, up offset, tension timing)
   until the sim matches. Cheap, explainable, runs on a laptop.
4. **Full reinforcement learning** later, if the tail should *react*
   (balance, respond to touch, adapt to load).

## 5. Mechanics found in the simulator (to verify on hardware)

These came out of the simulator with the current, partly guessed parameters.
They are predictions, not measurements.

- **Equal tension cancels:** every full-length cable gives the same torque at
  every joint (0.0212 N*m per N), so equal tension on all four gives zero
  bending torque. Gravity needs very different torque at each joint
  (0.28 N*m at the base down to ~0 at the end), so no tension set holds an
  ideal frictionless tail straight. It sags into an S-shape.
- **Cables fix only the total bend:** the four cable lengths fix the *sum* of
  pitch angles and the *sum* of yaw angles, not how the bend is shared between
  joints. Gravity and friction decide the distribution.
- **Bending goes to the weakest joint:** with friction locking, the
  least-loaded joint slips first and takes most of the motion.
- **The free end ring tips:** tilting the end ring on its single pin axis is
  the cheapest way to change cable lengths, so it often moves before the
  tail bends.
- **Pretension + friction can hold the tail straight:** with friction that
  grows with cable tension, reeling all cables in ~2 mm held the tail
  straight out in simulation. Depends entirely on the (unmeasured) friction
  coefficient.
- **Single-cable reel-in barely moves the tail** in length drive: the
  opposite cable resists. Bending needs one side in and the other side out
  (antagonistic pairs).
- **Bottom cables carry less load** when lifting against gravity (3-10 N vs
  12-20 N for the top pair in tests) but are needed for sideways motion,
  stiffness and controlled lowering.

## 6. What the simulator already provides

- Real geometry (STLs) and measured masses; inertias computed from the meshes.
- Length-driven spools (stiff, force-limited position servo per cable, pull
  only) with measured cable tension; optional tension drive.
- Mixed controls (Up/down, Side, Tighten all) plus per-cable trim.
- Tension-dependent stick-slip joint friction (placeholder coefficient).
- Base force/torque sensor; headless runs; CSV logging and plots.
- Config switches for end ring free/fixed and first ring locked.

## 7. What's missing

- [ ] **Calibration against the hardware:** photos at known spool positions
      (side view, ruler/grid behind), or better, tracked markers on the rings.
      Fit `friction_per_newton` (and stiffness, if any) to them.
- [ ] Spool radius -> commands in spool degrees, and spool torque output
      (torque = tension x radius) for motor sizing.
- [ ] Motion generator (wag etc.) with time-varying bend and tension.
- [ ] Export of spool command time series to drive the real tail.
- [ ] Video -> tail kinematics pipeline (for direction 1).
- [ ] Proper literature review to confirm the gap.

## Sources

1. TALE-teller: Tendon-Actuated Linked Element Robotic Testbed for Investigating
   Tail Functions - https://arxiv.org/html/2410.21445v1
2. Do Rigid-Body Simulators Dream of Soft Robots? Learning Contact-Rich
   Manipulation for Tendon-Driven Continuum Robots - https://arxiv.org/pdf/2606.22397
3. Learning-based control for tendon-driven continuum robotic arms (Frontiers
   in Robotics and AI, 2025) -
   https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2025.1488869/abstract
4. Simple Models, Real Swimming: Digital Twins for Tendon-Driven Underwater
   Robots - https://arxiv.org/pdf/2602.23283
5. An Untethered Bioinspired Robotic Tensegrity Dolphin with Multi-Flexibility
   Design for Aquatic Locomotion - https://arxiv.org/html/2411.00347v1
