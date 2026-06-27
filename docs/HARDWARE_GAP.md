# Hardware gap memo

A brief note flagging items that need to be added to the BOM before this controller can be deployed on a real greenhouse. Filed alongside the methodology document because it is not strictly a model question, but it is a deployment-blocking one.

## What the Rapidcircuitry quote covers

The Rapidcircuitry / Fyllo quotation referenced in the proposal covers **sensor nodes only** — soil moisture (2 depths), soil temperature, and (per the headers, but not delivered in the data) air temp/RH, VPD, leaf wetness, light, ET. A typical quote of this kind ships:

- N × Fyllo-compatible sensor nodes
- 1 × gateway / data logger
- Cloud dashboard subscription

It does **not** typically ship:

- Solenoid valves (the thing this controller actuates)
- Flow meters (the thing that lets us close the loop on "how much water did I actually deliver")
- Pressure regulator / manifold
- A microcontroller capable of switching 24 V AC to the solenoid

## What you need to add

| Item | Suggested part | Per plot? | Notes |
|---|---|---|---|
| Latching solenoid valve | Rain Bird 100-DV (24 V DC, 1″) or Hunter PGV-101 | 1× per plot | Latching saves continuous holding current; important on battery/solar |
| Flow meter | YF-S201 (Hall, 1–30 L/min) or Seeed G1/2 | 1× per plot | Pulse output → ESP32 ISR |
| Pressure regulator | Senninger PRL 1.0 bar | 1× at manifold | Drip lines need 1.0–1.5 bar, not mains pressure |
| Solenoid driver | DRV8871 or ULN2803 + flyback | 1× per node | If using latching, needs H-bridge |
| MCU + radio | ESP32-WROOM or LoRa node | 1× per plot | Speaks to Fyllo gateway or directly to MQTT |
| Spare sensor node | Same as primary | 1× cold spare | Sensor nodes do fail — having one means a 1-hour swap, not a 3-day reorder |
| 24 V DC PSU / solar | 24 V DC 3 A | 1× per zone | Solenoids draw briefly at pulse onset |

Indicative cost: ~₹3 500 – ₹6 000 per plot for the actuator-side hardware, on top of the sensor-side quote.

## Why this matters for the model

- Without a **flow meter**, the system has no closed-loop feedback on "how much water did I actually deliver". The model assumes pulse_minutes × flow_rate is accurate; in practice, drip emitters clog and pressure varies. Flow meter feedback can be folded into the state vector.
- Without a **solenoid**, there is no actuator — the controller has nothing to do. The current code emits `pulse_minutes`; with no solenoid this is a logged recommendation only, which is fine for Phase 2 (shadow mode) but blocks Phase 3.
- Without a **pressure regulator**, drip flow rate is mains-pressure dependent and varies across the day, which makes calibration of "minutes → mm delivered" unreliable.

## Sensor-side recommendation

The current Fyllo data has air-temp / RH / VPD / light / leaf-wetness columns present but empty. Two possibilities:

1. The sensors were ordered but failed to log → contact Fyllo support, recover the data.
2. The sensors were not in the BOM → add an SHT35 (air temp + RH, ~₹800) and an Adafruit photodiode or PAR sensor (~₹2 000) per plot.

Either way, getting these channels populated would let the simulator move from a Hargreaves surrogate to full Penman–Monteith ET₀, which is the single biggest source of calibration residual today.

## Sensor node count

The current proposal quotes 4 sensor nodes. The trial has 3 plots × 2 depths of soil moisture per plot = 6 channels just for moisture, plus soil temperature per plot, plus (ideally) one air-side sensor per plot. A single Fyllo node typically supports 2–4 soil channels; so **4 nodes is tight**. Recommend 5–6 nodes (one per plot + one shared air-side + one cold spare).
